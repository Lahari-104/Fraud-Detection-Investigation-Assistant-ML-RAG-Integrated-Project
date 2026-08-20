"""
Phase 3 -- retrieval pipeline.

Two pieces:
  1. A query builder that turns a flagged transaction (its amount, derived hour,
     probability, and SHAP-attributed anomaly components) into a natural-language
     query. This is the single bridge between the ML half and the RAG half.
  2. A retriever with a SWAPPABLE embedding backend:
        - preferred: sentence-transformers embeddings + FAISS inner-product index
        - fallback:  TF-IDF vectors + cosine similarity (pure scikit-learn)
     The backend is auto-selected. If sentence-transformers imports AND its model
     weights can actually be loaded (i.e. you are online / have them cached), the
     neural backend is used. Otherwise it silently falls back to TF-IDF so the
     pipeline is demonstrable offline. Both expose the same .search() signature,
     so Phase 4/5/6 are backend-agnostic.

Run locally with internet once and the neural backend takes over automatically;
no code changes required.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "knowledge_base")
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# Document loading
# --------------------------------------------------------------------------- #
@dataclass
class KBDocument:
    doc_id: str
    kind: str          # "policy" | "case"
    title: str
    body: str
    path: str

    @property
    def text(self) -> str:
        # what actually gets embedded: title + body, not the metadata header
        return f"{self.title}. {self.body}"

    def display(self, max_chars: int = 240) -> str:
        b = self.body if len(self.body) <= max_chars else self.body[:max_chars] + "..."
        return f"[{self.doc_id}] {self.title} ({self.kind})\n    {b}"


def load_kb(kb_dir: str = KB_DIR) -> list[KBDocument]:
    docs = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
        with open(path) as f:
            raw = f.read()
        meta = dict(re.findall(r"^# (\w+): (.+)$", raw, flags=re.MULTILINE))
        body = re.sub(r"^#.*$", "", raw, flags=re.MULTILINE).strip()
        docs.append(KBDocument(
            doc_id=meta.get("id", os.path.basename(path)),
            kind=meta.get("type", "unknown"),
            title=meta.get("title", ""),
            body=body,
            path=path,
        ))
    if not docs:
        raise FileNotFoundError(
            f"No .txt documents in {kb_dir}. Run rag/build_knowledge_base.py first.")
    return docs


# --------------------------------------------------------------------------- #
# Query builder: transaction features -> natural-language query
# --------------------------------------------------------------------------- #
# Human-readable hints for the anonymised components, used only to enrich the
# query text so it lexically/semantically overlaps the KB. These are directional
# facts measured in Phase 2 prep, not fabricated meaning.
_COMPONENT_HINT = {
    "V17": "strong anomaly component", "V14": "strong anomaly component",
    "V12": "strong anomaly component", "V10": "strong anomaly component",
    "V16": "anomaly component", "V3": "anomaly component",
    "V7": "anomaly component", "V11": "anomaly component",
}
_DOMINANT = {"V17", "V14", "V12", "V10"}


def build_query(amount: float, hour: int, probability: float,
                top_components: list[tuple[str, float]] | None = None) -> str:
    """Compose a retrieval query from derivable transaction features."""
    parts: list[str] = []

    # amount band
    if amount == 0:
        parts.append("zero-amount authorization, possible verification probe")
    elif amount <= 1:
        parts.append(f"micro-charge of ${amount:.2f}, possible card testing")
    elif amount <= 5:
        parts.append(f"very small charge of ${amount:.2f}")
    elif amount >= 1000:
        parts.append(f"high-value transaction of ${amount:,.2f}, manual-hold band")
    elif amount >= 200:
        parts.append(f"elevated-value transaction of ${amount:,.2f}")
    else:
        parts.append(f"mid-value transaction of ${amount:,.2f}")

    # time window
    if 0 <= hour < 6:
        parts.append(f"in the overnight elevated-risk window at {hour:02d}:00")
    else:
        parts.append(f"during daytime hours at {hour:02d}:00")

    # probability band
    if probability >= 0.90:
        parts.append("high model probability")
    elif probability >= 0.50:
        parts.append("moderate model probability")
    else:
        parts.append("borderline model probability")

    # anomaly profile from attribution
    if top_components:
        names = [c for c, _ in top_components[:4]]
        dominant = [c for c in names if c in _DOMINANT]
        if dominant:
            parts.append("dominant fraud anomaly components " + ", ".join(dominant))
        else:
            parts.append("anomaly components " + ", ".join(names) +
                         " without the dominant fraud signature")

    return "Flagged transaction: " + "; ".join(parts) + "."


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalResult:
    document: KBDocument
    score: float
    rank: int


class _TfidfBackend:
    name = "tfidf"

    def __init__(self, docs: list[KBDocument]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._cos = cosine_similarity
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        qv = self.vectorizer.transform([query])
        sims = self._cos(qv, self.matrix).ravel()
        order = np.argsort(sims)[::-1][:k]
        return [RetrievalResult(self.docs[i], float(sims[i]), r)
                for r, i in enumerate(order)]


class _NeuralBackend:
    name = "sentence-transformers+faiss"

    def __init__(self, docs: list[KBDocument], model):
        import faiss
        self.docs = docs
        self.model = model
        emb = model.encode([d.text for d in docs], normalize_embeddings=True,
                           show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine via normalized IP
        self.index.add(emb)

    def search(self, query: str, k: int) -> list[RetrievalResult]:
        qv = self.model.encode([query], normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")
        scores, idx = self.index.search(qv, k)
        return [RetrievalResult(self.docs[i], float(s), r)
                for r, (i, s) in enumerate(zip(idx[0], scores[0])) if i != -1]


def _try_neural(docs: list[KBDocument]):
    """Return a neural backend if weights can genuinely be loaded, else None."""
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        model = SentenceTransformer(ST_MODEL_NAME)          # downloads/loads weights
        _ = model.encode(["warmup"], show_progress_bar=False)
        return _NeuralBackend(docs, model)
    except Exception as e:
        print(f"  [retriever] sentence-transformers present but weights "
              f"unavailable ({type(e).__name__}); using TF-IDF fallback.")
        return None


# --------------------------------------------------------------------------- #
# Public retriever
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, kb_dir: str = KB_DIR, force_tfidf: bool = False):
        self.docs = load_kb(kb_dir)
        backend = None if force_tfidf else _try_neural(self.docs)
        self.backend = backend or _TfidfBackend(self.docs)
        self.backend_name = self.backend.name

    def search(self, query: str, k: int = 4,
               kind: str | None = None) -> list[RetrievalResult]:
        # over-fetch when filtering by kind so k results survive the filter
        raw = self.backend.search(query, k * 3 if kind else k)
        if kind:
            raw = [r for r in raw if r.document.kind == kind]
        return [RetrievalResult(r.document, r.score, i) for i, r in enumerate(raw[:k])]

    def retrieve_for_transaction(self, amount, hour, probability,
                                 top_components=None, k: int = 4):
        query = build_query(amount, hour, probability, top_components)
        return query, self.search(query, k=k)


if __name__ == "__main__":
    r = Retriever()
    print(f"Retriever backend: {r.backend_name}  ({len(r.docs)} documents)\n")

    scenarios = [
        ("micro card-test, overnight, high prob",
         dict(amount=0.79, hour=2, probability=0.97,
              top_components=[("V14", 6.7), ("V10", 2.5), ("V4", 1.6)])),
        ("high-value manual-hold band, moderate prob",
         dict(amount=1420.0, hour=15, probability=0.64,
              top_components=[("V4", 1.2), ("V11", 0.9)])),
        ("zero-amount probe",
         dict(amount=0.0, hour=11, probability=0.71,
              top_components=[("V12", 2.1), ("V17", 1.8)])),
    ]

    for label, feats in scenarios:
        query, results = r.retrieve_for_transaction(**feats, k=3)
        print("=" * 72)
        print(f"SCENARIO: {label}")
        print(f"QUERY: {query}")
        print("-" * 72)
        for res in results:
            print(f"  #{res.rank+1}  score={res.score:.3f}")
            print("  " + res.document.display().replace("\n", "\n  "))
        print()
