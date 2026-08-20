"""
Phase 4 -- generation pipeline.

Turns (transaction + ML flag + retrieved documents) into a plain-English,
evidence-backed explanation that cites specific policies and cases.

Grounding discipline (the whole point of RAG here):
  - The prompt instructs the model to use ONLY the retrieved documents and the
    supplied transaction facts, to cite every claim by document id, and to say
    so explicitly when the context does not support a conclusion.
  - After generation we EXTRACT the cited ids and VERIFY them against the ids
    actually retrieved. A citation to a document that was not in context is a
    hallucinated citation and is reported as such -- we don't trust the model's
    self-report, we check it.

Backend swap (same pattern as retrieval):
  - If GEMINI_API_KEY (or GOOGLE_API_KEY) is set and the google-genai SDK
    imports, real generation via Gemini.
  - Otherwise a deterministic offline stub composes an explanation by TEMPLATING
    the retrieved documents. The stub never invents facts -- it only quotes
    document ids and transaction values it was given -- so the end-to-end
    pipeline is demonstrable now and swaps to the real LLM when the key lands.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1500
REQUEST_TIMEOUT_S = 30  # cap the call so a slow/failing request falls back to stub
# API key is read from GEMINI_API_KEY (preferred) or GOOGLE_API_KEY, so either
# environment variable works.
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a fraud investigation assistant supporting a human analyst. \
You explain, in plain English, why a payment-fraud model flagged a transaction.

Strict rules:
1. Use ONLY the facts in the TRANSACTION section and the CONTEXT DOCUMENTS below. \
Do not use outside knowledge or invent details (no country, merchant, or \
cardholder facts -- they are not available).
2. Cite every substantive claim with the document id in square brackets, e.g. \
[POL-03] or [CASE-04]. Only cite documents that appear in CONTEXT DOCUMENTS.
3. If the context does not support a conclusion, say so explicitly rather than \
guessing.
4. You are advising a review, not issuing a verdict. A flag is a prompt for \
analyst review, never an automatic decline.
5. Be concise: a short paragraph of reasoning, then a one-line recommended \
action. Do not repeat the raw feature values back verbatim as a list."""

TRANSACTION_TEMPLATE = """TRANSACTION
- Amount: ${amount:,.2f}
- Time of day (derived): {hour:02d}:00
- Model fraud probability: {probability:.4f} (operating threshold {threshold:.4f})
- Model decision: {decision}
- Top anomaly drivers (SHAP): {drivers}"""

CONTEXT_TEMPLATE = """CONTEXT DOCUMENTS (retrieved as most relevant to this transaction)
{documents}"""

TASK_INSTRUCTION = """TASK
Explain to the analyst why this transaction was flagged, grounding every claim \
in the documents above and citing them by id. Then give a one-line recommended \
next action consistent with the cited policies. If the evidence is weak or \
mixed, say so."""


def assemble_prompt(amount, hour, probability, threshold, is_flagged,
                    top_components, retrieved) -> tuple[str, str]:
    drivers = ", ".join(f"{c} ({v:+.2f})" for c, v in (top_components or [])[:5]) \
        or "none reported"
    txn = TRANSACTION_TEMPLATE.format(
        amount=amount, hour=hour, probability=probability, threshold=threshold,
        decision="FLAGGED for review" if is_flagged else "not flagged",
        drivers=drivers)

    doc_lines = []
    for r in retrieved:
        d = r.document
        doc_lines.append(f"[{d.doc_id}] ({d.kind}) {d.title}\n{d.body}")
    context = CONTEXT_TEMPLATE.format(documents="\n\n".join(doc_lines))

    user_prompt = f"{txn}\n\n{context}\n\n{TASK_INSTRUCTION}"
    return SYSTEM_PROMPT, user_prompt


# --------------------------------------------------------------------------- #
# Citation extraction + verification
# --------------------------------------------------------------------------- #
_CITE_RE = re.compile(r"\[(POL-\d+|CASE-\d+)\]")


@dataclass
class Explanation:
    text: str
    backend: str
    cited_ids: list[str] = field(default_factory=list)
    available_ids: list[str] = field(default_factory=list)
    hallucinated_ids: list[str] = field(default_factory=list)
    uncited_available_ids: list[str] = field(default_factory=list)

    @property
    def all_citations_valid(self) -> bool:
        return len(self.hallucinated_ids) == 0

    def report(self) -> str:
        lines = [self.text, "", f"[backend: {self.backend}]"]
        lines.append(f"cited: {', '.join(self.cited_ids) or '(none)'}")
        if self.hallucinated_ids:
            lines.append(f"!! HALLUCINATED CITATIONS (not in retrieved context): "
                         f"{', '.join(self.hallucinated_ids)}")
        else:
            lines.append("citation check: all cited ids were in retrieved context")
        return "\n".join(lines)


def verify_citations(text: str, retrieved) -> tuple[list, list, list, list]:
    available = [r.document.doc_id for r in retrieved]
    cited = list(dict.fromkeys(_CITE_RE.findall(text)))  # unique, order-preserving
    hallucinated = [c for c in cited if c not in available]
    uncited = [a for a in available if a not in cited]
    return cited, available, hallucinated, uncited


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _resolve_api_key() -> str | None:
    for var in API_KEY_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


def _gemini_generate(system: str, user: str) -> str | None:
    """Real generation via Google Gemini using the current `google-genai` SDK.
    Returns None to trigger the stub if the key is absent, the SDK is missing, or
    the call fails -- identical fallback contract to before, so nothing
    downstream changes."""
    if not _resolve_api_key():
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None
    try:
        client = genai.Client(api_key=_resolve_api_key())
        # In google-genai the system prompt and generation params live in a
        # GenerateContentConfig; the user text is passed as `contents`.
        resp = client.models.generate_content(
            model=MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=MAX_TOKENS,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),
            ),
        )
        # resp.text is the convenience accessor; guard against empty/blocked output.
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") for p in parts)
        return text or None
    except Exception as e:
        print(f"  [generator] Gemini call failed ({type(e).__name__}: {e}); "
              f"using offline stub.")
        return None


def _stub_generate(amount, hour, probability, threshold, is_flagged,
                   top_components, retrieved) -> str:
    """Deterministic explanation built ONLY from supplied facts + retrieved docs.

    This is not a language model. It templates the retrieved evidence so the
    pipeline is demonstrable offline. It cites only real retrieved ids and
    states only values it was given -- it cannot hallucinate."""
    if not retrieved:
        return ("No context documents were retrieved, so no grounded explanation "
                "can be given. Recommend manual review.")

    available = {r.document.doc_id for r in retrieved}

    def cite(doc_id: str) -> str:
        """Only emit a citation if that document was actually retrieved."""
        return f" [{doc_id}]" if doc_id in available else ""

    drivers = [c for c, _ in (top_components or [])[:4]]
    dominant = [c for c in drivers if c in {"V17", "V14", "V12", "V10"}]

    # amount framing
    if amount == 0:
        amt = "a zero-amount authorization, consistent with a card-verification probe"
    elif amount <= 1:
        amt = f"a micro-charge of ${amount:.2f}, a pattern associated with card testing"
    elif amount >= 1000:
        amt = f"a high-value charge of ${amount:,.2f}, in the manual-hold band"
    elif amount >= 200:
        amt = f"an elevated-value charge of ${amount:,.2f}"
    else:
        amt = f"a mid-value charge of ${amount:,.2f}"

    when = ("the overnight elevated-risk window" if 0 <= hour < 6
            else "ordinary daytime hours")

    top = retrieved[0].document
    supporting = ", ".join(f"[{r.document.doc_id}]" for r in retrieved)

    prob_band = ("a high model probability" if probability >= 0.90 else
                 "a moderate model probability" if probability >= 0.50 else
                 "a borderline model probability")

    pol04 = cite("POL-04")
    anomaly = (f"The score is driven chiefly by the dominant fraud components "
               f"{', '.join(dominant)}, which historically separate fraud most "
               f"strongly{pol04}." if dominant else
               f"The anomaly profile ({', '.join(drivers) or 'unremarkable'}) "
               f"lacks the dominant fraud components, so the flag rests more on "
               f"amount and timing than on a classic anomaly signature{pol04}.")

    body = (
        f"This transaction was {'flagged for review' if is_flagged else 'scored'} "
        f"at {prob_band} ({probability:.2f}, threshold {threshold:.2f}). "
        f"It is {amt}, occurring during {when}. "
        f"{anomaly} "
        f"The most similar precedent in the knowledge base is "
        f"[{top.doc_id}] \"{top.title}\", alongside {supporting}. "
    )

    # action line: prefer the canonical policy for this band, but only cite it
    # if it was retrieved; otherwise fall back to any retrieved policy, then to
    # an uncited generic review instruction. Never cite an unretrieved id.
    policy = next((r.document for r in retrieved if r.document.kind == "policy"), None)
    pol10 = cite("POL-10")

    if amount >= 1000 and "POL-03" in available:
        action = ("Recommended action: route to manual analyst sign-off before "
                  "any release; automated approval is not permitted in this band "
                  "[POL-03].")
    elif amount <= 1 and "POL-01" in available:
        action = ("Recommended action: treat as suspected card-testing and hold "
                  "the card for step-up verification rather than approving on the "
                  "small value [POL-01].")
    elif policy:
        action = (f"Recommended action: proceed to analyst review guided by "
                  f"[{policy.doc_id}]; record a rationale before any decision"
                  f"{pol10}.")
    else:
        action = ("Recommended action: route to standard analyst review and "
                  f"record a rationale before any decision{pol10}.")

    return body + "\n\n" + action


# --------------------------------------------------------------------------- #
# Public generator
# --------------------------------------------------------------------------- #
class ExplanationGenerator:
    def __init__(self):
        self.has_key = bool(_resolve_api_key())

    def generate(self, amount, hour, probability, threshold, is_flagged,
                 top_components, retrieved) -> Explanation:
        system, user = assemble_prompt(amount, hour, probability, threshold,
                                       is_flagged, top_components, retrieved)
        text = _gemini_generate(system, user)
        backend = f"gemini:{MODEL}"
        if text is None:
            text = _stub_generate(amount, hour, probability, threshold,
                                  is_flagged, top_components, retrieved)
            backend = "offline-stub"

        cited, available, hallucinated, uncited = verify_citations(text, retrieved)
        return Explanation(text=text, backend=backend, cited_ids=cited,
                           available_ids=available, hallucinated_ids=hallucinated,
                           uncited_available_ids=uncited)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml"))
    from retriever import Retriever

    r = Retriever()
    gen = ExplanationGenerator()
    print(f"Generator: {'GEMINI_API_KEY detected' if gen.has_key else 'no key -> offline stub'}"
          f" | retriever backend: {r.backend_name}\n")

    scenarios = [
        ("micro card-test overnight",
         dict(amount=0.79, hour=2, probability=0.97, threshold=0.3829,
              is_flagged=True,
              top_components=[("V14", 6.7), ("V10", 2.5), ("V4", 1.6)])),
        ("high-value daytime, no dominant anomaly",
         dict(amount=1420.0, hour=15, probability=0.64, threshold=0.3829,
              is_flagged=True,
              top_components=[("V4", 1.2), ("V11", 0.9)])),
    ]
    for label, feats in scenarios:
        _, retrieved = r.retrieve_for_transaction(
            amount=feats["amount"], hour=feats["hour"],
            probability=feats["probability"],
            top_components=feats["top_components"], k=4)
        exp = gen.generate(retrieved=retrieved, **feats)
        print("=" * 72)
        print(f"SCENARIO: {label}")
        print("=" * 72)
        print(exp.report())
        print()
