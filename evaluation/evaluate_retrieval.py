"""
Phase 5a -- retrieval evaluation (runs fully offline, no LLM needed).

A hand-labelled test set maps each query to the KB documents that are genuinely
relevant to it. Relevance was judged by MEANING, not shared words: e.g. a
zero-amount query is relevant to POL-08 and CASE-03/CASE-17 whether or not they
share vocabulary. This is what makes the metric meaningful -- if labels were
keyword overlap, TF-IDF would trivially "win" by construction.

Metrics reported per query and in aggregate:
  - Precision@k : fraction of the top-k retrieved that are relevant
  - Recall@k    : fraction of all relevant docs that appear in the top-k
  - MRR         : 1/rank of the first relevant doc (rewards good ranking)
  - Hit@k       : did at least one relevant doc appear in top-k

The same harness runs on whichever retrieval backend is active, and PRINTS which
one produced the numbers, because TF-IDF and neural embeddings will score
differently and the report must not conflate them.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))

import numpy as np

from retriever import Retriever

# --------------------------------------------------------------------------- #
# Labelled test set: (query, set-of-relevant-doc-ids)
# Relevance judged by document meaning. Each query has >=2 relevant docs so
# recall is measurable. Some queries deliberately include hard cases where a
# legitimate-lookalike case is NOT relevant to a fraud query, and vice versa.
# --------------------------------------------------------------------------- #
TEST_SET = [
    ("Micro-charge under one dollar flagged overnight, possible card testing",
     {"POL-01", "CASE-01", "POL-12", "CASE-21", "CASE-06"}),

    ("Zero-amount authorization, possible card-verification probe",
     {"POL-08", "CASE-03", "CASE-17"}),

    ("High-value transaction over $1000 flagged, needs manual hold",
     {"POL-03", "CASE-04", "CASE-13"}),

    ("Overnight transaction in the small hours, elevated risk window",
     {"POL-02", "CASE-02", "CASE-19", "CASE-12"}),

    ("Flag driven by dominant anomaly components V17 V14 V12 V10",
     {"POL-04", "CASE-13", "CASE-18", "CASE-23"}),

    ("Large legitimate purchase flagged as false positive, no dominant anomaly",
     {"CASE-05", "CASE-20", "POL-09"}),

    ("Cluster of small charges in quick succession, coordinated run",
     {"POL-07", "CASE-11", "CASE-06", "CASE-19"}),

    ("Borderline low-probability flag below the action threshold, watch-list",
     {"POL-06", "CASE-10", "CASE-16"}),

    ("Confirmed fraud that a default 0.5 threshold would have missed",
     {"CASE-16", "POL-06"}),

    ("Escalating amounts on one card, probe then test then cash-out",
     {"CASE-22", "CASE-14", "POL-12"}),

    ("Small amount should not be auto-approved, card-testing exploit",
     {"POL-05", "POL-01", "CASE-07"}),

    ("High-value charge during ordinary daytime hours with strong anomaly",
     {"POL-11", "CASE-08", "CASE-13"}),

    ("Analyst needs a recorded rationale and citation before acting on a flag",
     {"POL-10", "POL-09"}),

    ("Legitimate overnight activity, weak anomaly profile, not fraud",
     {"CASE-12", "CASE-20", "POL-09"}),
]


def precision_recall_at_k(retrieved_ids, relevant, k):
    topk = retrieved_ids[:k]
    hits = sum(1 for d in topk if d in relevant)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall, hits


def reciprocal_rank(retrieved_ids, relevant):
    for i, d in enumerate(retrieved_ids):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(k: int = 5, force_tfidf: bool = False, verbose: bool = True):
    r = Retriever(force_tfidf=force_tfidf)
    rows = []
    for query, relevant in TEST_SET:
        results = r.search(query, k=k)
        ids = [res.document.doc_id for res in results]
        p, rec, hits = precision_recall_at_k(ids, relevant, k)
        rr = reciprocal_rank(ids, relevant)
        rows.append({"query": query, "relevant": relevant, "retrieved": ids,
                     "precision": p, "recall": rec, "hits": hits,
                     "rr": rr, "hit": hits > 0})

    agg = {
        "backend": r.backend_name,
        "k": k,
        "precision_at_k": float(np.mean([x["precision"] for x in rows])),
        "recall_at_k": float(np.mean([x["recall"] for x in rows])),
        "mrr": float(np.mean([x["rr"] for x in rows])),
        "hit_rate": float(np.mean([x["hit"] for x in rows])),
        "n_queries": len(rows),
    }

    if verbose:
        print(f"RETRIEVAL EVALUATION  (backend: {r.backend_name}, k={k}, "
              f"{len(rows)} queries)")
        print("=" * 78)
        for x in rows:
            marker = "ok " if x["hit"] else "MISS"
            print(f"[{marker}] P@{k}={x['precision']:.2f} R@{k}={x['recall']:.2f} "
                  f"RR={x['rr']:.2f}  {x['query'][:52]}")
            gold = ", ".join(sorted(x["relevant"]))
            got = ", ".join(x["retrieved"])
            print(f"        relevant: {gold}")
            print(f"        got     : {got}")
        print("=" * 78)
        print(f"  mean Precision@{k}: {agg['precision_at_k']:.3f}")
        print(f"  mean Recall@{k}   : {agg['recall_at_k']:.3f}")
        print(f"  MRR              : {agg['mrr']:.3f}")
        print(f"  Hit@{k}           : {agg['hit_rate']:.3f}  "
              f"({sum(x['hit'] for x in rows)}/{len(rows)} queries found "
              f">=1 relevant doc)")
        if r.backend_name == "tfidf":
            print("\n  NOTE: these are TF-IDF (lexical) numbers. Neural embeddings")
            print("  (run locally) should improve ranking on paraphrased matches;")
            print("  rerun this script there to get the comparison.")
    return agg, rows


if __name__ == "__main__":
    evaluate_retrieval(k=5)
