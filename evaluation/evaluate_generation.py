"""
Phase 5b -- generation evaluation.

Two layers, because one runs offline now and one needs the LLM:

  1. MANUAL RUBRIC (runs now, offline, against whatever backend generated the
     explanation -- stub or Gemini). Three checks that need no second LLM:
       - citation validity : every cited id was actually retrieved
                             (0 hallucinated citations). This IS a faithfulness
                             signal -- an explanation that cites only real
                             context cannot smuggle in outside "evidence".
       - grounding         : the explanation's concrete claims (amount band,
                             time window, probability) match the transaction
                             facts it was given.
       - citation coverage : did it cite at least one retrieved doc rather than
                             asserting ungrounded prose.
     These are computable deterministically and are reported per-example with
     the actual text, per the project's "show the output for manual inspection"
     standard.

  2. RAGAS HARNESS (ready, needs GEMINI_API_KEY + internet; skips cleanly
     otherwise). RAGAS uses an LLM judge to score faithfulness and answer/
     context relevance. Wired to use Gemini as BOTH the judge model and the
     embeddings so it runs end-to-end locally with your key. It is intentionally
     import-guarded so this file always runs; when RAGAS or the key is absent it
     prints exactly what's missing and falls back to the manual rubric only.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))

from retriever import Retriever
from generator import ExplanationGenerator, _resolve_api_key


# --------------------------------------------------------------------------- #
# Evaluation scenarios: fixed transactions so results are reproducible.
# (amount, hour, probability, threshold, is_flagged, top_components)
# --------------------------------------------------------------------------- #
SCENARIOS = [
    ("micro card-test overnight",
     dict(amount=0.79, hour=2, probability=0.97, threshold=0.3829, is_flagged=True,
          top_components=[("V14", 6.7), ("V10", 2.5), ("V4", 1.6)])),
    ("high-value manual-hold band",
     dict(amount=1420.0, hour=15, probability=0.64, threshold=0.3829, is_flagged=True,
          top_components=[("V4", 1.2), ("V11", 0.9)])),
    ("zero-amount probe",
     dict(amount=0.0, hour=11, probability=0.71, threshold=0.3829, is_flagged=True,
          top_components=[("V12", 2.1), ("V17", 1.8)])),
    ("legitimate-lookalike large purchase",
     dict(amount=1050.0, hour=14, probability=0.47, threshold=0.3829, is_flagged=True,
          top_components=[("V4", 0.6), ("V22", 0.4)])),
]


# --------------------------------------------------------------------------- #
# Manual rubric (offline)
# --------------------------------------------------------------------------- #
def _grounding_checks(text: str, feats: dict) -> list[tuple[str, bool]]:
    """Deterministic checks that stated facts match the transaction."""
    checks = []
    amount = feats["amount"]
    # amount appears (allow $0.79 / $1,420.00 / $1420 style)
    amt_variants = [f"{amount:,.2f}", f"{amount:.2f}", f"{amount:,.0f}"]
    checks.append(("amount stated correctly",
                   any(v in text for v in amt_variants)))
    # time window consistency
    overnight = 0 <= feats["hour"] < 6
    says_overnight = bool(re.search(r"overnight|small hours|night", text, re.I))
    says_daytime = bool(re.search(r"daytime|ordinary.*hours", text, re.I))
    checks.append(("time window consistent",
                   (overnight and says_overnight) or
                   (not overnight and says_daytime) or
                   (not says_overnight and not says_daytime)))
    # probability band consistency
    p = feats["probability"]
    band = "high" if p >= 0.90 else "moderate" if p >= 0.50 else "borderline"
    checks.append((f"probability band ~'{band}' present",
                   band in text.lower() or f"{p:.2f}" in text))
    return checks


def manual_rubric(k: int = 4):
    r = Retriever()
    gen = ExplanationGenerator()
    print(f"MANUAL GENERATION RUBRIC  (generator backend: "
          f"{'gemini' if gen.has_key else 'offline-stub'}, "
          f"retriever: {r.backend_name})")
    print("=" * 78)

    summary = []
    for label, feats in SCENARIOS:
        _, retrieved = r.retrieve_for_transaction(
            amount=feats["amount"], hour=feats["hour"],
            probability=feats["probability"],
            top_components=feats["top_components"], k=k)
        exp = gen.generate(retrieved=retrieved, **feats)

        grounding = _grounding_checks(exp.text, feats)
        cite_valid = exp.all_citations_valid
        has_citation = len(exp.cited_ids) > 0
        grounding_pass = all(v for _, v in grounding)

        print(f"\nSCENARIO: {label}")
        print("-" * 78)
        print(exp.text)
        print(f"\n  cited: {', '.join(exp.cited_ids) or '(none)'}")
        print(f"  [{'PASS' if cite_valid else 'FAIL'}] citation validity "
              f"(0 hallucinated): {len(exp.hallucinated_ids)} bad")
        print(f"  [{'PASS' if has_citation else 'FAIL'}] cites >=1 retrieved doc")
        for name, ok in grounding:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

        overall = cite_valid and has_citation and grounding_pass
        summary.append((label, overall))

    print("\n" + "=" * 78)
    print("RUBRIC SUMMARY")
    for label, ok in summary:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = sum(1 for _, ok in summary if ok)
    print(f"  {passed}/{len(summary)} scenarios pass all checks")
    return summary


# --------------------------------------------------------------------------- #
# RAGAS harness (needs key + internet; skips cleanly)
# --------------------------------------------------------------------------- #
def ragas_eval(k: int = 4):
    print("\n" + "=" * 78)
    print("RAGAS EVALUATION (LLM-judged faithfulness / relevance)")
    print("=" * 78)

    if not _resolve_api_key():
        print("  SKIPPED: no GEMINI_API_KEY / GOOGLE_API_KEY set.")
        print("  Set the key and re-run locally to compute faithfulness,")
        print("  answer_relevancy, and context_precision via a Gemini judge.")
        return None
    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import (ChatGoogleGenerativeAI,
                                            GoogleGenerativeAIEmbeddings)
        from datasets import Dataset
    except Exception as e:
        print(f"  SKIPPED: RAGAS/LangChain-Google not installed ({type(e).__name__}).")
        print("  Install locally with:")
        print("    pip install ragas langchain-google-genai datasets")
        return None

    r = Retriever()
    gen = ExplanationGenerator()
    # NOTE: on your machine this makes real Gemini calls -- one generation per
    # scenario plus several judge/embedding calls per RAGAS metric. Expect ~1-2
    # minutes for the 4 scenarios and to consume a little Gemini quota. If it
    # appears to hang, it is almost always a network/quota issue on the judge
    # calls, not a logic error.
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for label, feats in SCENARIOS:
        query, retrieved = r.retrieve_for_transaction(
            amount=feats["amount"], hour=feats["hour"],
            probability=feats["probability"],
            top_components=feats["top_components"], k=k)
        exp = gen.generate(retrieved=retrieved, **feats)
        rows["question"].append(query)
        rows["answer"].append(exp.text)
        rows["contexts"].append([rr.document.text for rr in retrieved])
        rows["ground_truth"].append(retrieved[0].document.text)

    api_key = _resolve_api_key()
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", temperature=0, google_api_key=api_key))
    embed = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key))

    ds = Dataset.from_dict(rows)
    result = evaluate(
        ds, metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge, embeddings=embed)
    print("  RAGAS scores:")
    print(" ", result)
    return result


if __name__ == "__main__":
    manual_rubric()
    ragas_eval()
