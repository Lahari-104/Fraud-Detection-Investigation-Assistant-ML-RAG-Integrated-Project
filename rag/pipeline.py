"""
End-to-end pipeline: transaction -> ML flag -> (if flagged) retrieval ->
grounded explanation with verified citations.

This is the single entry point the Streamlit app (Phase 6) will call. It ties
together the three swappable pieces so the caller never touches backends.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from predict import FraudScorer, FraudPrediction
from retriever import Retriever
from generator import ExplanationGenerator, Explanation


@dataclass
class PipelineOutput:
    prediction: FraudPrediction
    query: str | None
    retrieved: list
    explanation: Explanation | None

    @property
    def was_explained(self) -> bool:
        return self.explanation is not None


class FraudRAGPipeline:
    def __init__(self, k: int = 4):
        self.scorer = FraudScorer()
        self.retriever = Retriever()
        self.generator = ExplanationGenerator()
        self.k = k

    @property
    def info(self) -> dict:
        return {
            "model_strategy": self.scorer.strategy,
            "model_threshold": self.scorer.threshold,
            "data_source": self.scorer.data_source,
            "retriever_backend": self.retriever.backend_name,
            "generator_backend": ("gemini" if self.generator.has_key
                                  else "offline-stub"),
            "kb_size": len(self.retriever.docs),
        }

    def run(self, transaction: pd.Series | dict,
            explain_even_if_clean: bool = False) -> PipelineOutput:
        pred = self.scorer.score(transaction)

        if not pred.is_flagged and not explain_even_if_clean:
            return PipelineOutput(pred, None, [], None)

        query, retrieved = self.retriever.retrieve_for_transaction(
            amount=pred.amount, hour=pred.hour_of_day,
            probability=pred.probability,
            top_components=pred.top_contributions, k=self.k)

        explanation = self.generator.generate(
            amount=pred.amount, hour=pred.hour_of_day,
            probability=pred.probability, threshold=pred.threshold,
            is_flagged=pred.is_flagged,
            top_components=pred.top_contributions, retrieved=retrieved)

        return PipelineOutput(pred, query, retrieved, explanation)


if __name__ == "__main__":
    import numpy as np
    from data_loader import load_transactions

    pipe = FraudRAGPipeline()
    print("PIPELINE CONFIG")
    for k, v in pipe.info.items():
        print(f"  {k}: {v}")
    print()

    df, _ = load_transactions(verbose=False)
    rng = np.random.default_rng(11)
    idx = int(rng.choice(df.index[df.Class == 1], 1)[0])
    row = df.loc[idx]

    print("=" * 72)
    print(f"END-TO-END on real transaction (row {idx}, ground truth = FRAUD)")
    print("=" * 72)
    out = pipe.run(row)
    print(f"\n1. ML: {out.prediction.summary()}")
    print(f"   drivers: " +
          ", ".join(f"{n} {v:+.2f}" for n, v in out.prediction.top_contributions))
    if out.was_explained:
        print(f"\n2. QUERY: {out.query}")
        print(f"\n3. RETRIEVED ({len(out.retrieved)}): " +
              ", ".join(r.document.doc_id for r in out.retrieved))
        print(f"\n4. EXPLANATION:")
        print("   " + out.explanation.text.replace("\n", "\n   "))
        print(f"\n   {'PASS' if out.explanation.all_citations_valid else 'FAIL'}: "
              f"citation check | cited {out.explanation.cited_ids}")
    else:
        print("   not flagged -> no explanation generated")
