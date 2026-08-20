"""
Phase 1 demo + reusable inference.

Loads the saved model and scores a transaction, returning the flag, the
probability, and the features that pushed the score up. That last part matters:
the PCA columns (V1..V28) are anonymised and carry no human meaning, so the
attribution is what the RAG layer later turns into a natural-language query.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "artifacts", "fraud_model.joblib")


@dataclass
class FraudPrediction:
    is_flagged: bool
    probability: float
    threshold: float
    amount: float
    time_seconds: float
    top_contributions: list[tuple[str, float]] = field(default_factory=list)

    @property
    def hour_of_day(self) -> int:
        return int((self.time_seconds // 3600) % 24)

    def summary(self) -> str:
        verdict = "FLAGGED" if self.is_flagged else "not flagged"
        return (f"{verdict} | p(fraud)={self.probability:.4f} "
                f"(threshold {self.threshold:.4f}) | "
                f"amount {self.amount:,.2f} | hour {self.hour_of_day:02d}:00")


class FraudScorer:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run: python ml/train.py")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.threshold = bundle["threshold"]
        self.feature_cols = bundle["feature_cols"]
        self.strategy = bundle["strategy"]
        self.data_source = bundle["data_source"]

    def _attribute(self, x_scaled: np.ndarray, k: int = 5):
        """Per-row contribution via XGBoost's SHAP output (pred_contribs)."""
        try:
            import xgboost as xgb
            dm = xgb.DMatrix(x_scaled, feature_names=self.feature_cols)
            contribs = self.model.get_booster().predict(dm, pred_contribs=True)[0]
            pairs = list(zip(self.feature_cols, contribs[:-1]))  # drop bias term
        except Exception:
            imp = self.model.feature_importances_
            pairs = list(zip(self.feature_cols, imp * x_scaled[0]))
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        return [(n, float(v)) for n, v in pairs[:k]]

    def score(self, transaction: pd.Series | dict) -> FraudPrediction:
        if isinstance(transaction, dict):
            transaction = pd.Series(transaction)
        x = transaction[self.feature_cols].values.astype(float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        proba = float(self.model.predict_proba(x_scaled)[0, 1])
        return FraudPrediction(
            is_flagged=proba >= self.threshold,
            probability=proba,
            threshold=self.threshold,
            amount=float(transaction["Amount"]),
            time_seconds=float(transaction["Time"]),
            top_contributions=self._attribute(x_scaled),
        )


if __name__ == "__main__":
    from data_loader import load_transactions

    scorer = FraudScorer()
    df, info = load_transactions(verbose=False)
    print(f"Model: strategy='{scorer.strategy}', threshold={scorer.threshold:.4f}, "
          f"trained on {scorer.data_source} data\n")

    rng = np.random.default_rng(7)
    fraud_idx = rng.choice(df.index[df.Class == 1], 2, replace=False)
    legit_idx = rng.choice(df.index[df.Class == 0], 2, replace=False)

    for label, idxs in [("KNOWN FRAUD", fraud_idx), ("KNOWN LEGITIMATE", legit_idx)]:
        for i in idxs:
            row = df.loc[i]
            pred = scorer.score(row)
            correct = pred.is_flagged == bool(row.Class)
            print(f"[{label}] row {i}")
            print(f"  {pred.summary()}")
            print(f"  model {'agrees' if correct else 'MISSES'} with ground truth")
            print("  top score drivers: " +
                  ", ".join(f"{n} {v:+.3f}" for n, v in pred.top_contributions))
            print()
