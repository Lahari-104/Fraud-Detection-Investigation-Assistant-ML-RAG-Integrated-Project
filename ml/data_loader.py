"""
Transaction data loading.

PRIMARY PATH: the real Kaggle "Credit Card Fraud Detection" dataset at
    data/transactions/creditcard.csv
If that file exists, it is used and nothing here is synthetic.

FALLBACK PATH: if the file is absent, generate a synthetic surrogate with the
IDENTICAL schema (Time, V1..V28, Amount, Class). This exists only so the rest of
the pipeline is runnable and testable. Any metric computed on the surrogate
describes THIS GENERATOR, not real-world fraud detection performance.

Drop the real creditcard.csv into data/transactions/ and every downstream step
works unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "transactions")
REAL_CSV = os.path.join(DATA_DIR, "creditcard.csv")
SYNTH_CSV = os.path.join(DATA_DIR, "creditcard_SYNTHETIC.csv")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"

# Directional shifts mirroring which PCA components separate fraud in the real
# dataset (V14/V12/V10/V17 run negative for fraud; V4/V11 run positive).
_SIGNAL = {
    "V14": -2.4, "V12": -2.0, "V10": -1.8, "V17": -2.2,
    "V4": 2.1, "V11": 1.9, "V3": -1.5, "V16": -1.3, "V7": -1.2, "V18": -1.0,
}
_STEALTH_FRACTION = 0.22  # frauds drawn from the legitimate distribution


@dataclass
class DatasetInfo:
    source: str          # "real" or "synthetic"
    path: str
    n_rows: int
    n_fraud: int
    fraud_rate: float

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    def banner(self) -> str:
        if self.is_synthetic:
            return (
                "!! SYNTHETIC SURROGATE DATA !!\n"
                "   The real Kaggle creditcard.csv was not found.\n"
                "   Metrics below measure the pipeline, NOT real fraud detection skill.\n"
                f"   Drop the real file at: {REAL_CSV}"
            )
        return f"Real dataset loaded from {self.path}"


def _generate_synthetic(n_rows: int = 284_807,
                        fraud_rate: float = 0.001727,
                        seed: int = 42) -> pd.DataFrame:
    """Build a surrogate that is deliberately *not* trivially separable."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    # PCA components have decreasing variance in the real data; mimic that.
    scales = np.linspace(1.9, 0.45, 28)

    legit = rng.standard_normal((n_legit, 28)) * scales
    fraud = rng.standard_normal((n_fraud, 28)) * scales

    # Apply separating shifts to the informative components, with per-row
    # jitter so the boundary is fuzzy rather than a clean offset.
    for col, shift in _SIGNAL.items():
        j = int(col[1:]) - 1
        jitter = rng.normal(1.0, 0.55, size=n_fraud)
        fraud[:, j] += shift * scales[j] * jitter

    # A slice of frauds look completely ordinary -> caps achievable recall.
    n_stealth = int(n_fraud * _STEALTH_FRACTION)
    if n_stealth:
        idx = rng.choice(n_fraud, size=n_stealth, replace=False)
        fraud[idx, :] = rng.standard_normal((n_stealth, 28)) * scales

    # A slice of legit rows look anomalous -> caps achievable precision.
    n_noisy = int(n_legit * 0.0009)
    if n_noisy:
        idx = rng.choice(n_legit, size=n_noisy, replace=False)
        for col, shift in _SIGNAL.items():
            j = int(col[1:]) - 1
            legit[idx, j] += shift * scales[j] * rng.normal(0.85, 0.4, size=n_noisy)

    V = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Amount: heavy-tailed; fraud skews toward small "card test" charges plus a
    # high-value tail.
    amt_legit = np.clip(rng.lognormal(3.05, 1.35, n_legit), 0, 25_691.16)
    small = rng.random(n_fraud) < 0.45
    amt_fraud = np.where(small,
                         np.clip(rng.lognormal(0.6, 0.9, n_fraud), 0, 60),
                         np.clip(rng.lognormal(4.4, 1.5, n_fraud), 0, 25_691.16))
    amount = np.concatenate([amt_legit, amt_fraud])

    # Time: seconds across ~2 days, diurnal for legit, night-weighted for fraud.
    t_legit = (rng.beta(2.2, 2.2, n_legit) * 172_792)
    t_fraud = np.where(rng.random(n_fraud) < 0.55,
                       rng.uniform(0, 172_792, n_fraud) % 21_600 + rng.choice([0, 86_400], n_fraud),
                       rng.uniform(0, 172_792, n_fraud))
    time = np.concatenate([t_legit, t_fraud])

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.round(time).astype(int))
    df["Amount"] = np.round(amount, 2)
    df[TARGET_COL] = y

    # Shuffle so ordering carries no signal, then sort by Time like the real file.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    return df[FEATURE_COLS + [TARGET_COL]]


def load_transactions(force_synthetic: bool = False,
                      verbose: bool = True) -> tuple[pd.DataFrame, DatasetInfo]:
    """Return (dataframe, DatasetInfo). Prefers the real CSV when present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(REAL_CSV) and not force_synthetic:
        df = pd.read_csv(REAL_CSV)
        source, path = "real", REAL_CSV
    else:
        if os.path.exists(SYNTH_CSV):
            df = pd.read_csv(SYNTH_CSV)
        else:
            df = _generate_synthetic()
            df.to_csv(SYNTH_CSV, index=False)
        source, path = "synthetic", SYNTH_CSV

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    n_fraud = int(df[TARGET_COL].sum())
    info = DatasetInfo(source=source, path=path, n_rows=len(df),
                       n_fraud=n_fraud, fraud_rate=n_fraud / len(df))
    if verbose:
        print(info.banner())
    return df, info


if __name__ == "__main__":
    df, info = load_transactions()
    print(f"\nRows: {info.n_rows:,}   Columns: {df.shape[1]}")
    print(f"Class balance:")
    print(f"  legitimate (0): {info.n_rows - info.n_fraud:,}  "
          f"({100 * (1 - info.fraud_rate):.4f}%)")
    print(f"  fraud      (1): {info.n_fraud:,}  ({100 * info.fraud_rate:.4f}%)")
    print(f"  imbalance ratio: 1 fraud per {int(1 / info.fraud_rate):,} transactions")
    print(f"\nAmount  -- legit median {df.loc[df.Class==0,'Amount'].median():.2f} | "
          f"fraud median {df.loc[df.Class==1,'Amount'].median():.2f}")
    print(f"Time span: {df.Time.min():,} to {df.Time.max():,} seconds "
          f"({df.Time.max()/3600:.1f} hours)")
