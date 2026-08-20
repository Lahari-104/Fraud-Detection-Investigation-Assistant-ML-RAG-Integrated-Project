"""
Phase 1 -- fraud classifier.

Compares three ways of handling the ~0.17% class imbalance so the choice is
evidenced rather than asserted:
    A. threshold-only  -- no resampling/reweighting, but a tuned decision
                          threshold (threshold tuning is itself imbalance handling)
    B. class weighting -- XGBoost scale_pos_weight
    C. SMOTE           -- synthetic minority oversampling on the training split

Reports precision / recall / F1 / PR-AUC / confusion matrix on a held-out test
split. Accuracy is printed only alongside the base rate, to make clear it is
uninformative here.

Decision threshold is tuned on a validation split (not the test split), because
0.5 is arbitrary for imbalanced problems.
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import FEATURE_COLS, TARGET_COL, load_transactions

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
RANDOM_STATE = 42


def _fmt_cm(cm: np.ndarray) -> str:
    tn, fp, fn, tp = cm.ravel()
    return (f"        pred_legit  pred_fraud\n"
            f"  legit   {tn:9,}  {fp:10,}\n"
            f"  fraud   {fn:9,}  {tp:10,}")


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float((tp + tn) / len(y_true)),
    }


def best_threshold(y_val, proba_val) -> float:
    """Threshold maximising F1 on the validation split."""
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def build_model(scale_pos_weight: float | None = None):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight if scale_pos_weight else 1.0,
    )


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df, info = load_transactions()

    print("\n" + "=" * 68)
    print("CLASS BALANCE")
    print("=" * 68)
    counts = df[TARGET_COL].value_counts().sort_index()
    print(f"  legitimate (0): {counts[0]:,}")
    print(f"  fraud      (1): {counts[1]:,}")
    print(f"  fraud rate    : {info.fraud_rate:.5%}")
    print(f"  -> a model predicting 'never fraud' scores "
          f"{100 * (1 - info.fraud_rate):.3f}% accuracy and catches 0 fraud.")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = (scaler.transform(a)
                                    for a in (X_train, X_val, X_test))

    print(f"\n  train {len(y_train):,} ({y_train.sum()} fraud) | "
          f"val {len(y_val):,} ({y_val.sum()} fraud) | "
          f"test {len(y_test):,} ({y_test.sum()} fraud)")

    results = {}

    # ---- A. baseline, imbalance ignored ---------------------------------
    print("\n" + "=" * 68)
    print("STRATEGY A -- threshold tuning only (no resampling/reweighting)")
    print("=" * 68)
    t0 = time.time()
    m_a = build_model()
    m_a.fit(X_train_s, y_train)
    thr_a = best_threshold(y_val, m_a.predict_proba(X_val_s)[:, 1])
    res_a = evaluate(y_test, m_a.predict_proba(X_test_s)[:, 1], thr_a)
    results["threshold_only"] = res_a
    print(f"  trained in {time.time()-t0:.1f}s | tuned threshold {thr_a:.4f}")
    print(f"  precision {res_a['precision']:.4f}  recall {res_a['recall']:.4f}  "
          f"F1 {res_a['f1']:.4f}  PR-AUC {res_a['pr_auc']:.4f}")

    # ---- B. class weighting ---------------------------------------------
    print("\n" + "=" * 68)
    print("STRATEGY B -- class weighting (scale_pos_weight)")
    print("=" * 68)
    spw = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    print(f"  scale_pos_weight = {spw:.1f}")
    t0 = time.time()
    m_b = build_model(scale_pos_weight=spw)
    m_b.fit(X_train_s, y_train)
    thr_b = best_threshold(y_val, m_b.predict_proba(X_val_s)[:, 1])
    res_b = evaluate(y_test, m_b.predict_proba(X_test_s)[:, 1], thr_b)
    results["class_weight"] = res_b
    print(f"  trained in {time.time()-t0:.1f}s | tuned threshold {thr_b:.4f}")
    print(f"  precision {res_b['precision']:.4f}  recall {res_b['recall']:.4f}  "
          f"F1 {res_b['f1']:.4f}  PR-AUC {res_b['pr_auc']:.4f}")

    # ---- C. SMOTE --------------------------------------------------------
    print("\n" + "=" * 68)
    print("STRATEGY C -- SMOTE oversampling (training split only)")
    print("=" * 68)
    from imblearn.over_sampling import SMOTE
    t0 = time.time()
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=5, sampling_strategy=0.10)
    X_sm, y_sm = sm.fit_resample(X_train_s, y_train)
    print(f"  resampled {len(y_train):,} -> {len(y_sm):,} rows "
          f"({y_sm.sum():,} fraud) in {time.time()-t0:.1f}s")
    print("  NOTE: SMOTE applied AFTER the split, to the training rows only. "
          "Resampling\n        before splitting leaks synthetic minority rows "
          "into the test set.")
    t0 = time.time()
    m_c = build_model()
    m_c.fit(X_sm, y_sm)
    thr_c = best_threshold(y_val, m_c.predict_proba(X_val_s)[:, 1])
    res_c = evaluate(y_test, m_c.predict_proba(X_test_s)[:, 1], thr_c)
    results["smote"] = res_c
    print(f"  trained in {time.time()-t0:.1f}s | tuned threshold {thr_c:.4f}")
    print(f"  precision {res_c['precision']:.4f}  recall {res_c['recall']:.4f}  "
          f"F1 {res_c['f1']:.4f}  PR-AUC {res_c['pr_auc']:.4f}")

    # ---- comparison ------------------------------------------------------
    print("\n" + "=" * 68)
    print("COMPARISON (held-out test split)")
    print("=" * 68)
    print(f"{'strategy':<16}{'precision':>10}{'recall':>9}{'F1':>8}"
          f"{'PR-AUC':>9}{'FP':>7}{'FN':>6}")
    for name, r in results.items():
        print(f"{name:<16}{r['precision']:>10.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>8.4f}{r['pr_auc']:>9.4f}{r['fp']:>7,}{r['fn']:>6,}")

    models = {"threshold_only": m_a, "class_weight": m_b, "smote": m_c}
    winner = max(results, key=lambda k: results[k]["f1"])
    best = results[winner]
    model = models[winner]

    print(f"\n  selected: {winner}  (highest F1)")
    print(f"\n  Confusion matrix -- {winner} @ threshold {best['threshold']:.4f}")
    print(_fmt_cm(np.array(best["confusion_matrix"])))
    print(f"\n  accuracy {best['accuracy']:.5f} -- but 'always legitimate' scores "
          f"{1 - y_test.mean():.5f},\n  so accuracy is not evidence of anything "
          f"here. Precision/recall/F1 are.")

    # ---- persist ---------------------------------------------------------
    joblib.dump({"model": model, "scaler": scaler,
                 "threshold": best["threshold"],
                 "feature_cols": FEATURE_COLS,
                 "strategy": winner,
                 "data_source": info.source},
                os.path.join(ARTIFACT_DIR, "fraud_model.joblib"))

    importances = sorted(zip(FEATURE_COLS, model.feature_importances_.tolist()),
                         key=lambda kv: kv[1], reverse=True)
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump({"data_source": info.source,
                   "fraud_rate": info.fraud_rate,
                   "n_rows": info.n_rows,
                   "selected_strategy": winner,
                   "results": results,
                   "feature_importance": importances}, f, indent=2)

    print(f"\n  saved -> {ARTIFACT_DIR}/fraud_model.joblib")
    print(f"  saved -> {ARTIFACT_DIR}/metrics.json")
    print("\n  top features: " +
          ", ".join(f"{k} ({v:.3f})" for k, v in importances[:8]))

    if info.is_synthetic:
        print("\n" + "!" * 68)
        print("These numbers come from SYNTHETIC data. They validate that the")
        print("pipeline runs and that imbalance handling changes outcomes. They")
        print("say nothing about real fraud detection performance.")
        print("!" * 68)


if __name__ == "__main__":
    main()
