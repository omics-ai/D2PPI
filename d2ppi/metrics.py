"""Binary-classification metrics: AUPR (average precision) and AUROC, and the thresholded metrics at 0.5."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def all_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "aupr": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "mcc": (tp * tn - fp * fn) / den if den else 0.0,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "balanced_acc": (rec + spec) / 2,
        "accuracy": (tp + tn) / len(y),
        "ppr": (tp + fp) / len(y),  # predicted-positive rate at the threshold
        "threshold": threshold,
    }
