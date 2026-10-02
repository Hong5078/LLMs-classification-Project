"""Metrics reported for every experiment."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true, y_pred, labels):
    idx = list(range(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=idx)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    per_class = f1_score(y_true, y_pred, labels=idx, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "per_class_f1": {lab: float(f) for lab, f in zip(labels, per_class)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "n": int(len(y_true)),
    }
