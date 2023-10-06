import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate(
    y_true, y_pred_score, threshold=0.5, precision_k=[10, 100, 200], round_dec=4
):
    metrics = {}

    # Binary classification
    y_pred_bin = np.where(y_pred_score > threshold, 1, 0)
    metrics["precision"] = round(precision_score(y_true, y_pred_bin), round_dec)
    metrics["recall"] = round(recall_score(y_true, y_pred_bin), round_dec)
    metrics["F1"] = round(f1_score(y_true, y_pred_bin), round_dec)

    # Ranking
    # sort = np.argsort(y_pred_score[:, 0])[::-1]
    sort = np.argsort(y_pred_score)[::-1]
    # y_true_ranked = y_true[sort, :]
    y_true_ranked = y_true[sort]
    # y_pred_ranked = y_pred_bin[sort, :]
    y_pred_ranked = y_pred_bin[sort]

    for k in precision_k:
        metrics[f"P{k}"] = round(
            precision_score(y_true_ranked[:k], y_pred_ranked[:k]), round_dec
        )

    return metrics
