import numpy as np
from collections import defaultdict


def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    classes = set(y_true) | set(y_pred)

    correct = 0

    # Count TP, FP, FN, support
    for true, pred in zip(y_true, y_pred):
        support[true] += 1

        if true == pred:
            correct += 1
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    accuracy = correct / len(y_true)

    def metrics_for_class(c):
        precision_den = tp[c] + fp[c]
        recall_den = tp[c] + fn[c]

        if precision_den == 0:
            precision = 0.0
        else:
            precision = tp[c] / precision_den

        if recall_den == 0:
            recall = 0.0
        else:
            recall = tp[c] / recall_den

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    if average == "micro":
        total_tp = sum(tp[c] for c in classes)
        total_fp = sum(fp[c] for c in classes)
        total_fn = sum(fn[c] for c in classes)

        precision_den = total_tp + total_fp
        recall_den = total_tp + total_fn

        if precision_den == 0:
            precision = 0.0
        else:
            precision = total_tp / precision_den

        if recall_den == 0:
            recall = 0.0
        else:
            recall = total_tp / recall_den

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    elif average == "macro":
        precisions = []
        recalls = []
        f1s = []

        for c in classes:
            p, r, f = metrics_for_class(c)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        precision = sum(precisions) / len(classes)
        recall = sum(recalls) / len(classes)
        f1 = sum(f1s) / len(classes)

    elif average == "weighted":
        total_support = len(y_true)

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        for c in classes:
            p, r, f = metrics_for_class(c)

            weight = support[c] / total_support

            precision += p * weight
            recall += r * weight
            f1 += f * weight

    elif average == "binary":
        precision, recall, f1 = metrics_for_class(pos_label)

    else:
        raise ValueError(
            'average must be "micro", "macro", "weighted", or "binary"'
        )

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }
