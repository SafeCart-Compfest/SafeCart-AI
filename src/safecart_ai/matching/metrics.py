from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from safecart_ai.matching.pair_text import ID_TO_LABEL


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, object]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    if not labels:
        raise ValueError("classification metrics require at least one row")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    per_class = {
        ID_TO_LABEL[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in (0, 1)
    }
    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "label_order": [ID_TO_LABEL[0], ID_TO_LABEL[1]],
    }
