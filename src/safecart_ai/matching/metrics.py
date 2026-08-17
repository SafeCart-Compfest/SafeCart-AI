from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
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


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bootstrap_classification_ci(
    labels: Sequence[int],
    predictions: Sequence[int],
    seed: int = 42,
    samples: int = 2000,
) -> dict[str, object]:
    """Bootstrap confusion-derived metrics without materializing row-sized samples."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    if not labels:
        raise ValueError("bootstrap requires at least one row")
    if samples < 1:
        raise ValueError("samples must be positive")

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    observed = [int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])]
    probabilities = [count / len(labels) for count in observed]
    draws = cast(
        list[list[int]],
        np.random.default_rng(seed).multinomial(len(labels), probabilities, size=samples).tolist(),
    )
    estimates: dict[str, list[float]] = {
        "macro_f1": [],
        "match_precision": [],
        "match_recall": [],
        "mismatch_precision": [],
        "mismatch_recall": [],
    }
    for true_match, false_mismatch, false_match, true_mismatch in draws:
        match_precision = _ratio(true_match, true_match + false_match)
        match_recall = _ratio(true_match, true_match + false_mismatch)
        mismatch_precision = _ratio(true_mismatch, true_mismatch + false_mismatch)
        mismatch_recall = _ratio(true_mismatch, true_mismatch + false_match)
        estimates["macro_f1"].append(
            (_f1(match_precision, match_recall) + _f1(mismatch_precision, mismatch_recall)) / 2
        )
        estimates["match_precision"].append(match_precision)
        estimates["match_recall"].append(match_recall)
        estimates["mismatch_precision"].append(mismatch_precision)
        estimates["mismatch_recall"].append(mismatch_recall)

    return {
        "method": f"percentile bootstrap ({samples} multinomial resamples)",
        **{
            name: {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
            }
            for name, values in estimates.items()
        },
    }
