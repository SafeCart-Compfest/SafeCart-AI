from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path

from safecart_ai.domain.normalization import normalize_nie, normalize_package, normalize_text
from safecart_ai.matching.metrics import bootstrap_classification_ci, classification_metrics
from safecart_ai.matching.pair_text import ID_TO_LABEL, label_id


class PairBaseline(StrEnum):
    EXACT_NIE = "exact-nie"
    DETERMINISTIC = "deterministic"


def predict_exact_nie(row: Mapping[str, str]) -> int:
    listing_nie = normalize_nie(row.get("listing_nie"))
    official_nie = normalize_nie(row.get("official_nie"))
    is_match = listing_nie is not None and listing_nie == official_nie
    return 0 if is_match else 1


def predict_deterministic(row: Mapping[str, str]) -> int:
    if predict_exact_nie(row) == 1:
        return 1
    fields: tuple[tuple[str, str, Callable[[str | None], str | None]], ...] = (
        ("listing_brand", "official_brand", normalize_text),
        ("listing_product_name", "official_product_name", normalize_text),
        ("listing_package", "official_package", normalize_package),
    )
    for listing_field, official_field, normalize in fields:
        listing_value = normalize(row.get(listing_field))
        official_value = normalize(row.get(official_field))
        if listing_value and official_value and listing_value != official_value:
            return 1
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_pair_baseline(
    pairs_path: Path,
    baseline: PairBaseline,
    split: str = "dev",
) -> dict[str, object]:
    predictor = {
        PairBaseline.EXACT_NIE: predict_exact_nie,
        PairBaseline.DETERMINISTIC: predict_deterministic,
    }[baseline]
    labels: list[int] = []
    predictions: list[int] = []
    mutation_totals: Counter[str] = Counter()
    mutation_correct: Counter[str] = Counter()
    errors: defaultdict[str, list[str]] = defaultdict(list)

    with pairs_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != split:
                continue
            actual = label_id(row["label"])
            predicted = predictor(row)
            labels.append(actual)
            predictions.append(predicted)
            mutation = row["mutation_type"]
            mutation_totals[mutation] += 1
            mutation_correct[mutation] += int(actual == predicted)
            if actual != predicted and len(errors[mutation]) < 20:
                errors[mutation].append(row["pair_id"])

    if not labels:
        raise ValueError(f"No pair rows found for split: {split}")
    metrics = classification_metrics(labels, predictions)
    return {
        "baseline": baseline.value,
        "split": split,
        "query_count": len(labels),
        "pairs_sha256": _sha256(pairs_path),
        "metrics": metrics,
        "confidence_intervals_95": bootstrap_classification_ci(labels, predictions),
        "by_mutation": {
            mutation: {
                "count": count,
                "accuracy": mutation_correct[mutation] / count,
                "sample_error_pair_ids": errors[mutation],
            }
            for mutation, count in sorted(mutation_totals.items())
        },
        "prediction_labels": ID_TO_LABEL,
    }
