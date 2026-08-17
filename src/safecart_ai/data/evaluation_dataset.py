from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LABELS = {"MATCH", "MISMATCH", "INSUFFICIENT_EVIDENCE"}
REVIEW_STAGES = {"initial", "resolution"}
READABILITY_VALUES = {"READABLE", "PARTIALLY_READABLE", "UNREADABLE"}
REASON_CODES = {
    "BRAND_MISMATCH",
    "IDENTITY_CONSISTENT",
    "MISSING_IDENTITY_FIELDS",
    "MULTIPLE_PLAUSIBLE_RECORDS",
    "NIE_MISMATCH",
    "NIE_NOT_FOUND",
    "OFFICIAL_NIE_AMBIGUOUS",
    "PACKAGE_MISMATCH",
    "PRODUCT_NAME_MISMATCH",
    "SHADE_MISMATCH",
    "SOURCE_UNVERIFIABLE",
    "SPF_MISMATCH",
    "STRENGTH_MISMATCH",
    "UNREADABLE",
    "VARIANT_MISMATCH",
}
REQUIRED_COLUMNS = {
    "sample_id",
    "source_url",
    "captured_at",
    "image_private_path",
    "image_sha256",
    "reviewer_id",
    "review_stage",
    "label",
    "reason_codes",
    "listing_nie",
    "listing_brand",
    "listing_product_name",
    "listing_package",
    "official_nie",
    "official_brand",
    "official_product_name",
    "official_package",
    "official_registrant",
    "readability",
    "notes",
}
TARGET_LABEL_COUNTS = {
    "MATCH": 50,
    "MISMATCH": 50,
    "INSUFFICIENT_EVIDENCE": 20,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL_REASON_CODES = {
    "MATCH": {"IDENTITY_CONSISTENT"},
    "MISMATCH": {
        "BRAND_MISMATCH",
        "NIE_MISMATCH",
        "NIE_NOT_FOUND",
        "PACKAGE_MISMATCH",
        "PRODUCT_NAME_MISMATCH",
        "SHADE_MISMATCH",
        "SPF_MISMATCH",
        "STRENGTH_MISMATCH",
        "VARIANT_MISMATCH",
    },
    "INSUFFICIENT_EVIDENCE": {
        "MISSING_IDENTITY_FIELDS",
        "MULTIPLE_PLAUSIBLE_RECORDS",
        "OFFICIAL_NIE_AMBIGUOUS",
        "SOURCE_UNVERIFIABLE",
        "UNREADABLE",
    },
}


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation dataset violates its CSV contract."""


def _reason_codes(value: str, row_number: int) -> list[str]:
    codes = [code.strip() for code in value.split(";") if code.strip()]
    if not codes:
        raise EvaluationDatasetError(f"row {row_number}: reason_codes is required")
    unknown = sorted(set(codes) - REASON_CODES)
    if unknown:
        raise EvaluationDatasetError(
            f"row {row_number}: unsupported reason codes: {', '.join(unknown)}"
        )
    return codes


def load_evaluation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise EvaluationDatasetError(f"missing required columns: {', '.join(missing)}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise EvaluationDatasetError("evaluation dataset contains no rows")
    return rows


def _validate_row(row: dict[str, str], row_number: int) -> None:
    for field in (
        "sample_id",
        "source_url",
        "captured_at",
        "image_private_path",
        "image_sha256",
        "reviewer_id",
    ):
        if not row[field]:
            raise EvaluationDatasetError(f"row {row_number}: {field} is required")
    if row["label"] not in LABELS:
        raise EvaluationDatasetError(f"row {row_number}: unsupported label")
    if row["review_stage"] not in REVIEW_STAGES:
        raise EvaluationDatasetError(f"row {row_number}: unsupported review_stage")
    if not _SHA256.fullmatch(row["image_sha256"]):
        raise EvaluationDatasetError(f"row {row_number}: invalid image_sha256")
    try:
        datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise EvaluationDatasetError(f"row {row_number}: captured_at must be ISO 8601") from error
    codes = set(_reason_codes(row["reason_codes"], row_number))
    if not codes <= _LABEL_REASON_CODES[row["label"]]:
        raise EvaluationDatasetError(
            f"row {row_number}: reason codes do not support label {row['label']}"
        )
    if row["readability"] not in READABILITY_VALUES:
        raise EvaluationDatasetError(f"row {row_number}: unsupported readability")


def validate_evaluation_dataset(
    rows: list[dict[str, str]], *, final: bool = False
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        _validate_row(row, row_number)
        grouped[row["sample_id"]].append(row)

    final_labels: Counter[str] = Counter()
    agreements = 0
    disagreements = 0
    resolved = 0
    incomplete: list[str] = []

    for sample_id, sample_rows in sorted(grouped.items()):
        initial_reviews = [row for row in sample_rows if row["review_stage"] == "initial"]
        resolutions = [row for row in sample_rows if row["review_stage"] == "resolution"]
        stable_fields = ("source_url", "captured_at", "image_private_path", "image_sha256")
        for field in stable_fields:
            if len({row[field] for row in sample_rows}) != 1:
                raise EvaluationDatasetError(
                    f"sample {sample_id}: inconsistent {field} across reviews"
                )
        if len(initial_reviews) != 2 or len({row["reviewer_id"] for row in initial_reviews}) != 2:
            incomplete.append(sample_id)
            continue
        if len(resolutions) > 1:
            raise EvaluationDatasetError(f"sample {sample_id}: at most one resolution is allowed")

        initial_labels = {row["label"] for row in initial_reviews}
        if len(initial_labels) == 1:
            agreements += 1
            final_label = next(iter(initial_labels))
        else:
            disagreements += 1
            if not resolutions:
                incomplete.append(sample_id)
                continue
            resolved += 1
            final_label = resolutions[0]["label"]
        final_labels[final_label] += 1

    if final:
        if incomplete:
            raise EvaluationDatasetError(
                "cannot finalize incomplete samples: " + ", ".join(incomplete)
            )
        observed = {label: final_labels[label] for label in sorted(LABELS)}
        if observed != TARGET_LABEL_COUNTS:
            raise EvaluationDatasetError(
                f"final composition mismatch: expected {TARGET_LABEL_COUNTS}, got {observed}"
            )

    completed = sum(final_labels.values())
    reviewed = agreements + disagreements
    agreement_rate = agreements / reviewed if reviewed else 0.0
    return {
        "rows": len(rows),
        "samples": len(grouped),
        "completed_samples": completed,
        "incomplete_sample_ids": incomplete,
        "final_label_counts": {label: final_labels[label] for label in sorted(LABELS)},
        "review_agreements": agreements,
        "review_disagreements": disagreements,
        "resolved_disagreements": resolved,
        "agreement_rate": agreement_rate,
        "final_validation_passed": final,
    }


def evaluation_report_json(path: Path, *, final: bool = False) -> str:
    return json.dumps(
        validate_evaluation_dataset(load_evaluation_rows(path), final=final),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
