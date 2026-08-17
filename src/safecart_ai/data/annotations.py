from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LABELS = {"MATCH", "MISMATCH", "INSUFFICIENT_EVIDENCE"}
ROUNDS = {"independent", "adjudication"}
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
    "annotator_id",
    "annotation_round",
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


class AnnotationValidationError(ValueError):
    """Raised when a gold annotation file violates its contract."""


def _reason_codes(value: str, row_number: int) -> list[str]:
    codes = [code.strip() for code in value.split(";") if code.strip()]
    if not codes:
        raise AnnotationValidationError(f"row {row_number}: reason_codes is required")
    unknown = sorted(set(codes) - REASON_CODES)
    if unknown:
        raise AnnotationValidationError(
            f"row {row_number}: unsupported reason codes: {', '.join(unknown)}"
        )
    return codes


def load_annotation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise AnnotationValidationError(f"missing required columns: {', '.join(missing)}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise AnnotationValidationError("annotation file contains no rows")
    return rows


def _validate_row(row: dict[str, str], row_number: int) -> None:
    for field in (
        "sample_id",
        "source_url",
        "captured_at",
        "image_private_path",
        "image_sha256",
        "annotator_id",
    ):
        if not row[field]:
            raise AnnotationValidationError(f"row {row_number}: {field} is required")
    if row["label"] not in LABELS:
        raise AnnotationValidationError(f"row {row_number}: unsupported label")
    if row["annotation_round"] not in ROUNDS:
        raise AnnotationValidationError(f"row {row_number}: unsupported annotation_round")
    if not _SHA256.fullmatch(row["image_sha256"]):
        raise AnnotationValidationError(f"row {row_number}: invalid image_sha256")
    try:
        datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise AnnotationValidationError(
            f"row {row_number}: captured_at must be ISO 8601"
        ) from error
    codes = set(_reason_codes(row["reason_codes"], row_number))
    if not codes <= _LABEL_REASON_CODES[row["label"]]:
        raise AnnotationValidationError(
            f"row {row_number}: reason codes do not support label {row['label']}"
        )
    if row["readability"] not in READABILITY_VALUES:
        raise AnnotationValidationError(f"row {row_number}: unsupported readability")


def audit_annotations(rows: list[dict[str, str]], *, freeze: bool = False) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        _validate_row(row, row_number)
        grouped[row["sample_id"]].append(row)

    final_labels: Counter[str] = Counter()
    agreements = 0
    disagreements = 0
    adjudicated = 0
    incomplete: list[str] = []

    for sample_id, sample_rows in sorted(grouped.items()):
        independent = [row for row in sample_rows if row["annotation_round"] == "independent"]
        adjudications = [row for row in sample_rows if row["annotation_round"] == "adjudication"]
        stable_fields = ("source_url", "captured_at", "image_private_path", "image_sha256")
        for field in stable_fields:
            if len({row[field] for row in sample_rows}) != 1:
                raise AnnotationValidationError(
                    f"sample {sample_id}: inconsistent {field} across annotations"
                )
        if len(independent) != 2 or len({row["annotator_id"] for row in independent}) != 2:
            incomplete.append(sample_id)
            continue
        if len(adjudications) > 1:
            raise AnnotationValidationError(
                f"sample {sample_id}: at most one adjudication is allowed"
            )

        independent_labels = {row["label"] for row in independent}
        if len(independent_labels) == 1:
            agreements += 1
            final_label = next(iter(independent_labels))
        else:
            disagreements += 1
            if not adjudications:
                incomplete.append(sample_id)
                continue
            adjudicated += 1
            final_label = adjudications[0]["label"]
        final_labels[final_label] += 1

    if freeze:
        if incomplete:
            raise AnnotationValidationError(
                "cannot freeze incomplete samples: " + ", ".join(incomplete)
            )
        observed = {label: final_labels[label] for label in sorted(LABELS)}
        if observed != TARGET_LABEL_COUNTS:
            raise AnnotationValidationError(
                f"freeze composition mismatch: expected {TARGET_LABEL_COUNTS}, got {observed}"
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
        "independent_agreements": agreements,
        "independent_disagreements": disagreements,
        "adjudicated_disagreements": adjudicated,
        "raw_agreement_rate": agreement_rate,
        "freeze_validated": freeze,
    }


def annotation_audit_json(path: Path, *, freeze: bool = False) -> str:
    return json.dumps(
        audit_annotations(load_annotation_rows(path), freeze=freeze),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
