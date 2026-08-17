import csv
import sys
from pathlib import Path

import pytest

from safecart_ai.cli.validate_gold_annotations import main as validate_annotations_main
from safecart_ai.data.annotations import (
    REQUIRED_COLUMNS,
    AnnotationValidationError,
    audit_annotations,
    load_annotation_rows,
)


def annotation(
    sample_id: str,
    annotator_id: str,
    label: str,
    reason_codes: str,
    *,
    annotation_round: str = "independent",
) -> dict[str, str]:
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "sample_id": sample_id,
            "source_url": f"https://marketplace.invalid/{sample_id}",
            "captured_at": "2026-08-18T09:00:00+07:00",
            "image_private_path": f"private/{sample_id}.png",
            "image_sha256": "a" * 64,
            "annotator_id": annotator_id,
            "annotation_round": annotation_round,
            "label": label,
            "reason_codes": reason_codes,
            "listing_nie": "NA123",
            "listing_brand": "Lumi",
            "listing_product_name": "Day Cream",
            "listing_package": "30 g",
            "official_nie": "NA123",
            "official_brand": "Lumi",
            "official_product_name": "Day Cream",
            "official_package": "30 g",
            "official_registrant": "Example, PT",
            "readability": "READABLE",
            "notes": "fixture",
        }
    )
    return row


def write_annotations(path: Path, rows: list[dict[str, str]]) -> None:
    columns = sorted(REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_tracks_agreement_disagreement_and_adjudication() -> None:
    rows = [
        annotation("match", "a", "MATCH", "IDENTITY_CONSISTENT"),
        annotation("match", "b", "MATCH", "IDENTITY_CONSISTENT"),
        annotation("reviewed", "a", "MATCH", "IDENTITY_CONSISTENT"),
        annotation("reviewed", "b", "MISMATCH", "PACKAGE_MISMATCH"),
        annotation(
            "reviewed",
            "judge",
            "MISMATCH",
            "PACKAGE_MISMATCH",
            annotation_round="adjudication",
        ),
    ]

    result = audit_annotations(rows)

    assert result["completed_samples"] == 2
    assert result["final_label_counts"] == {
        "INSUFFICIENT_EVIDENCE": 0,
        "MATCH": 1,
        "MISMATCH": 1,
    }
    assert result["independent_agreements"] == 1
    assert result["independent_disagreements"] == 1
    assert result["adjudicated_disagreements"] == 1
    assert result["raw_agreement_rate"] == 0.5


def test_audit_reports_incomplete_work_without_dividing_by_zero() -> None:
    result = audit_annotations([annotation("pending", "a", "MATCH", "IDENTITY_CONSISTENT")])

    assert result["completed_samples"] == 0
    assert result["incomplete_sample_ids"] == ["pending"]
    assert result["raw_agreement_rate"] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", "UNKNOWN", "unsupported label"),
        ("annotation_round", "first", "unsupported annotation_round"),
        ("image_sha256", "bad", "invalid image_sha256"),
        ("captured_at", "yesterday", "ISO 8601"),
        ("reason_codes", "MADE_UP", "unsupported reason codes"),
        ("reason_codes", "PACKAGE_MISMATCH", "do not support label"),
        ("readability", "GOOD", "unsupported readability"),
    ],
)
def test_audit_rejects_invalid_row_values(field: str, value: str, message: str) -> None:
    row = annotation("sample", "a", "MATCH", "IDENTITY_CONSISTENT")
    row[field] = value

    with pytest.raises(AnnotationValidationError, match=message):
        audit_annotations([row])


def test_audit_rejects_inconsistent_provenance_and_extra_adjudication() -> None:
    rows = [
        annotation("sample", "a", "MATCH", "IDENTITY_CONSISTENT"),
        annotation("sample", "b", "MISMATCH", "NIE_MISMATCH"),
        annotation(
            "sample",
            "judge-a",
            "MISMATCH",
            "NIE_MISMATCH",
            annotation_round="adjudication",
        ),
        annotation(
            "sample",
            "judge-b",
            "MISMATCH",
            "NIE_MISMATCH",
            annotation_round="adjudication",
        ),
    ]
    with pytest.raises(AnnotationValidationError, match="at most one"):
        audit_annotations(rows)

    rows = rows[:2]
    rows[1]["image_sha256"] = "b" * 64
    with pytest.raises(AnnotationValidationError, match="inconsistent image_sha256"):
        audit_annotations(rows)


def test_freeze_requires_complete_target_composition() -> None:
    rows: list[dict[str, str]] = []
    targets = [
        ("MATCH", "IDENTITY_CONSISTENT", 50),
        ("MISMATCH", "VARIANT_MISMATCH", 50),
        ("INSUFFICIENT_EVIDENCE", "UNREADABLE", 20),
    ]
    sample_number = 0
    for label, reason, count in targets:
        for _ in range(count):
            sample_id = f"sample-{sample_number:03d}"
            rows.extend(
                [
                    annotation(sample_id, "a", label, reason),
                    annotation(sample_id, "b", label, reason),
                ]
            )
            sample_number += 1

    result = audit_annotations(rows, freeze=True)

    assert result["samples"] == 120
    assert result["freeze_validated"] is True
    with pytest.raises(AnnotationValidationError, match="composition mismatch"):
        audit_annotations(rows[:-2], freeze=True)


def test_load_and_cli_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "annotations.csv"
    rows = [
        annotation("sample", "a", "MATCH", "IDENTITY_CONSISTENT"),
        annotation("sample", "b", "MATCH", "IDENTITY_CONSISTENT"),
    ]
    write_annotations(path, rows)

    assert len(load_annotation_rows(path)) == 2
    monkeypatch.setattr(
        sys,
        "argv",
        ["safecart-ai-validate-gold-annotations", str(path)],
    )
    validate_annotations_main()
    assert '"completed_samples": 1' in capsys.readouterr().out

    path.write_text("sample_id\nonly\n", encoding="utf-8")
    with pytest.raises(AnnotationValidationError, match="missing required columns"):
        load_annotation_rows(path)
