import csv
import sys
from pathlib import Path

import pytest

from safecart_ai.cli.validate_evaluation_dataset import main as validate_dataset_main
from safecart_ai.data.evaluation_dataset import (
    REQUIRED_COLUMNS,
    EvaluationDatasetError,
    load_evaluation_rows,
    validate_evaluation_dataset,
)


def review(
    sample_id: str,
    reviewer_id: str,
    label: str,
    reason_codes: str,
    *,
    review_stage: str = "initial",
) -> dict[str, str]:
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "sample_id": sample_id,
            "source_url": f"https://marketplace.invalid/{sample_id}",
            "captured_at": "2026-08-18T09:00:00+07:00",
            "image_private_path": f"private/{sample_id}.png",
            "image_sha256": "a" * 64,
            "reviewer_id": reviewer_id,
            "review_stage": review_stage,
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


def write_evaluation_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    columns = sorted(REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_validation_tracks_agreement_disagreement_and_resolution() -> None:
    rows = [
        review("match", "a", "MATCH", "IDENTITY_CONSISTENT"),
        review("match", "b", "MATCH", "IDENTITY_CONSISTENT"),
        review("reviewed", "a", "MATCH", "IDENTITY_CONSISTENT"),
        review("reviewed", "b", "MISMATCH", "PACKAGE_MISMATCH"),
        review(
            "reviewed",
            "judge",
            "MISMATCH",
            "PACKAGE_MISMATCH",
            review_stage="resolution",
        ),
    ]

    result = validate_evaluation_dataset(rows)

    assert result["completed_samples"] == 2
    assert result["final_label_counts"] == {
        "INSUFFICIENT_EVIDENCE": 0,
        "MATCH": 1,
        "MISMATCH": 1,
    }
    assert result["review_agreements"] == 1
    assert result["review_disagreements"] == 1
    assert result["resolved_disagreements"] == 1
    assert result["agreement_rate"] == 0.5


def test_validation_reports_incomplete_work_without_dividing_by_zero() -> None:
    result = validate_evaluation_dataset([review("pending", "a", "MATCH", "IDENTITY_CONSISTENT")])

    assert result["completed_samples"] == 0
    assert result["incomplete_sample_ids"] == ["pending"]
    assert result["agreement_rate"] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", "UNKNOWN", "unsupported label"),
        ("review_stage", "first", "unsupported review_stage"),
        ("image_sha256", "bad", "invalid image_sha256"),
        ("captured_at", "yesterday", "ISO 8601"),
        ("reason_codes", "MADE_UP", "unsupported reason codes"),
        ("reason_codes", "PACKAGE_MISMATCH", "do not support label"),
        ("readability", "GOOD", "unsupported readability"),
    ],
)
def test_validation_rejects_invalid_row_values(field: str, value: str, message: str) -> None:
    row = review("sample", "a", "MATCH", "IDENTITY_CONSISTENT")
    row[field] = value

    with pytest.raises(EvaluationDatasetError, match=message):
        validate_evaluation_dataset([row])


def test_validation_rejects_inconsistent_source_and_extra_resolution() -> None:
    rows = [
        review("sample", "a", "MATCH", "IDENTITY_CONSISTENT"),
        review("sample", "b", "MISMATCH", "NIE_MISMATCH"),
        review(
            "sample",
            "judge-a",
            "MISMATCH",
            "NIE_MISMATCH",
            review_stage="resolution",
        ),
        review(
            "sample",
            "judge-b",
            "MISMATCH",
            "NIE_MISMATCH",
            review_stage="resolution",
        ),
    ]
    with pytest.raises(EvaluationDatasetError, match="at most one"):
        validate_evaluation_dataset(rows)

    rows = rows[:2]
    rows[1]["image_sha256"] = "b" * 64
    with pytest.raises(EvaluationDatasetError, match="inconsistent image_sha256"):
        validate_evaluation_dataset(rows)


def test_final_validation_requires_complete_target_composition() -> None:
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
                    review(sample_id, "a", label, reason),
                    review(sample_id, "b", label, reason),
                ]
            )
            sample_number += 1

    result = validate_evaluation_dataset(rows, final=True)

    assert result["samples"] == 120
    assert result["final_validation_passed"] is True
    with pytest.raises(EvaluationDatasetError, match="composition mismatch"):
        validate_evaluation_dataset(rows[:-2], final=True)


def test_load_and_cli_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "evaluation-dataset.csv"
    rows = [
        review("sample", "a", "MATCH", "IDENTITY_CONSISTENT"),
        review("sample", "b", "MATCH", "IDENTITY_CONSISTENT"),
    ]
    write_evaluation_dataset(path, rows)

    assert len(load_evaluation_rows(path)) == 2
    monkeypatch.setattr(
        sys,
        "argv",
        ["safecart-ai-validate-evaluation-dataset", str(path)],
    )
    validate_dataset_main()
    assert '"completed_samples": 1' in capsys.readouterr().out

    path.write_text("sample_id\nonly\n", encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="missing required columns"):
        load_evaluation_rows(path)
