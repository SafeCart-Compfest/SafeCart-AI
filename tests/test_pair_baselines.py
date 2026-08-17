import csv
import sys
from pathlib import Path

import pytest

from safecart_ai.cli.evaluate_pair_baseline import main as evaluate_pair_baseline_main
from safecart_ai.matching.baselines import (
    PairBaseline,
    evaluate_pair_baseline,
    predict_deterministic,
    predict_exact_nie,
)


def pair_row(**overrides: str) -> dict[str, str]:
    row = {
        "pair_id": "pair-1",
        "split": "dev",
        "label": "MATCH",
        "mutation_type": "NORMALIZED_POSITIVE",
        "listing_nie": "NA 123",
        "listing_brand": "Lumi Glow",
        "listing_product_name": "Day Cream SPF 30",
        "listing_package": "30 ml",
        "official_nie": "NA123",
        "official_brand": "Lumi Glow",
        "official_product_name": "Day Cream SPF 30",
        "official_package": "30 mL",
    }
    row.update(overrides)
    return row


def write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_exact_nie_requires_same_nonempty_identifier() -> None:
    assert predict_exact_nie(pair_row()) == 0
    assert predict_exact_nie(pair_row(listing_nie="")) == 1
    assert predict_exact_nie(pair_row(official_nie="NA999")) == 1


def test_deterministic_baseline_detects_field_contradictions() -> None:
    assert predict_deterministic(pair_row()) == 0
    assert predict_deterministic(pair_row(official_product_name="Night Cream")) == 1
    assert predict_deterministic(pair_row(official_package="50 g")) == 1
    assert predict_deterministic(pair_row(official_brand="")) == 0


def test_evaluation_reports_mutation_breakdown_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs = tmp_path / "pairs.csv"
    output = tmp_path / "metrics.json"
    write_pairs(
        pairs,
        [
            pair_row(),
            pair_row(
                pair_id="pair-2",
                label="MISMATCH",
                mutation_type="SAME_BRAND_HARD_NEGATIVE",
                official_nie="NA999",
            ),
        ],
    )

    result = evaluate_pair_baseline(pairs, PairBaseline.EXACT_NIE)

    assert result["query_count"] == 2
    assert result["by_mutation"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safecart-ai-evaluate-pair-baseline",
            str(pairs),
            "--baseline",
            "deterministic",
            "--output",
            str(output),
        ],
    )
    evaluate_pair_baseline_main()
    assert output.is_file()


def test_evaluation_rejects_missing_split(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.csv"
    write_pairs(pairs, [pair_row()])

    with pytest.raises(ValueError, match="No pair rows"):
        evaluate_pair_baseline(pairs, PairBaseline.EXACT_NIE, split="calibration")
