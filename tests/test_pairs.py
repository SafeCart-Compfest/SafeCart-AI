import csv
import sys
from pathlib import Path

import pytest

from safecart_ai.cli.generate_pairs import main as generate_pairs_main
from safecart_ai.data.pairs import (
    CatalogRecord,
    SplitLeakageError,
    assign_splits,
    generate_pairs,
    product_family_id,
    validate_pair_splits,
)


def record(
    record_id: str,
    nie: str,
    brand: str,
    name: str,
    package: str = "30 mL",
) -> CatalogRecord:
    return CatalogRecord(
        record_id=record_id,
        nie=nie,
        brand=brand,
        product_name=name,
        package=package,
        registrant="Example, PT",
    )


def write_catalog(path: Path, records: list[CatalogRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_id",
                "nie",
                "brand",
                "product_name",
                "package",
                "registrant",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "record_id": item.record_id,
                "nie": item.nie,
                "brand": item.brand,
                "product_name": item.product_name,
                "package": item.package,
                "registrant": item.registrant,
            }
            for item in records
        )


def test_product_family_groups_numeric_variants() -> None:
    first = record("0" * 24, "NA1", "Sun Lab", "Daily Sunscreen SPF 30")
    second = record("1" * 24, "NA2", "Sun Lab", "Daily Sunscreen SPF 50")

    assert product_family_id(first) == product_family_id(second)


def test_split_connects_families_and_ambiguous_nie() -> None:
    records = [
        record("0" * 24, "NA1", "Sun Lab", "Daily Sunscreen SPF 30"),
        record("1" * 24, "NA2", "Sun Lab", "Daily Sunscreen SPF 50"),
        record("2" * 24, "NA2", "Other", "Different Product"),
    ]

    assignments = assign_splits(records)

    assert len({assignment.group_id for assignment in assignments.values()}) == 1
    assert len({assignment.split for assignment in assignments.values()}) == 1


def test_generate_pairs_is_deterministic_and_leakage_safe(tmp_path: Path) -> None:
    records = [
        record(f"{index:024x}", f"NA{index:04}", f"Brand {index % 3}", f"Product {index}")
        for index in range(30)
    ]
    catalog = tmp_path / "catalog.csv"
    first_output = tmp_path / "pairs-1.csv"
    second_output = tmp_path / "pairs-2.csv"
    write_catalog(catalog, records)

    first_report = generate_pairs(catalog, first_output)
    second_report = generate_pairs(catalog, second_output)

    assert first_report == second_report
    assert first_output.read_bytes() == second_output.read_bytes()
    pair_rows = first_report["pair_rows"]
    assert isinstance(pair_rows, int)
    assert pair_rows > len(records)
    assert validate_pair_splits(first_output)["pair_rows"] == first_report["pair_rows"]


def test_generate_pairs_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        record(f"{index:024x}", f"NA{index:04}", f"Brand {index % 2}", f"Product {index}")
        for index in range(8)
    ]
    catalog = tmp_path / "catalog.csv"
    output = tmp_path / "pairs.csv"
    write_catalog(catalog, records)
    monkeypatch.setattr(
        sys,
        "argv",
        ["safecart-ai-generate-pairs", str(catalog), str(output), "--seed", "42"],
    )

    generate_pairs_main()

    assert output.is_file()
    assert '"seed": 42' in capsys.readouterr().out


def test_validator_rejects_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(
        "group_id,candidate_group_id,family_id,split\n"
        "shared,candidate-a,family-a,train\n"
        "shared,candidate-b,family-a,dev\n",
        encoding="utf-8",
    )

    with pytest.raises(SplitLeakageError, match="train and dev"):
        validate_pair_splits(path)
