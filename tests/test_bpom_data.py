import csv
from pathlib import Path

from safecart_ai.data.bpom import (
    build_catalog,
    canonicalize_row,
    clean_optional,
    identity_signature,
)


def test_canonicalize_row_cleans_nulls_and_nie() -> None:
    result = canonicalize_row(
        {
            "nie": "BPOM RI NA 18-2501-16783",
            "brand": " LumiGlow ",
            "nama_produk": "Night Cream",
            "kemasan": "NULL",
            "pendaftar": "LUMI INDONESIA, PT",
            "status_produk": "Berlaku",
            "masa_berlaku": "2028-10-15",
        },
        "2026-08-17",
    )

    assert result is not None
    assert result["nie"] == "NA18250116783"
    assert result["brand"] == "LumiGlow"
    assert result["package"] is None


def test_canonicalize_row_rejects_missing_nie() -> None:
    assert canonicalize_row({"nie": ""}, "2026-08-17") is None
    assert clean_optional(None) is None


def test_identity_signature_normalizes_equivalent_values() -> None:
    left = {
        "product_name": "Brightening  Cream",
        "brand": "LUMI-GLOW",
        "package": "Pot, 30 gram",
        "registrant": "Example, PT",
    }
    right = {
        "product_name": "brightening cream",
        "brand": "lumi glow",
        "package": "30 g",
        "registrant": "EXAMPLE PT",
    }

    assert identity_signature(left) == identity_signature(right)


def test_build_catalog_collapses_duplicates_and_preserves_ambiguity(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    duplicated = tmp_path / "duplicated.csv"
    duplicated.write_bytes(source.read_bytes() + source.read_bytes().splitlines(keepends=True)[1])
    output = tmp_path / "catalog.csv"

    report = build_catalog([duplicated], output, "2026-08-17")
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert report == {
        "source_rows": 4,
        "canonical_records": 3,
        "collapsed_duplicate_rows": 1,
        "ambiguous_nie_count": 1,
    }
    assert len(rows) == 3
    assert rows[0]["source_snapshot_at"] == "2026-08-17"
    assert sum(int(row["source_row_count"]) for row in rows) == 4
