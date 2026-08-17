from pathlib import Path

from safecart_ai.cli.audit_bpom import audit_bpom


def test_audit_flags_ambiguous_official_records() -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"

    report = audit_bpom([fixture])

    assert report["commodity_rows"] == 3
    assert report["unique_nie"] == 2
    assert report["ambiguous_nie_count"] == 1
    assert "NA18241700093" in report["ambiguous_nie_examples"]
