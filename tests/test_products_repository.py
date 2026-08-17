from pathlib import Path

from safecart_ai.infrastructure.products import CsvOfficialProductRepository


def test_repository_preserves_all_records_for_one_nie() -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    repository = CsvOfficialProductRepository(fixture)

    records = repository.find_by_nie("BPOM NA18241700093")

    assert len(records) == 2
    assert {record.brand for record in records} == {"KARSKIN BEAUTY CARE", "PRONAFA"}
    assert repository.find_by_nie(None) == []
    assert repository.find_by_nie("NA00000000000") == []


def test_repository_retrieves_by_identity_when_nie_is_missing() -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    repository = CsvOfficialProductRepository(fixture)

    records = repository.find_candidates(
        nie=None,
        brand="Lumi Glow",
        product_name="Intensive Night Cream",
    )

    assert repository.product_count == 3
    assert records[0].nie == "NA18250116783"


def test_repository_does_not_ignore_an_unknown_supplied_nie() -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    repository = CsvOfficialProductRepository(fixture)

    records = repository.find_candidates(
        nie="NA00000000000",
        brand="LumiGlow",
        product_name="Intensive Night Cream",
    )

    assert records == []
