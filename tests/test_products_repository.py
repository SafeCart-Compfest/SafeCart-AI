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
