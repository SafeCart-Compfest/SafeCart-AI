import pytest

from safecart_ai.domain.normalization import (
    extract_packages,
    normalize_nie,
    normalize_package,
    normalize_text,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BPOM NA 18 2501 16783", "NA18250116783"),
        ("na18250116783", "NA18250116783"),
        (None, None),
    ],
)
def test_normalize_nie(raw: str | None, expected: str | None) -> None:
    assert normalize_nie(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("30GR", "30 g"), ("30 Gram", "30 g"), ("30 ml", "30 ml"), (None, None)],
)
def test_normalize_package(raw: str | None, expected: str | None) -> None:
    assert normalize_package(raw) == expected


def test_normalize_text_removes_punctuation_and_accents() -> None:
    assert normalize_text("Crème — Night!") == "creme night"


def test_extract_packages_preserves_multiple_official_sizes() -> None:
    assert extract_packages("Tube, Dus 15 g, Tube, Dus 30 gram") == {"15 g", "30 g"}


def test_normalization_handles_empty_and_unstructured_values() -> None:
    assert normalize_text("---") is None
    assert normalize_package("one travel size") == "one travel size"
    assert extract_packages(None) == set()
