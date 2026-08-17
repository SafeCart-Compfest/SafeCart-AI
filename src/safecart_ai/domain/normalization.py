import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_PACKAGE = re.compile(r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<unit>kg|g|gr|gram|ml|l)\b", re.I)


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = _WHITESPACE.sub(" ", _NON_ALNUM.sub(" ", ascii_value.lower())).strip()
    return normalized or None


def normalize_nie(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    normalized = re.sub(r"^(?:BPOMRI|BPOM)", "", normalized)
    return normalized or None


def normalize_package(value: str | None) -> str | None:
    if value is None:
        return None
    match = _PACKAGE.search(value)
    if match is None:
        return normalize_text(value)
    amount = match.group("amount").replace(",", ".")
    unit = match.group("unit").lower()
    unit = {"gr": "g", "gram": "g"}.get(unit, unit)
    return f"{amount} {unit}"


def extract_packages(value: str | None) -> set[str]:
    if value is None:
        return set()
    packages = set()
    for match in _PACKAGE.finditer(value):
        amount = match.group("amount").replace(",", ".")
        unit = match.group("unit").lower()
        unit = {"gr": "g", "gram": "g"}.get(unit, unit)
        packages.add(f"{amount} {unit}")
    return packages
