from collections.abc import Mapping

LABEL_TO_ID = {"MATCH": 0, "MISMATCH": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def _value(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    return "" if value is None else str(value).strip()


def format_pair(row: Mapping[str, object]) -> str:
    """Serialize one listing/official pair without free-form prompt generation."""
    listing = (
        f"nie={_value(row, 'listing_nie')} | "
        f"brand={_value(row, 'listing_brand')} | "
        f"name={_value(row, 'listing_product_name')} | "
        f"package={_value(row, 'listing_package')}"
    )
    official = (
        f"nie={_value(row, 'official_nie')} | "
        f"brand={_value(row, 'official_brand')} | "
        f"name={_value(row, 'official_product_name')} | "
        f"package={_value(row, 'official_package')} | "
        f"registrant={_value(row, 'official_registrant')}"
    )
    return f"[LISTING] {listing}\n[OFFICIAL] {official}"


def label_id(label: str) -> int:
    try:
        return LABEL_TO_ID[label]
    except KeyError as error:
        raise ValueError(f"Unsupported pair label: {label}") from error
