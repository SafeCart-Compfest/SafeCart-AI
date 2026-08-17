from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from safecart_ai.domain.normalization import normalize_nie, normalize_package, normalize_text

IDENTITY_FIELDS = ("product_name", "brand", "package", "registrant")
CANONICAL_FIELDS = (
    "record_id",
    "nie",
    "brand",
    "product_name",
    "package",
    "registrant",
    "registration_status",
    "valid_until",
    "source",
    "source_snapshot_at",
    "source_row_count",
)


class CanonicalRow(TypedDict):
    nie: str
    brand: str | None
    product_name: str | None
    package: str | None
    registrant: str | None
    registration_status: str | None
    valid_until: str | None
    source: str
    source_snapshot_at: str


def clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return None if not cleaned or cleaned.upper() == "NULL" else cleaned


@dataclass(frozen=True, slots=True)
class CanonicalProduct:
    record_id: str
    nie: str
    brand: str | None
    product_name: str | None
    package: str | None
    registrant: str | None
    registration_status: str | None
    valid_until: str | None
    source: str
    source_snapshot_at: str
    source_row_count: int = 1

    def to_row(self) -> dict[str, str | int | None]:
        return asdict(self)


def identity_signature(row: Mapping[str, str | None]) -> tuple[str, ...]:
    return (
        normalize_text(row.get("product_name")) or "",
        normalize_text(row.get("brand")) or "",
        normalize_package(row.get("package")) or "",
        normalize_text(row.get("registrant")) or "",
    )


def canonicalize_row(row: dict[str, str], snapshot_at: str) -> CanonicalRow | None:
    nie = normalize_nie(row.get("nie"))
    if not nie:
        return None
    return {
        "nie": nie,
        "brand": clean_optional(row.get("brand")),
        "product_name": clean_optional(row.get("nama_produk")),
        "package": clean_optional(row.get("kemasan")),
        "registrant": clean_optional(row.get("pendaftar")),
        "registration_status": clean_optional(row.get("status_produk")),
        "valid_until": clean_optional(row.get("masa_berlaku")),
        "source": "BPOM",
        "source_snapshot_at": snapshot_at,
    }


def record_id_for(row: CanonicalRow) -> str:
    identity: dict[str, str | None] = {
        "product_name": row["product_name"],
        "brand": row["brand"],
        "package": row["package"],
        "registrant": row["registrant"],
    }
    payload = "\x1f".join((row["nie"], *identity_signature(identity)))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def iter_cosmetic_rows(paths: Iterable[Path]) -> Iterator[dict[str, str]]:
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("nama_komoditi") == "Kosmetik":
                    yield row


def build_catalog(paths: Iterable[Path], output: Path, snapshot_at: str) -> dict[str, int]:
    unique: dict[str, tuple[CanonicalRow, int]] = {}
    source_rows = 0
    for source_row in iter_cosmetic_rows(paths):
        source_rows += 1
        canonical = canonicalize_row(source_row, snapshot_at)
        if canonical is None:
            continue
        record_id = record_id_for(canonical)
        previous = unique.get(record_id)
        unique[record_id] = (canonical, 1 if previous is None else previous[1] + 1)

    products = [
        CanonicalProduct(record_id=record_id, source_row_count=count, **row)
        for record_id, (row, count) in unique.items()
    ]
    products.sort(key=lambda product: (product.nie, product.record_id))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(product.to_row() for product in products)

    ambiguous_nie = 0
    previous_nie: str | None = None
    count_for_nie = 0
    for product in products:
        if product.nie != previous_nie:
            ambiguous_nie += int(count_for_nie > 1)
            previous_nie = product.nie
            count_for_nie = 0
        count_for_nie += 1
    ambiguous_nie += int(count_for_nie > 1)

    return {
        "source_rows": source_rows,
        "canonical_records": len(products),
        "collapsed_duplicate_rows": source_rows - len(products),
        "ambiguous_nie_count": ambiguous_nie,
    }
