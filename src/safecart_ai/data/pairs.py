from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from safecart_ai.domain.normalization import normalize_nie, normalize_package, normalize_text

_VARIANT_MARKERS = re.compile(
    r"\b(?:spf|pa|shade|no|number)\s*[a-z0-9+.-]*\b|\b\d+(?:[.,]\d+)?(?:ml|g|gr)?\b",
    re.IGNORECASE,
)

PAIR_FIELDS = (
    "pair_id",
    "split",
    "label",
    "mutation_type",
    "group_id",
    "candidate_group_id",
    "family_id",
    "source_record_id",
    "candidate_record_id",
    "listing_nie",
    "listing_brand",
    "listing_product_name",
    "listing_package",
    "official_nie",
    "official_brand",
    "official_product_name",
    "official_package",
    "official_registrant",
)


class DataSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    SYNTHETIC_TEST = "synthetic_test"


class PairLabel(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"


class SplitLeakageError(ValueError):
    """Raised when a connected record group appears in multiple splits."""


@dataclass(frozen=True, slots=True)
class CatalogRecord:
    record_id: str
    nie: str
    brand: str | None
    product_name: str | None
    package: str | None
    registrant: str | None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> CatalogRecord:
        return cls(
            record_id=row["record_id"],
            nie=normalize_nie(row.get("nie")) or "",
            brand=row.get("brand") or None,
            product_name=row.get("product_name") or None,
            package=row.get("package") or None,
            registrant=row.get("registrant") or None,
        )


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    family_id: str
    group_id: str
    split: DataSplit


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def product_family_id(record: CatalogRecord) -> str:
    brand = normalize_text(record.brand) or normalize_text(record.registrant) or "unknown"
    name = normalize_text(record.product_name) or record.nie
    stem = normalize_text(_VARIANT_MARKERS.sub(" ", name)) or name
    return _digest(f"{brand}\x1f{stem}")[:20]


def _split_for(group_id: str, seed: int) -> DataSplit:
    bucket = int(_digest(f"{seed}:{group_id}")[:8], 16) % 100
    if bucket < 80:
        return DataSplit.TRAIN
    if bucket < 90:
        return DataSplit.DEV
    if bucket < 95:
        return DataSplit.CALIBRATION
    return DataSplit.SYNTHETIC_TEST


def assign_splits(records: list[CatalogRecord], seed: int = 42) -> dict[str, SplitAssignment]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_family: dict[str, int] = {}
    first_by_nie: dict[str, int] = {}
    families: list[str] = []
    for index, record in enumerate(records):
        family_id = product_family_id(record)
        families.append(family_id)
        if family_id in first_by_family:
            union(index, first_by_family[family_id])
        else:
            first_by_family[family_id] = index
        if record.nie in first_by_nie:
            union(index, first_by_nie[record.nie])
        else:
            first_by_nie[record.nie] = index

    record_ids_by_root: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        record_ids_by_root[find(index)].append(record.record_id)
    group_id_by_root = {
        root: _digest("|".join(sorted(record_ids)))[:20]
        for root, record_ids in record_ids_by_root.items()
    }

    return {
        record.record_id: SplitAssignment(
            family_id=families[index],
            group_id=group_id_by_root[find(index)],
            split=_split_for(group_id_by_root[find(index)], seed),
        )
        for index, record in enumerate(records)
    }


def load_catalog(path: Path) -> list[CatalogRecord]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [CatalogRecord.from_row(row) for row in csv.DictReader(stream)]


def _pick_candidate(
    source: CatalogRecord,
    pool: list[CatalogRecord],
    assignments: dict[str, SplitAssignment],
    excluded: set[str],
    require_different_family: bool,
) -> CatalogRecord | None:
    if not pool:
        return None
    start = int(source.record_id[:8], 16) % len(pool)
    source_family = assignments[source.record_id].family_id
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate.record_id in excluded:
            continue
        if require_different_family and assignments[candidate.record_id].family_id == source_family:
            continue
        return candidate
    return None


def _normalized_listing(record: CatalogRecord) -> dict[str, str | None]:
    return {
        "nie": record.nie,
        "brand": normalize_text(record.brand),
        "product_name": normalize_text(record.product_name),
        "package": normalize_package(record.package),
    }


def _pair_row(
    source: CatalogRecord,
    candidate: CatalogRecord,
    assignments: dict[str, SplitAssignment],
    label: PairLabel,
    mutation_type: str,
) -> dict[str, str | None]:
    source_assignment = assignments[source.record_id]
    candidate_assignment = assignments[candidate.record_id]
    listing = _normalized_listing(source)
    pair_id = _digest(f"{source.record_id}:{candidate.record_id}:{label.value}:{mutation_type}")[
        :24
    ]
    return {
        "pair_id": pair_id,
        "split": source_assignment.split.value,
        "label": label.value,
        "mutation_type": mutation_type,
        "group_id": source_assignment.group_id,
        "candidate_group_id": candidate_assignment.group_id,
        "family_id": source_assignment.family_id,
        "source_record_id": source.record_id,
        "candidate_record_id": candidate.record_id,
        "listing_nie": listing["nie"],
        "listing_brand": listing["brand"],
        "listing_product_name": listing["product_name"],
        "listing_package": listing["package"],
        "official_nie": candidate.nie,
        "official_brand": candidate.brand,
        "official_product_name": candidate.product_name,
        "official_package": candidate.package,
        "official_registrant": candidate.registrant,
    }


def generate_pairs(catalog: Path, output: Path, seed: int = 42) -> dict[str, object]:
    records = load_catalog(catalog)
    assignments = assign_splits(records, seed)
    by_split: dict[DataSplit, list[CatalogRecord]] = defaultdict(list)
    by_split_nie: dict[tuple[DataSplit, str], list[CatalogRecord]] = defaultdict(list)
    by_split_brand: dict[tuple[DataSplit, str], list[CatalogRecord]] = defaultdict(list)
    for record in records:
        split = assignments[record.record_id].split
        by_split[split].append(record)
        by_split_nie[(split, record.nie)].append(record)
        brand = normalize_text(record.brand)
        if brand:
            by_split_brand[(split, brand)].append(record)

    counts: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for source in records:
            assignment = assignments[source.record_id]
            writer.writerow(
                _pair_row(source, source, assignments, PairLabel.MATCH, "NORMALIZED_POSITIVE")
            )
            counts[f"{assignment.split.value}:MATCH"] += 1

            excluded = {source.record_id}
            candidates: list[tuple[CatalogRecord, str]] = []
            ambiguous = _pick_candidate(
                source,
                by_split_nie[(assignment.split, source.nie)],
                assignments,
                excluded,
                require_different_family=False,
            )
            if ambiguous:
                candidates.append((ambiguous, "AMBIGUOUS_NIE"))
                excluded.add(ambiguous.record_id)

            brand = normalize_text(source.brand)
            same_brand = _pick_candidate(
                source,
                by_split_brand[(assignment.split, brand)] if brand else [],
                assignments,
                excluded,
                require_different_family=True,
            )
            if same_brand:
                candidates.append((same_brand, "SAME_BRAND_HARD_NEGATIVE"))
                excluded.add(same_brand.record_id)

            while len(candidates) < 2:
                fallback = _pick_candidate(
                    source,
                    by_split[assignment.split],
                    assignments,
                    excluded,
                    require_different_family=True,
                )
                if fallback is None:
                    break
                candidates.append((fallback, "CROSS_BRAND_NEGATIVE"))
                excluded.add(fallback.record_id)

            for candidate, mutation_type in candidates:
                writer.writerow(
                    _pair_row(
                        source,
                        candidate,
                        assignments,
                        PairLabel.MISMATCH,
                        mutation_type,
                    )
                )
                counts[f"{assignment.split.value}:MISMATCH"] += 1

    validation = validate_pair_splits(output)
    return {
        "catalog_records": len(records),
        "pair_rows": sum(counts.values()),
        "seed": seed,
        "counts": dict(sorted(counts.items())),
        "validation": validation,
    }


def validate_pair_splits(path: Path) -> dict[str, int]:
    split_by_group: dict[str, str] = {}
    split_by_family: dict[str, str] = {}
    rows = 0
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            split = row["split"]
            for key, registry in (
                (row["group_id"], split_by_group),
                (row["candidate_group_id"], split_by_group),
                (row["family_id"], split_by_family),
            ):
                previous = registry.setdefault(key, split)
                if previous != split:
                    raise SplitLeakageError(f"{key} appears in {previous} and {split}")
    return {
        "pair_rows": rows,
        "unique_groups": len(split_by_group),
        "unique_families": len(split_by_family),
    }
