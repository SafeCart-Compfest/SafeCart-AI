from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from safecart_ai.data.bpom import identity_signature
from safecart_ai.domain.normalization import normalize_nie

IDENTITY_FIELDS = ("product_name", "brand", "package", "registrant")


def audit_bpom(files: list[Path], commodity: str = "Kosmetik") -> dict[str, Any]:
    rows = 0
    commodity_rows = 0
    status_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    identities_by_nie: dict[str, set[tuple[str, ...]]] = defaultdict(set)

    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                rows += 1
                if row.get("nama_komoditi") != commodity:
                    continue
                commodity_rows += 1
                status_counts[row.get("status_produk") or "<missing>"] += 1
                for field in ("nie", *IDENTITY_FIELDS):
                    if not (row.get(field) or "").strip():
                        missing_counts[field] += 1
                nie = normalize_nie(row.get("nie"))
                if nie:
                    identity = identity_signature(
                        {
                            "product_name": row.get("nama_produk"),
                            "brand": row.get("brand"),
                            "package": row.get("kemasan"),
                            "registrant": row.get("pendaftar"),
                        }
                    )
                    identities_by_nie[nie].add(identity)

    ambiguous = {
        nie: [dict(zip(IDENTITY_FIELDS, identity, strict=True)) for identity in identities]
        for nie, identities in identities_by_nie.items()
        if len(identities) > 1
    }
    return {
        "input_files": [str(path) for path in files],
        "total_rows": rows,
        "commodity": commodity,
        "commodity_rows": commodity_rows,
        "unique_nie": len(identities_by_nie),
        "status_counts": dict(status_counts),
        "missing_counts": dict(missing_counts),
        "ambiguous_nie_count": len(ambiguous),
        "ambiguous_nie_examples": dict(list(sorted(ambiguous.items()))[:20]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BPOM CSV snapshots before modeling.")
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or directories")
    parser.add_argument("--commodity", default="Kosmetik")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    if not files:
        parser.error("no CSV files found")

    report = json.dumps(audit_bpom(files, args.commodity), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
