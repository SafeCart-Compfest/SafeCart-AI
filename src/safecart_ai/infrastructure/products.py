import csv
from pathlib import Path

from safecart_ai.domain.models import OfficialProduct
from safecart_ai.domain.normalization import normalize_nie


class CsvOfficialProductRepository:
    """Small, deterministic CSV adapter for baselines and local demonstrations."""

    def __init__(self, path: Path) -> None:
        self._products_by_nie: dict[str, list[OfficialProduct]] = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                product = OfficialProduct(
                    nie=row.get("nie"),
                    brand=row.get("brand"),
                    product_name=row.get("nama_produk") or row.get("product_name"),
                    package=row.get("kemasan") or row.get("package"),
                    registrant=row.get("pendaftar") or row.get("registrant"),
                    registration_status=row.get("status_produk") or row.get("registration_status"),
                    valid_until=row.get("masa_berlaku") or row.get("valid_until"),
                    source=row.get("source") or "BPOM",
                    source_snapshot_at=row.get("source_snapshot_at"),
                )
                key = normalize_nie(product.nie)
                if key:
                    self._products_by_nie.setdefault(key, []).append(product)

    def find_by_nie(self, nie: str | None) -> list[OfficialProduct]:
        key = normalize_nie(nie)
        return [] if key is None else list(self._products_by_nie.get(key, []))
