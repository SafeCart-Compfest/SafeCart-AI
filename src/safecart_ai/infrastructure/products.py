import csv
from pathlib import Path

from safecart_ai.data.pairs import CatalogRecord
from safecart_ai.domain.models import OfficialProduct
from safecart_ai.domain.normalization import normalize_nie
from safecart_ai.retrieval.hybrid import HybridRetriever, RetrievalQuery


class CsvOfficialProductRepository:
    """Small, deterministic CSV adapter for baselines and local demonstrations."""

    def __init__(self, path: Path) -> None:
        self._products_by_nie: dict[str, list[OfficialProduct]] = {}
        self._products_by_id: dict[str, OfficialProduct] = {}
        records: list[CatalogRecord] = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                record_id = row.get("record_id") or row.get("id") or str(index)
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
                    self._products_by_id[record_id] = product
                    records.append(
                        CatalogRecord(
                            record_id=record_id,
                            nie=key,
                            brand=product.brand,
                            product_name=product.product_name,
                            package=product.package,
                            registrant=product.registrant,
                        )
                    )
        self._retriever = HybridRetriever(records)

    @property
    def product_count(self) -> int:
        return len(self._products_by_id)

    def find_by_nie(self, nie: str | None) -> list[OfficialProduct]:
        key = normalize_nie(nie)
        return [] if key is None else list(self._products_by_nie.get(key, []))

    def find_candidates(
        self,
        *,
        nie: str | None,
        brand: str | None,
        product_name: str | None,
        top_k: int = 5,
    ) -> list[OfficialProduct]:
        normalized_nie = normalize_nie(nie)
        if normalized_nie:
            return self.find_by_nie(normalized_nie)
        candidates = self._retriever.retrieve(
            RetrievalQuery(brand=brand, product_name=product_name),
            top_k=top_k,
        )
        return [self._products_by_id[candidate.record.record_id] for candidate in candidates]
