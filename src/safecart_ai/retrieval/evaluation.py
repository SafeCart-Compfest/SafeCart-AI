from __future__ import annotations

import csv
import hashlib
import random
import time
from pathlib import Path

from safecart_ai.data.pairs import PairLabel, load_catalog
from safecart_ai.retrieval.hybrid import HybridRetriever, RetrievalQuery, mean, reciprocal_rank

_BOOTSTRAP_SAMPLES = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(
    values: list[float], seed: int, samples: int = _BOOTSTRAP_SAMPLES
) -> dict[str, float]:
    """Return a deterministic percentile-bootstrap 95% confidence interval."""
    if not values:
        return {"low": 0.0, "high": 0.0}
    if samples < 1:
        raise ValueError("samples must be positive")

    randomizer = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean([values[randomizer.randrange(count)] for _ in range(count)]) for _ in range(samples)
    )
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return {"low": estimates[low_index], "high": estimates[high_index]}


def _evaluation_rows(
    pairs_path: Path, split: str, max_queries: int | None, seed: int
) -> list[dict[str, str]]:
    with pairs_path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["split"] == split and row["label"] == PairLabel.MATCH.value
        ]
    if max_queries is not None and len(rows) > max_queries:
        rows = random.Random(seed).sample(rows, max_queries)
    return rows


def evaluate_retrieval(
    catalog_path: Path,
    pairs_path: Path,
    split: str = "dev",
    max_queries: int | None = 1000,
    seed: int = 42,
    lexical_only: bool = False,
) -> dict[str, object]:
    started_at = time.perf_counter()
    records = load_catalog(catalog_path)
    retriever = HybridRetriever(records)
    index_seconds = time.perf_counter() - started_at
    rows = _evaluation_rows(pairs_path, split, max_queries, seed)
    hits_at_1: list[float] = []
    hits_at_5: list[float] = []
    hits_at_20: list[float] = []
    reciprocal_ranks: list[float] = []
    query_started_at = time.perf_counter()
    for row in rows:
        query = RetrievalQuery(
            nie=None if lexical_only else row["listing_nie"],
            brand=row["listing_brand"],
            product_name=row["listing_product_name"],
        )
        candidates = retriever.retrieve(query, top_k=20)
        target = row["source_record_id"]
        candidate_ids = [candidate.record.record_id for candidate in candidates]
        hits_at_1.append(float(target in candidate_ids[:1]))
        hits_at_5.append(float(target in candidate_ids[:5]))
        hits_at_20.append(float(target in candidate_ids[:20]))
        reciprocal_ranks.append(reciprocal_rank(candidates, target))

    query_count = len(rows)
    query_seconds = time.perf_counter() - query_started_at
    return {
        "split": split,
        "seed": seed,
        "lexical_only": lexical_only,
        "catalog_records": len(records),
        "query_count": query_count,
        "recall_at_1": mean(hits_at_1),
        "recall_at_5": mean(hits_at_5),
        "recall_at_20": mean(hits_at_20),
        "mean_reciprocal_rank": mean(reciprocal_ranks),
        "confidence_intervals_95": {
            "method": f"percentile bootstrap ({_BOOTSTRAP_SAMPLES} resamples)",
            "recall_at_1": bootstrap_mean_ci(hits_at_1, seed),
            "recall_at_5": bootstrap_mean_ci(hits_at_5, seed),
            "recall_at_20": bootstrap_mean_ci(hits_at_20, seed),
            "mean_reciprocal_rank": bootstrap_mean_ci(reciprocal_ranks, seed),
        },
        "timing_seconds": {
            "index_build": index_seconds,
            "queries": query_seconds,
            "per_query": query_seconds / query_count if query_count else 0.0,
        },
        "inputs": {
            "catalog_sha256": _sha256(catalog_path),
            "pairs_sha256": _sha256(pairs_path),
        },
    }
