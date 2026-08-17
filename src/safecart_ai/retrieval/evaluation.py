from __future__ import annotations

import csv
import random
from pathlib import Path

from safecart_ai.data.pairs import PairLabel, load_catalog
from safecart_ai.retrieval.hybrid import HybridRetriever, RetrievalQuery, mean, reciprocal_rank


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
) -> dict[str, float | int | str | bool]:
    records = load_catalog(catalog_path)
    retriever = HybridRetriever(records)
    rows = _evaluation_rows(pairs_path, split, max_queries, seed)
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_20 = 0
    reciprocal_ranks: list[float] = []
    for row in rows:
        query = RetrievalQuery(
            nie=None if lexical_only else row["listing_nie"],
            brand=row["listing_brand"],
            product_name=row["listing_product_name"],
        )
        candidates = retriever.retrieve(query, top_k=20)
        target = row["source_record_id"]
        candidate_ids = [candidate.record.record_id for candidate in candidates]
        hits_at_1 += int(target in candidate_ids[:1])
        hits_at_5 += int(target in candidate_ids[:5])
        hits_at_20 += int(target in candidate_ids[:20])
        reciprocal_ranks.append(reciprocal_rank(candidates, target))

    query_count = len(rows)
    denominator = query_count or 1
    return {
        "split": split,
        "seed": seed,
        "lexical_only": lexical_only,
        "query_count": query_count,
        "recall_at_1": hits_at_1 / denominator,
        "recall_at_5": hits_at_5 / denominator,
        "recall_at_20": hits_at_20 / denominator,
        "mean_reciprocal_rank": mean(reciprocal_ranks),
    }
