from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from safecart_ai.data.pairs import CatalogRecord
from safecart_ai.domain.normalization import normalize_nie, normalize_text

_GENERIC_TOKENS = {
    "cream",
    "gel",
    "serum",
    "skin",
    "body",
    "face",
    "with",
    "dan",
    "untuk",
}


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    nie: str | None = None
    brand: str | None = None
    product_name: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    record: CatalogRecord
    score: float
    sources: tuple[str, ...]


def retrieval_text(brand: str | None, product_name: str | None) -> str:
    return " ".join(part for part in (normalize_text(brand), normalize_text(product_name)) if part)


def retrieval_tokens(value: str | None) -> tuple[str, ...]:
    normalized = normalize_text(value)
    if not normalized:
        return ()
    return tuple(
        sorted(
            {
                token
                for token in normalized.split()
                if len(token) >= 4 and token not in _GENERIC_TOKENS
            }
        )
    )


class HybridRetriever:
    """Exact-NIE retrieval with blocked TF-IDF and RapidFuzz lexical ranking."""

    def __init__(self, records: list[CatalogRecord], max_pool: int = 5000) -> None:
        self._records = records
        self._max_pool = max_pool
        self._by_nie: dict[str, list[int]] = defaultdict(list)
        self._by_brand: dict[str, list[int]] = defaultdict(list)
        self._by_token: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self._by_nie[record.nie].append(index)
            brand = normalize_text(record.brand)
            if brand:
                self._by_brand[brand].append(index)
            for token in retrieval_tokens(record.product_name)[:4]:
                self._by_token[token].append(index)

    def _pool(self, query: RetrievalQuery) -> list[int]:
        pool: set[int] = set()
        brand = normalize_text(query.brand)
        if brand:
            pool.update(self._by_brand.get(brand, ()))
        tokens = retrieval_tokens(query.product_name)
        for token in sorted(tokens, key=lambda item: len(self._by_token.get(item, ())))[:3]:
            pool.update(self._by_token.get(token, ()))
        if len(pool) <= self._max_pool:
            return sorted(pool)

        query_text = retrieval_text(query.brand, query.product_name)
        preliminary = sorted(
            pool,
            key=lambda index: (
                -fuzz.WRatio(
                    query_text,
                    retrieval_text(
                        self._records[index].brand,
                        self._records[index].product_name,
                    ),
                ),
                self._records[index].record_id,
            ),
        )
        return preliminary[: self._max_pool]

    def retrieve(self, query: RetrievalQuery, top_k: int = 20) -> list[RetrievalCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be positive")

        exact_indices: list[int] = []
        nie = normalize_nie(query.nie)
        if nie:
            exact_indices = self._by_nie.get(nie, [])

        lexical_indices = self._pool(query)
        query_text = retrieval_text(query.brand, query.product_name)
        lexical_scores: dict[int, float] = {}
        if lexical_indices and query_text:
            documents = [
                query_text,
                *(
                    retrieval_text(self._records[index].brand, self._records[index].product_name)
                    for index in lexical_indices
                ),
            ]
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=False,
            )
            matrix = vectorizer.fit_transform(documents)
            similarities = (matrix[1:] @ matrix[0].T).toarray().ravel()
            for index, similarity in zip(lexical_indices, similarities, strict=True):
                candidate_text = retrieval_text(
                    self._records[index].brand,
                    self._records[index].product_name,
                )
                fuzzy = fuzz.WRatio(query_text, candidate_text) / 100
                lexical_scores[index] = 0.7 * float(similarity) + 0.3 * fuzzy

        combined: dict[int, RetrievalCandidate] = {}
        for index in lexical_indices:
            combined[index] = RetrievalCandidate(
                record=self._records[index],
                score=lexical_scores.get(index, 0.0),
                sources=("LEXICAL",),
            )
        for index in exact_indices:
            previous = combined.get(index)
            sources = ("EXACT_NIE",) if previous is None else ("EXACT_NIE", "LEXICAL")
            combined[index] = RetrievalCandidate(
                record=self._records[index],
                score=1.0,
                sources=sources,
            )

        ranked = sorted(
            combined.values(),
            key=lambda candidate: (-candidate.score, candidate.record.record_id),
        )
        return ranked[: min(top_k, len(ranked))]


def reciprocal_rank(candidates: list[RetrievalCandidate], target_record_id: str) -> float:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.record.record_id == target_record_id:
            return 1 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0
