import csv
import sys
from pathlib import Path

import pytest

from safecart_ai.cli.evaluate_retrieval import main as evaluate_retrieval_main
from safecart_ai.data.pairs import CatalogRecord, generate_pairs
from safecart_ai.retrieval.evaluation import evaluate_retrieval
from safecart_ai.retrieval.hybrid import HybridRetriever, RetrievalQuery, mean


def sample_records() -> list[CatalogRecord]:
    return [
        CatalogRecord(
            record_id="0" * 24,
            nie="NA1000",
            brand="Sun Lab",
            product_name="Daily Sunscreen SPF 30",
            package="30 mL",
            registrant="Example, PT",
        ),
        CatalogRecord(
            record_id="1" * 24,
            nie="NA1000",
            brand="Other Brand",
            product_name="Night Cream",
            package="20 g",
            registrant="Other, PT",
        ),
        CatalogRecord(
            record_id="2" * 24,
            nie="NA2000",
            brand="Sun Lab",
            product_name="Daily Sunscreen SPF 50",
            package="30 mL",
            registrant="Example, PT",
        ),
    ]


def write_catalog(path: Path, records: list[CatalogRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_id",
                "nie",
                "brand",
                "product_name",
                "package",
                "registrant",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "record_id": record.record_id,
                "nie": record.nie,
                "brand": record.brand,
                "product_name": record.product_name,
                "package": record.package,
                "registrant": record.registrant,
            }
            for record in records
        )


def test_exact_nie_preserves_ambiguous_candidates() -> None:
    retriever = HybridRetriever(sample_records())

    candidates = retriever.retrieve(RetrievalQuery(nie="NA 1000"), top_k=5)

    assert [candidate.record.record_id for candidate in candidates] == ["0" * 24, "1" * 24]
    assert all("EXACT_NIE" in candidate.sources for candidate in candidates)


def test_lexical_retrieval_ranks_matching_product_first() -> None:
    retriever = HybridRetriever(sample_records())

    candidates = retriever.retrieve(
        RetrievalQuery(brand="Sun Lab", product_name="daily sunscreen spf30"),
        top_k=2,
    )

    assert candidates[0].record.record_id == "0" * 24
    assert candidates[0].score > candidates[1].score


def test_retrieval_validates_top_k_and_empty_query() -> None:
    retriever = HybridRetriever(sample_records())

    assert retriever.retrieve(RetrievalQuery(), top_k=5) == []
    assert mean([]) == 0.0
    with pytest.raises(ValueError, match="positive"):
        retriever.retrieve(RetrievalQuery(), top_k=0)


def test_evaluation_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        CatalogRecord(
            record_id=f"{index:024x}",
            nie=f"NA{index:04}",
            brand=f"Brand {index % 3}",
            product_name=f"Distinct Product Alpha {index}",
            package="30 mL",
            registrant="Example, PT",
        )
        for index in range(30)
    ]
    catalog = tmp_path / "catalog.csv"
    pairs = tmp_path / "pairs.csv"
    write_catalog(catalog, records)
    generate_pairs(catalog, pairs)

    result = evaluate_retrieval(catalog, pairs, split="train", max_queries=10)

    assert result["query_count"] == 10
    assert result["recall_at_5"] == 1.0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safecart-ai-evaluate-retrieval",
            str(catalog),
            str(pairs),
            "--split",
            "train",
            "--max-queries",
            "5",
            "--lexical-only",
        ],
    )
    evaluate_retrieval_main()
