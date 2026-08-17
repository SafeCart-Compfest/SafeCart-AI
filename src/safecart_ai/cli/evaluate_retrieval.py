from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecart_ai.retrieval.evaluation import evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official-record retrieval.")
    parser.add_argument("catalog", type=Path)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_retrieval(
        args.catalog,
        args.pairs,
        split=args.split,
        max_queries=args.max_queries,
        seed=args.seed,
        lexical_only=args.lexical_only,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
