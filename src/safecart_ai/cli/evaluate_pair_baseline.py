from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecart_ai.matching.baselines import PairBaseline, evaluate_pair_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a deterministic pair baseline.")
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--baseline", type=PairBaseline, choices=list(PairBaseline), required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_pair_baseline(args.pairs, args.baseline, args.split)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
