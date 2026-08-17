from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecart_ai.data.pairs import generate_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe product pairs.")
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate_pairs(args.catalog, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
