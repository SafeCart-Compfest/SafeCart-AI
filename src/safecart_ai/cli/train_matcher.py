from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the SafeCart pair matcher.")
    parser.add_argument("pairs", type=Path, help="Leakage-safe pair CSV.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.pairs.is_file():
        parser.error(f"Pair dataset does not exist: {args.pairs}")
    if not args.config.is_file():
        parser.error(f"Training config does not exist: {args.config}")

    from safecart_ai.matching.hf_training import run_training

    run_training(args.pairs, args.config, args.output)


if __name__ == "__main__":
    main()
