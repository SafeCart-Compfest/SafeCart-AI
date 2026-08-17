from __future__ import annotations

import argparse
from pathlib import Path

from safecart_ai.data.evaluation_dataset import evaluation_report_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SafeCart evaluation dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require two reviews, conflict resolution, and the 50/50/20 target.",
    )
    args = parser.parse_args()
    print(evaluation_report_json(args.dataset, final=args.final))


if __name__ == "__main__":
    main()
