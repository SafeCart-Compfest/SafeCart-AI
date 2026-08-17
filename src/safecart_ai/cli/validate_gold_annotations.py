from __future__ import annotations

import argparse
from pathlib import Path

from safecart_ai.data.annotations import annotation_audit_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SafeCart gold annotations.")
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Require two independent labels, adjudication, and the 50/50/20 target.",
    )
    args = parser.parse_args()
    print(annotation_audit_json(args.annotations, freeze=args.freeze))


if __name__ == "__main__":
    main()
