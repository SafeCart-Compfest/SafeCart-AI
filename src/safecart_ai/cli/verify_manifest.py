from __future__ import annotations

import argparse
from pathlib import Path

from safecart_ai.data.manifests import verification_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify local files against a data manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("base_dir", type=Path)
    args = parser.parse_args()
    print(verification_json(args.manifest, args.base_dir))


if __name__ == "__main__":
    main()
