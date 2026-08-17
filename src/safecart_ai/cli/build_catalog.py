from __future__ import annotations

import argparse
import json
from pathlib import Path

from safecart_ai.data.bpom import build_catalog
from safecart_ai.data.manifests import load_manifest, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic BPOM catalog.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    verify_manifest(manifest, args.input_dir)
    files = [args.input_dir / source_file.path for source_file in manifest.files]
    result = build_catalog(files, args.output, manifest.acquired_at)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
