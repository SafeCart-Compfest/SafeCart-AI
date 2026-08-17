from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/SafeCart-Compfest/SafeCart-AI.git"
GIT_SHA = "1bcbe4c19723f91d449001fc7a1e2f35f9d91f8f"
EXPECTED_PAIRS_SHA256 = "a2016a60458c3bed4edcc8ad5902ead2dd2f615ebcc2f0166956829ee4ba42f1"
KAGGLE_INPUT_PATH = Path("/kaggle/input")
REPOSITORY_PATH = Path("/kaggle/working/SafeCart-AI")
OUTPUT_PATH = Path("/kaggle/working/safecart-matcher-smoke")


def run(*command: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_pairs_path() -> Path:
    candidates = sorted(KAGGLE_INPUT_PATH.rglob("product-pairs.csv"))
    if len(candidates) != 1:
        mounted_files = sorted(
            path.relative_to(KAGGLE_INPUT_PATH).as_posix()
            for path in KAGGLE_INPUT_PATH.rglob("*")
            if path.is_file()
        )
        raise FileNotFoundError(
            f"Expected one product-pairs.csv, found {len(candidates)}; "
            f"mounted files: {mounted_files[:20]}"
        )
    return candidates[0]


def main() -> None:
    pairs_path = find_pairs_path()
    actual_hash = sha256(pairs_path)
    if actual_hash != EXPECTED_PAIRS_SHA256:
        raise RuntimeError(f"Pair dataset hash mismatch: {actual_hash}")

    run("git", "clone", "--filter=blob:none", REPOSITORY_URL, str(REPOSITORY_PATH))
    run("git", "-C", str(REPOSITORY_PATH), "checkout", "--detach", GIT_SHA)
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--editable",
        f"{REPOSITORY_PATH}[ml]",
    )
    run(
        sys.executable,
        "-c",
        "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
    )

    environment = os.environ.copy()
    environment["SAFECART_GIT_SHA"] = GIT_SHA
    run(
        sys.executable,
        "-m",
        "safecart_ai.cli.train_matcher",
        str(pairs_path),
        "--config",
        str(REPOSITORY_PATH / "training/configs/distilmbert-smoke.toml"),
        "--output",
        str(OUTPUT_PATH),
        env=environment,
    )

    manifest = json.loads((OUTPUT_PATH / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest["git_sha"] != GIT_SHA:
        raise RuntimeError("Run manifest Git SHA does not match the pinned source")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
