from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field, field_validator


class SourceFile(BaseModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("manifest paths must be safe relative paths")
        return value


class SourceManifest(BaseModel):
    schema_version: int = Field(ge=1)
    dataset_id: str
    title: str
    owner: str
    source_url: str
    acquired_at: str
    license_status: str
    redistribution_allowed: bool
    allowed_use: str
    fields_used: list[str]
    preprocessing: list[str]
    files: list[SourceFile]


class ManifestVerificationError(ValueError):
    """Raised when local data does not match a committed source manifest."""


def load_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: SourceManifest, base_dir: Path) -> dict[str, object]:
    verified: list[str] = []
    for source_file in manifest.files:
        path = base_dir / Path(source_file.path)
        if not path.is_file():
            raise ManifestVerificationError(f"missing source file: {source_file.path}")
        if path.stat().st_size != source_file.size_bytes:
            raise ManifestVerificationError(f"size mismatch: {source_file.path}")
        if sha256_file(path) != source_file.sha256:
            raise ManifestVerificationError(f"SHA-256 mismatch: {source_file.path}")
        verified.append(source_file.path)
    return {
        "dataset_id": manifest.dataset_id,
        "verified_files": verified,
        "verified_file_count": len(verified),
    }


def verification_json(manifest_path: Path, base_dir: Path) -> str:
    result = verify_manifest(load_manifest(manifest_path), base_dir)
    return json.dumps(result, ensure_ascii=False, indent=2)
