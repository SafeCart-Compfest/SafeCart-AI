import hashlib
import sys
from pathlib import Path

import pytest

from safecart_ai.cli.build_catalog import main as build_catalog_main
from safecart_ai.cli.verify_manifest import main as verify_manifest_main
from safecart_ai.data.manifests import (
    ManifestVerificationError,
    SourceFile,
    SourceManifest,
    verify_manifest,
)


def manifest_for(path: Path) -> SourceManifest:
    content = path.read_bytes()
    return SourceManifest(
        schema_version=1,
        dataset_id="fixture",
        title="Fixture",
        owner="SafeCart tests",
        source_url="https://example.invalid/fixture",
        acquired_at="2026-08-17",
        license_status="Synthetic test data",
        redistribution_allowed=True,
        allowed_use="Tests",
        fields_used=["value"],
        preprocessing=[],
        files=[
            SourceFile(
                path=path.name,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        ],
    )


def test_manifest_verifies_size_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\nfixture\n", encoding="utf-8")

    result = verify_manifest(manifest_for(source), tmp_path)

    assert result["verified_file_count"] == 1


def test_manifest_rejects_changed_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\nfixture\n", encoding="utf-8")
    manifest = manifest_for(source)
    source.write_text("value\nchanged\n", encoding="utf-8")

    with pytest.raises(ManifestVerificationError, match="mismatch"):
        verify_manifest(manifest, tmp_path)


def test_manifest_rejects_parent_path() -> None:
    with pytest.raises(ValueError, match="safe relative"):
        SourceFile(path="../secret", size_bytes=0, sha256="0" * 64)


def test_manifest_rejects_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\nfixture\n", encoding="utf-8")
    manifest = manifest_for(source)
    source.unlink()

    with pytest.raises(ManifestVerificationError, match="missing"):
        verify_manifest(manifest, tmp_path)


def test_verify_manifest_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\nfixture\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_for(source).model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["safecart-ai-verify-manifest", str(manifest_path), str(tmp_path)],
    )

    verify_manifest_main()

    assert '"verified_file_count": 1' in capsys.readouterr().out


def test_build_catalog_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    source = tmp_path / fixture.name
    source.write_bytes(fixture.read_bytes())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_for(source).model_dump_json(), encoding="utf-8")
    output = tmp_path / "catalog.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safecart-ai-build-catalog",
            str(tmp_path),
            str(output),
            "--manifest",
            str(manifest_path),
        ],
    )

    build_catalog_main()

    assert output.is_file()
    assert '"canonical_records": 3' in capsys.readouterr().out
