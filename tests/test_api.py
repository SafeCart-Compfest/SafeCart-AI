from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from safecart_ai.api.routes import get_ocr_engine, get_product_repository
from safecart_ai.config import Settings
from safecart_ai.infrastructure.products import CsvOfficialProductRepository
from safecart_ai.main import app
from safecart_ai.ocr.models import OcrEngine, OcrResult, OcrWord
from safecart_ai.ocr.tesseract import OcrError, OcrUnavailableError

client = TestClient(app)


class FakeOcr:
    def extract(self, image: bytes) -> OcrResult:
        assert image == b"image"
        line_id = (1, 1, 1, 1)
        return OcrResult(
            text="NA18250116783 15 g",
            mean_confidence=0.95,
            words=[
                OcrWord(
                    text="NA18250116783",
                    confidence=0.96,
                    box=(0, 0, 100, 20),
                    line_id=line_id,
                ),
                OcrWord(text="15", confidence=0.94, box=(0, 30, 20, 50), line_id=line_id),
                OcrWord(text="g", confidence=0.95, box=(25, 30, 35, 50), line_id=line_id),
            ],
        )


class FailingOcr:
    def extract(self, image: bytes) -> OcrResult:
        del image
        raise OcrError("invalid image")


class MissingOcr:
    def extract(self, image: bytes) -> OcrResult:
        del image
        raise OcrUnavailableError("missing OCR")


def use_test_services(ocr: OcrEngine) -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    app.dependency_overrides[get_product_repository] = lambda: CsvOfficialProductRepository(fixture)
    app.dependency_overrides[get_ocr_engine] = lambda: ocr


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_assessment_endpoint() -> None:
    fixture = Path(__file__).parent / "fixtures" / "bpom_sample.csv"
    app.dependency_overrides[get_product_repository] = lambda: CsvOfficialProductRepository(fixture)
    try:
        response = client.post(
            "/_internal/assessments",
            json={
                "nie": "NA18250116783",
                "brand": "LumiGlow",
                "product_name": "Brightening Day Cream",
                "package": "15 g",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "HIGH_PRIORITY_REVIEW"
    assert response.json()["selected_candidate"]["official"]["nie"] == "NA18250116783"


def test_missing_catalog_is_service_unavailable(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as error:
        get_product_repository(Settings(data_path=tmp_path / "missing.csv"))

    assert error.value.status_code == 503


def test_image_assessment_endpoint() -> None:
    use_test_services(FakeOcr())
    try:
        response = client.post(
            "/_internal/image-assessments",
            files={"image": ("listing.png", b"image", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["extracted_identity"]["nie"]["value"] == "NA18250116783"
    assert body["extracted_identity"]["package"]["value"] == "15 g"
    assert body["assessment"]["status"] == "HIGH_PRIORITY_REVIEW"
    assert body["assessment"]["reason_codes"] == ["PACKAGE_MISMATCH"]


def test_image_assessment_rejects_unsupported_media_type() -> None:
    use_test_services(FakeOcr())
    try:
        response = client.post(
            "/_internal/image-assessments",
            files={"image": ("listing.gif", b"image", "image/gif")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 415


def test_image_assessment_rejects_unreadable_image() -> None:
    use_test_services(FailingOcr())
    try:
        response = client.post(
            "/_internal/image-assessments",
            files={"image": ("listing.png", b"image", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_image_assessment_rejects_empty_image() -> None:
    use_test_services(FakeOcr())
    try:
        response = client.post(
            "/_internal/image-assessments",
            files={"image": ("listing.png", b"", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_image_assessment_rejects_large_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("safecart_ai.api.routes.MAX_IMAGE_BYTES", 4)
    use_test_services(FakeOcr())
    try:
        response = client.post(
            "/_internal/image-assessments",
            files={"image": ("listing.png", b"large", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413


def test_image_assessment_reports_missing_ocr() -> None:
    use_test_services(MissingOcr())
    try:
        response = client.post(
            "/_internal/image-assessments",
            files={"image": ("listing.png", b"image", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
