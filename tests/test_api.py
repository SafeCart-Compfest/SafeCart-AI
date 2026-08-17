from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from safecart_ai.api.routes import get_product_repository
from safecart_ai.config import Settings
from safecart_ai.infrastructure.products import CsvOfficialProductRepository
from safecart_ai.main import app

client = TestClient(app)


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
