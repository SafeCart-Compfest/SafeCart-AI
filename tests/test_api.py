from fastapi.testclient import TestClient

from safecart_ai.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_assessment_endpoint() -> None:
    response = client.post(
        "/_internal/baseline/assessments",
        json={
            "nie": "NA18250116783",
            "brand": "LumiGlow",
            "product_name": "Brightening Day Cream",
            "package": "15 g",
            "official_candidates": [
                {
                    "nie": "NA18250116783",
                    "brand": "LumiGlow",
                    "product_name": "Intensive Night Cream",
                    "package": "30 g",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "HIGH_PRIORITY_REVIEW"
