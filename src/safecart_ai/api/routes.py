from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from safecart_ai.config import Settings, get_settings
from safecart_ai.domain.matching import assess_identity
from safecart_ai.domain.models import IdentityAssessment, ProductIdentity
from safecart_ai.infrastructure.products import CsvOfficialProductRepository

router = APIRouter()


@lru_cache
def _load_repository(path: str) -> CsvOfficialProductRepository:
    return CsvOfficialProductRepository(Path(path))


def get_product_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CsvOfficialProductRepository:
    try:
        return _load_repository(str(settings.data_path))
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BPOM catalog is unavailable",
        ) from error


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/_internal/assessments", response_model=IdentityAssessment)
def create_assessment(
    request: ProductIdentity,
    repository: Annotated[CsvOfficialProductRepository, Depends(get_product_repository)],
) -> IdentityAssessment:
    candidates = repository.find_candidates(
        nie=request.nie,
        brand=request.brand,
        product_name=request.product_name,
    )
    return assess_identity(request, candidates)
