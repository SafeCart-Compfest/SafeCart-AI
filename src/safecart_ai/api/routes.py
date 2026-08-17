from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from safecart_ai.config import Settings, get_settings
from safecart_ai.domain.matching import assess_identity
from safecart_ai.domain.models import IdentityAssessment, ProductIdentity
from safecart_ai.infrastructure.products import CsvOfficialProductRepository
from safecart_ai.ocr.entities import extract_identity
from safecart_ai.ocr.models import ImageAssessment, OcrEngine
from safecart_ai.ocr.tesseract import OcrError, OcrUnavailableError, TesseractOcr

router = APIRouter()
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


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


@lru_cache
def _load_ocr_engine(command: str, language: str, timeout_seconds: int) -> TesseractOcr:
    return TesseractOcr(
        command=command,
        language=language,
        timeout_seconds=timeout_seconds,
    )


def get_ocr_engine(settings: Annotated[Settings, Depends(get_settings)]) -> OcrEngine:
    return _load_ocr_engine(
        settings.ocr_command,
        settings.ocr_language,
        settings.ocr_timeout_seconds,
    )


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


@router.post("/_internal/image-assessments", response_model=ImageAssessment)
def create_image_assessment(
    image: Annotated[UploadFile, File(description="Marketplace listing screenshot")],
    repository: Annotated[CsvOfficialProductRepository, Depends(get_product_repository)],
    ocr: Annotated[OcrEngine, Depends(get_ocr_engine)],
) -> ImageAssessment:
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Image must be PNG, JPEG, or WebP",
        )

    content = image.file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Image is empty",
        )
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Image exceeds 10 MB",
        )

    try:
        ocr_result = ocr.extract(content)
    except OcrUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR service is unavailable",
        ) from error
    except OcrError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Image could not be read",
        ) from error

    extracted = extract_identity(ocr_result)
    identity = extracted.to_product_identity()
    candidates = repository.find_candidates(
        nie=identity.nie,
        brand=identity.brand,
        product_name=identity.product_name,
    )
    return ImageAssessment(
        extracted_identity=extracted,
        ocr_confidence=ocr_result.mean_confidence,
        assessment=assess_identity(identity, candidates),
    )
