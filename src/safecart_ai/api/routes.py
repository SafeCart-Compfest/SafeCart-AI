from fastapi import APIRouter

from safecart_ai.domain.matching import assess_identity
from safecart_ai.domain.models import IdentityAssessment, OfficialProduct, ProductIdentity

router = APIRouter()


class AssessmentRequest(ProductIdentity):
    official_candidates: list[OfficialProduct]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/_internal/baseline/assessments", response_model=IdentityAssessment)
def create_assessment(request: AssessmentRequest) -> IdentityAssessment:
    return assess_identity(request, request.official_candidates)
