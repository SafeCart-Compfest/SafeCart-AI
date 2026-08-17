from enum import StrEnum

from pydantic import BaseModel, Field


class AssessmentStatus(StrEnum):
    PASS_WITH_CURRENT_EVIDENCE = "PASS_WITH_CURRENT_EVIDENCE"
    REVIEW = "REVIEW"
    HIGH_PRIORITY_REVIEW = "HIGH_PRIORITY_REVIEW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReasonCode(StrEnum):
    MATCH = "MATCH"
    MISSING_LISTING_IDENTITY = "MISSING_LISTING_IDENTITY"
    NIE_NOT_FOUND = "NIE_NOT_FOUND"
    OFFICIAL_RECORD_AMBIGUOUS = "OFFICIAL_RECORD_AMBIGUOUS"
    BRAND_MISMATCH = "BRAND_MISMATCH"
    PRODUCT_NAME_MISMATCH = "PRODUCT_NAME_MISMATCH"
    PACKAGE_MISMATCH = "PACKAGE_MISMATCH"


class ProductIdentity(BaseModel):
    nie: str | None = None
    brand: str | None = None
    product_name: str | None = None
    package: str | None = None


class OfficialProduct(ProductIdentity):
    registrant: str | None = None
    registration_status: str | None = None
    valid_until: str | None = None
    source: str = "BPOM"
    source_snapshot_at: str | None = None


class FieldEvidence(BaseModel):
    field: str
    listing_value: str | None
    official_value: str | None
    similarity: float | None = Field(default=None, ge=0, le=1)
    reason: ReasonCode


class CandidateAssessment(BaseModel):
    official: OfficialProduct
    score: float = Field(ge=0, le=1)
    evidence: list[FieldEvidence]


class IdentityAssessment(BaseModel):
    status: AssessmentStatus
    reason_codes: list[ReasonCode]
    selected_candidate: CandidateAssessment | None = None
    candidates: list[CandidateAssessment] = Field(default_factory=list)
    disclaimer: str = (
        "SafeCart assesses listing-to-record consistency and does not determine product "
        "authenticity, chemical safety, or legal liability. Human review is required."
    )
