from rapidfuzz.fuzz import ratio, token_sort_ratio

from safecart_ai.domain.models import (
    AssessmentStatus,
    CandidateAssessment,
    FieldEvidence,
    IdentityAssessment,
    OfficialProduct,
    ProductIdentity,
    ReasonCode,
)
from safecart_ai.domain.normalization import (
    extract_packages,
    normalize_nie,
    normalize_package,
    normalize_text,
)


def _similarity(left: str | None, right: str | None) -> float | None:
    normalized_left = normalize_text(left)
    normalized_right = normalize_text(right)
    if normalized_left is None or normalized_right is None:
        return None
    return token_sort_ratio(normalized_left, normalized_right) / 100


def _field_evidence(
    field: str,
    listing_value: str | None,
    official_value: str | None,
    threshold: float,
    mismatch_reason: ReasonCode,
) -> FieldEvidence:
    if field == "package":
        listing_normalized = normalize_package(listing_value)
        official_packages = extract_packages(official_value)
        similarity = (
            None
            if listing_normalized is None or not official_packages
            else float(listing_normalized in official_packages)
        )
    elif field == "brand":
        listing_normalized = normalize_text(listing_value)
        official_normalized = normalize_text(official_value)
        similarity = (
            None
            if listing_normalized is None or official_normalized is None
            else ratio(
                listing_normalized.replace(" ", ""),
                official_normalized.replace(" ", ""),
            )
            / 100
        )
    else:
        similarity = _similarity(listing_value, official_value)
    reason = ReasonCode.MATCH if similarity is None or similarity >= threshold else mismatch_reason
    return FieldEvidence(
        field=field,
        listing_value=listing_value,
        official_value=official_value,
        similarity=similarity,
        reason=reason,
    )


def _assess_candidate(listing: ProductIdentity, official: OfficialProduct) -> CandidateAssessment:
    evidence = [
        _field_evidence("brand", listing.brand, official.brand, 0.88, ReasonCode.BRAND_MISMATCH),
        _field_evidence(
            "product_name",
            listing.product_name,
            official.product_name,
            0.84,
            ReasonCode.PRODUCT_NAME_MISMATCH,
        ),
        _field_evidence(
            "package", listing.package, official.package, 1.0, ReasonCode.PACKAGE_MISMATCH
        ),
    ]
    available = [item.similarity for item in evidence if item.similarity is not None]
    score = sum(available) / len(available) if available else 0.0
    return CandidateAssessment(official=official, score=score, evidence=evidence)


def assess_identity(
    listing: ProductIdentity,
    official_candidates: list[OfficialProduct],
) -> IdentityAssessment:
    """Rule-based baseline; it is intentionally not the competition model."""

    if not normalize_nie(listing.nie) and not (listing.brand and listing.product_name):
        return IdentityAssessment(
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            reason_codes=[ReasonCode.MISSING_LISTING_IDENTITY],
        )

    if not official_candidates:
        return IdentityAssessment(
            status=AssessmentStatus.HIGH_PRIORITY_REVIEW,
            reason_codes=[ReasonCode.NIE_NOT_FOUND],
        )

    if not (listing.brand or listing.product_name or listing.package):
        return IdentityAssessment(
            status=AssessmentStatus.INSUFFICIENT_EVIDENCE,
            reason_codes=[ReasonCode.MISSING_LISTING_IDENTITY],
        )

    assessed = sorted(
        (_assess_candidate(listing, product) for product in official_candidates),
        key=lambda item: item.score,
        reverse=True,
    )
    selected = assessed[0]
    mismatch_reasons = [
        item.reason for item in selected.evidence if item.reason is not ReasonCode.MATCH
    ]
    reasons: list[ReasonCode] = []
    if len(official_candidates) > 1:
        reasons.append(ReasonCode.OFFICIAL_RECORD_AMBIGUOUS)
    reasons.extend(mismatch_reasons)

    if mismatch_reasons:
        status = AssessmentStatus.HIGH_PRIORITY_REVIEW
    elif len(official_candidates) > 1:
        status = AssessmentStatus.REVIEW
    else:
        status = AssessmentStatus.PASS_WITH_CURRENT_EVIDENCE

    return IdentityAssessment(
        status=status,
        reason_codes=reasons or [ReasonCode.MATCH],
        selected_candidate=selected,
        candidates=assessed,
    )
