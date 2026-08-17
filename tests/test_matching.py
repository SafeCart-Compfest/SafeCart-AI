from safecart_ai.domain.matching import assess_identity
from safecart_ai.domain.models import (
    AssessmentStatus,
    OfficialProduct,
    ProductIdentity,
    ReasonCode,
)


def official(**overrides: str) -> OfficialProduct:
    values = {
        "nie": "NA18250116783",
        "brand": "LumiGlow",
        "product_name": "Intensive Night Cream",
        "package": "Pot, 30 g",
    }
    values.update(overrides)
    return OfficialProduct(**values)


def test_matching_identity_passes_with_current_evidence() -> None:
    result = assess_identity(
        ProductIdentity(
            nie="NA18250116783",
            brand="Lumi Glow",
            product_name="Intensive Night Cream",
            package="30 gram",
        ),
        [official()],
    )

    assert result.status is AssessmentStatus.PASS_WITH_CURRENT_EVIDENCE
    assert result.reason_codes == [ReasonCode.MATCH]


def test_listing_matches_any_registered_package_size() -> None:
    result = assess_identity(
        ProductIdentity(
            nie="NA18250116783",
            brand="LumiGlow",
            product_name="Intensive Night Cream",
            package="30 g",
        ),
        [official(package="Tube, Dus 15 g, Tube, Dus 30 g")],
    )

    assert result.status is AssessmentStatus.PASS_WITH_CURRENT_EVIDENCE


def test_subtle_identity_mismatch_is_high_priority() -> None:
    result = assess_identity(
        ProductIdentity(
            nie="NA18250116783",
            brand="LumiGlow",
            product_name="Brightening Day Cream",
            package="15 g",
        ),
        [official()],
    )

    assert result.status is AssessmentStatus.HIGH_PRIORITY_REVIEW
    assert ReasonCode.PRODUCT_NAME_MISMATCH in result.reason_codes
    assert ReasonCode.PACKAGE_MISMATCH in result.reason_codes


def test_multiple_official_records_are_never_auto_passed() -> None:
    result = assess_identity(
        ProductIdentity(
            nie="NA18250116783",
            brand="LumiGlow",
            product_name="Intensive Night Cream",
            package="30 g",
        ),
        [official(), official(brand="Other Brand", product_name="Other Product")],
    )

    assert result.status is AssessmentStatus.REVIEW
    assert ReasonCode.OFFICIAL_RECORD_AMBIGUOUS in result.reason_codes
    assert len(result.candidates) == 2
    assert ReasonCode.PRODUCT_NAME_MISMATCH in {
        evidence.reason for evidence in result.candidates[1].evidence
    }


def test_no_official_record_is_high_priority_review() -> None:
    result = assess_identity(ProductIdentity(nie="NA00000000000"), [])

    assert result.status is AssessmentStatus.HIGH_PRIORITY_REVIEW
    assert result.reason_codes == [ReasonCode.NIE_NOT_FOUND]


def test_nie_only_cannot_establish_identity_consistency() -> None:
    result = assess_identity(ProductIdentity(nie="NA18250116783"), [official()])

    assert result.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.reason_codes == [ReasonCode.MISSING_LISTING_IDENTITY]


def test_missing_identity_abstains() -> None:
    result = assess_identity(ProductIdentity(), [])

    assert result.status is AssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.reason_codes == [ReasonCode.MISSING_LISTING_IDENTITY]


def test_missing_optional_field_does_not_create_false_mismatch() -> None:
    result = assess_identity(
        ProductIdentity(nie="NA18250116783", product_name="Intensive Night Cream"),
        [official()],
    )

    assert result.status is AssessmentStatus.PASS_WITH_CURRENT_EVIDENCE
