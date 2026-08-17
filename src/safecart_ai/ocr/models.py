from typing import Protocol

from pydantic import BaseModel, Field

from safecart_ai.domain.models import IdentityAssessment, ProductIdentity


class OcrWord(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    box: tuple[int, int, int, int]
    line_id: tuple[int, int, int, int] = Field(exclude=True)


class OcrResult(BaseModel):
    text: str
    mean_confidence: float = Field(ge=0, le=1)
    words: list[OcrWord]


class ExtractedField(BaseModel):
    value: str
    confidence: float = Field(ge=0, le=1)
    box: tuple[int, int, int, int]


class ExtractedIdentity(BaseModel):
    nie: ExtractedField | None = None
    brand: ExtractedField | None = None
    product_name: ExtractedField | None = None
    package: ExtractedField | None = None

    def to_product_identity(self) -> ProductIdentity:
        return ProductIdentity(
            nie=self.nie.value if self.nie else None,
            brand=self.brand.value if self.brand else None,
            product_name=self.product_name.value if self.product_name else None,
            package=self.package.value if self.package else None,
        )


class ImageAssessment(BaseModel):
    extracted_identity: ExtractedIdentity
    ocr_confidence: float = Field(ge=0, le=1)
    assessment: IdentityAssessment


class OcrEngine(Protocol):
    def extract(self, image: bytes) -> OcrResult: ...
