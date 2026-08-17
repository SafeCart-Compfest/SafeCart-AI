import re
from collections.abc import Iterable

from safecart_ai.domain.normalization import normalize_nie, normalize_package, normalize_text
from safecart_ai.ocr.models import ExtractedField, ExtractedIdentity, OcrResult, OcrWord

_NIE = re.compile(r"N[A-Z]\d{11}")
_PACKAGE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:kg|g|gr|gram|ml|l)\b", re.IGNORECASE)
_BRAND_LABELS = {"brand", "merek"}
_PRODUCT_LABELS = {"product", "produk"}


def extract_identity(result: OcrResult) -> ExtractedIdentity:
    return ExtractedIdentity(
        nie=_extract_nie(result.words),
        brand=_extract_labeled_value(result.words, _BRAND_LABELS),
        product_name=_extract_labeled_value(result.words, _PRODUCT_LABELS),
        package=_extract_package(result.words),
    )


def _extract_nie(words: list[OcrWord]) -> ExtractedField | None:
    for window in _word_windows(words, maximum=4):
        normalized = normalize_nie("".join(word.text for word in window))
        if normalized and _NIE.fullmatch(normalized):
            return _field(normalized, window)
    return None


def _extract_package(words: list[OcrWord]) -> ExtractedField | None:
    for window in _word_windows(words, maximum=2):
        value = " ".join(word.text for word in window)
        match = _PACKAGE.search(value)
        if match:
            normalized = normalize_package(match.group())
            if normalized:
                return _field(normalized, window)
    return None


def _extract_labeled_value(
    words: list[OcrWord],
    labels: set[str],
) -> ExtractedField | None:
    for index, word in enumerate(words):
        label = normalize_text(word.text.rstrip(":"))
        if label not in labels:
            continue
        line_words = [
            candidate for candidate in words[index + 1 :] if candidate.line_id == word.line_id
        ]
        if line_words:
            return _field(" ".join(item.text for item in line_words), line_words)
    return None


def _word_windows(words: list[OcrWord], maximum: int) -> Iterable[list[OcrWord]]:
    for size in range(1, maximum + 1):
        for start in range(len(words)):
            window = words[start : start + size]
            if len(window) != size or any(word.line_id != window[0].line_id for word in window):
                continue
            yield window


def _field(value: str, words: list[OcrWord]) -> ExtractedField:
    return ExtractedField(
        value=value,
        confidence=sum(word.confidence for word in words) / len(words),
        box=(
            min(word.box[0] for word in words),
            min(word.box[1] for word in words),
            max(word.box[2] for word in words),
            max(word.box[3] for word in words),
        ),
    )
