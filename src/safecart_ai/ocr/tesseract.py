import csv
import io
import subprocess

from safecart_ai.ocr.models import OcrResult, OcrWord


class OcrError(RuntimeError):
    """Raised when the OCR process cannot read an image."""


class OcrUnavailableError(OcrError):
    """Raised when the configured OCR engine is unavailable."""


class TesseractOcr:
    def __init__(
        self,
        *,
        command: str = "tesseract",
        language: str = "eng+ind",
        timeout_seconds: int = 15,
    ) -> None:
        self._command = command
        self._language = language
        self._timeout_seconds = timeout_seconds

    def extract(self, image: bytes) -> OcrResult:
        try:
            process = subprocess.run(
                [
                    self._command,
                    "stdin",
                    "stdout",
                    "-l",
                    self._language,
                    "--psm",
                    "6",
                    "tsv",
                ],
                input=image,
                capture_output=True,
                check=False,
                timeout=self._timeout_seconds,
            )
        except FileNotFoundError as error:
            raise OcrUnavailableError("Tesseract is not installed") from error
        except subprocess.TimeoutExpired as error:
            raise OcrError("OCR timed out") from error

        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise OcrError(detail or "Tesseract could not read the image")

        return parse_tsv(process.stdout.decode("utf-8", errors="replace"))


def parse_tsv(value: str) -> OcrResult:
    words: list[OcrWord] = []
    for row in csv.DictReader(io.StringIO(value), delimiter="\t"):
        text = row.get("text", "").strip()
        if not text:
            continue
        try:
            confidence = float(row["conf"])
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            line_id = (
                int(row["page_num"]),
                int(row["block_num"]),
                int(row["par_num"]),
                int(row["line_num"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise OcrError("Tesseract returned invalid TSV output") from error
        if confidence < 0:
            continue
        words.append(
            OcrWord(
                text=text,
                confidence=min(confidence / 100, 1.0),
                box=(left, top, left + width, top + height),
                line_id=line_id,
            )
        )

    text = _build_text(words)
    mean_confidence = sum(word.confidence for word in words) / len(words) if words else 0.0
    return OcrResult(text=text, mean_confidence=mean_confidence, words=words)


def _build_text(words: list[OcrWord]) -> str:
    lines: list[str] = []
    current_line: tuple[int, int, int, int] | None = None
    current_words: list[str] = []
    for word in words:
        if current_line is not None and word.line_id != current_line:
            lines.append(" ".join(current_words))
            current_words = []
        current_line = word.line_id
        current_words.append(word.text)
    if current_words:
        lines.append(" ".join(current_words))
    return "\n".join(lines)
