import subprocess
from unittest.mock import MagicMock, patch

import pytest

from safecart_ai.ocr.entities import extract_identity
from safecart_ai.ocr.models import OcrResult, OcrWord
from safecart_ai.ocr.tesseract import OcrError, OcrUnavailableError, TesseractOcr, parse_tsv

TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext\n"
)
TSV = (
    TSV_HEADER
    + """\
5\t1\t1\t1\t1\t1\t10\t20\t40\t12\t96.0\tBPOM
5\t1\t1\t1\t1\t2\t55\t20\t100\t12\t94.0\tNA18250116783
5\t1\t1\t1\t2\t1\t10\t40\t45\t12\t90.0\tMerek:
5\t1\t1\t1\t2\t2\t60\t40\t70\t12\t92.0\tLumiGlow
5\t1\t1\t1\t3\t1\t10\t60\t50\t12\t91.0\tProduk:
5\t1\t1\t1\t3\t2\t65\t60\t70\t12\t93.0\tNight
5\t1\t1\t1\t3\t3\t140\t60\t60\t12\t95.0\tCream
5\t1\t1\t1\t4\t1\t10\t80\t20\t12\t89.0\t30
5\t1\t1\t1\t4\t2\t35\t80\t10\t12\t91.0\tg
"""
)


def test_parse_tesseract_tsv_and_extract_identity() -> None:
    result = parse_tsv(TSV)
    identity = extract_identity(result)

    assert result.text.splitlines()[0] == "BPOM NA18250116783"
    assert result.mean_confidence == pytest.approx(0.923333, rel=1e-5)
    assert identity.nie is not None
    assert identity.nie.value == "NA18250116783"
    assert identity.brand is not None
    assert identity.brand.value == "LumiGlow"
    assert identity.product_name is not None
    assert identity.product_name.value == "Night Cream"
    assert identity.package is not None
    assert identity.package.value == "30 g"


@patch("safecart_ai.ocr.tesseract.subprocess.run")
def test_tesseract_reads_image_from_stdin(run: MagicMock) -> None:
    process = subprocess.CompletedProcess([], 0, stdout=TSV.encode(), stderr=b"")
    run.return_value = process

    result = TesseractOcr().extract(b"image")

    assert result.words[1].text == "NA18250116783"
    run.assert_called_once()
    assert run.call_args.kwargs["input"] == b"image"


@patch("safecart_ai.ocr.tesseract.subprocess.run", side_effect=FileNotFoundError)
def test_missing_tesseract_is_reported(run: MagicMock) -> None:
    del run
    with pytest.raises(OcrUnavailableError, match="not installed"):
        TesseractOcr().extract(b"image")


@patch("safecart_ai.ocr.tesseract.subprocess.run")
def test_tesseract_failure_is_reported(run: MagicMock) -> None:
    run.return_value = subprocess.CompletedProcess(
        [],
        1,
        stdout=b"",
        stderr=b"unsupported image",
    )

    with pytest.raises(OcrError, match="unsupported image"):
        TesseractOcr().extract(b"image")


@patch(
    "safecart_ai.ocr.tesseract.subprocess.run",
    side_effect=subprocess.TimeoutExpired("tesseract", 15),
)
def test_tesseract_timeout_is_reported(run: MagicMock) -> None:
    del run
    with pytest.raises(OcrError, match="timed out"):
        TesseractOcr().extract(b"image")


def test_invalid_tsv_is_rejected() -> None:
    with pytest.raises(OcrError, match="invalid TSV"):
        parse_tsv("conf\ttext\nnot-a-number\tvalue\n")


def test_empty_ocr_result_has_zero_confidence() -> None:
    result = parse_tsv("conf\ttext\n")

    assert result == OcrResult(text="", mean_confidence=0, words=[])


def test_nie_can_span_multiple_words() -> None:
    line_id = (1, 1, 1, 1)
    result = OcrResult(
        text="BPOM NA 18250116783",
        mean_confidence=0.9,
        words=[
            OcrWord(text="BPOM", confidence=0.9, box=(0, 0, 30, 10), line_id=line_id),
            OcrWord(text="NA", confidence=0.9, box=(35, 0, 50, 10), line_id=line_id),
            OcrWord(
                text="18250116783",
                confidence=0.9,
                box=(55, 0, 130, 10),
                line_id=line_id,
            ),
        ],
    )

    identity = extract_identity(result)

    assert identity.nie is not None
    assert identity.nie.value == "NA18250116783"
    assert identity.nie.box == (35, 0, 130, 10)
