# SafeCart AI

AI service for matching product information from marketplace screenshots against
versioned BPOM records. It compares listing identity only; it does not determine
product authenticity, chemical safety, or legal status.

## Features

- Builds a normalized BPOM cosmetics catalog.
- Reads PNG, JPEG, and WebP screenshots up to 10 MB with Tesseract OCR.
- Extracts NIE, brand, product name, package, confidence, and text boxes.
- Retrieves exact-NIE and similar BPOM candidates while preserving ambiguous records.
- Compares listing fields with official records and returns deterministic reason codes.
- Provides training and evaluation commands for the multilingual pair matcher.

## Pipeline

```text
screenshot -> OCR -> field extraction -> BPOM retrieval -> matching -> assessment
```

The service exposes:

- `GET /health`
- `POST /_internal/assessments` for structured product fields
- `POST /_internal/image-assessments` for screenshots

`SafeCart-API` is the public HTTP entry point. The PWA does not call this service
directly.

## Status

- [x] Catalog normalization and validation
- [x] OCR and product-field extraction
- [x] Candidate retrieval and ambiguity handling
- [x] Deterministic matching, status, and reason codes
- [x] Private structured-text and image endpoints
- [ ] Review representative marketplace screenshots
- [ ] Integrate the trained pair matcher
- [ ] Calibrate thresholds and run final evaluation
- [ ] Complete the API integration test

## Local development

Requirements: Python 3.11-3.13, [uv](https://docs.astral.sh/uv/), and Tesseract.

```bash
uv sync --extra dev
uv run uvicorn safecart_ai.main:app --reload --port 8001
```

Open `http://localhost:8001/docs` or run `curl http://localhost:8001/health`.

## Data and training

Raw datasets, screenshots, generated datasets, and model weights are not committed.
Source metadata and checksums are stored in `data/manifests/` and
`artifacts/manifests/`.

```bash
uv run safecart-ai-verify-manifest data/manifests/bpom-cosmetics-2026-08-17.json "../dataset/Data BPOM"
uv run safecart-ai-build-catalog "../dataset/Data BPOM" \
  --manifest data/manifests/bpom-cosmetics-2026-08-17.json \
  --output data/processed/bpom-cosmetics.csv
uv run safecart-ai-generate-pairs data/processed/bpom-cosmetics.csv \
  data/processed/product-pairs.csv --seed 42
uv run safecart-ai-train-matcher data/processed/product-pairs.csv \
  --config training/configs/distilmbert-v1.toml \
  --output outputs/distilmbert-v1
```

See `docs/DATA.md`, `docs/EVALUATION.md`, and `training/README.md` for the remaining
technical details.

## Validation

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=safecart_ai --cov-report=term-missing
docker build -t safecart-ai .
```

See `CONTRIBUTING.md` for the pull-request workflow.
