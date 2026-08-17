# SafeCart AI

Internal AI service and reproducible experimentation code for SafeCart, an
evidence-grounded marketplace listing identity assessment system for COMPFEST 18 AIC.

SafeCart AI compares identity information extracted from a marketplace screenshot with
versioned BPOM records. It does **not** determine whether a product is authentic, safe,
or legal.

## Service boundary

This repository owns:

- BPOM source manifests, verification, normalization, and canonical catalog builders;
- leakage-safe pair generation and retrieval evaluation;
- OCR, entity extraction, matching, calibration, and model export code;
- the private HTTP inference boundary used by `SafeCart-API`;
- AI experiments, evaluation reports, and model documentation.

It does not own the public assessment contract, frontend, scraping jobs, deployment
composition, raw datasets, screenshots, or model weights. The PWA must call
`SafeCart-API`, never this service directly.

## Current pipeline

```text
verified BPOM snapshots
  -> canonical product catalog
  -> product-family split
  -> leakage-safe positive and hard-negative pairs
  -> hybrid retrieval baseline
  -> matcher, calibration, and deterministic evidence policy
```

The current private endpoint is a deterministic baseline used for integration and
golden tests. OCR and the fine-tuned cross-encoder will replace its client-supplied
fixture input only after their acceptance gates pass.

## Local development

Requirements: Python 3.11-3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run uvicorn safecart_ai.main:app --reload --port 8001
```

Open `http://localhost:8001/docs` or check process health:

```bash
curl http://localhost:8001/health
```

## Data workflow

Raw sources and generated datasets are intentionally ignored by Git. Review
`data/README.md` and the source manifests before running a builder.

```bash
uv run safecart-ai-verify-manifest data/manifests/bpom-source.json "../dataset/Data BPOM"
uv run safecart-ai-audit-bpom "../dataset/Data BPOM" --output outputs/bpom-audit.json
uv run safecart-ai-build-catalog "../dataset/Data BPOM" \
  --manifest data/manifests/bpom-source.json \
  --output data/processed/bpom-cosmetics.csv
uv run safecart-ai-generate-pairs data/processed/bpom-cosmetics.csv \
  data/processed/product-pairs.csv --seed 42
uv run safecart-ai-evaluate-retrieval data/processed/bpom-cosmetics.csv \
  data/processed/product-pairs.csv --split dev \
  --output outputs/retrieval-dev.json
uv run safecart-ai-train-matcher data/processed/product-pairs.csv \
  --config training/configs/distilmbert-v1.toml \
  --output outputs/distilmbert-v1
```

Do not use the frozen test split to choose features, models, or thresholds.

## Validation

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest --cov=safecart_ai --cov-report=term-missing
docker build -t safecart-ai .
```

CI runs formatting, linting, strict type checking, branch coverage (minimum 85%),
dependency auditing, and Docker build checks.

## Docker

```bash
docker build -t safecart-ai .
docker run --rm -p 8001:8001 safecart-ai
```

Cross-service orchestration belongs in `SafeCart-Deployment`; this repository provides
only its independently deployable image.

## Repository map

- `src/safecart_ai/`: tested service and experiment implementation.
- `tests/`: unit, contract, leakage, and golden tests.
- `training/`: thin Kaggle launchers and versioned training configurations.
- `data/manifests/`: source provenance and integrity metadata.
- `data/samples/`: small synthetic or explicitly redistributable fixtures.
- `artifacts/manifests/`: model artifact metadata and checksums, never weights.
- `docs/`: architecture, data, annotation, evaluation, and limitation documents.

See `CONTRIBUTING.md` for the protected-branch workflow.
