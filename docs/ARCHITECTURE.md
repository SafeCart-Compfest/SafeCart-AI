# AI service architecture

## Decision

SafeCart is a listing identity consistency and compliance-triage system. The core
research question is whether a fine-tuned pair matcher detects subtle listing-to-record
mismatches that exact NIE lookup and lexical similarity miss.

## End-to-end ownership

```text
SafeCart-PWA -> SafeCart-API -> SafeCart-AI
                                  |
                                  +-> versioned catalog and model artifacts
```

`SafeCart-API` owns the public multipart request and stable assessment response.
`SafeCart-AI` owns the private inference operation and AI artifacts. The PWA never
calls AI directly, and the scraping repository is not on the runtime path.

## AI components

1. OCR extracts text, bounding boxes, and confidence from one listing screenshot.
2. Entity extraction normalizes NIE, brand, product name, variant, and package.
3. Retrieval returns every exact NIE match plus top lexical candidates.
4. A fine-tuned cross-encoder predicts `MATCH` or `MISMATCH` for candidate pairs.
5. Calibration and deterministic field comparisons produce a confidence and reason codes.
6. The private FastAPI adapter returns the result to the public API orchestrator.

## Dependency direction

```text
HTTP/OCR/artifact adapters -> application orchestration -> domain models and policies
```

The domain layer must not import FastAPI, OCR libraries, storage clients, or model
frameworks. Model and baseline implementations share an application-level matcher
interface when the trained model is added.

## Critical invariant

An NIE is not assumed to identify exactly one row. Snapshots and the public portal can
contain multiple distinct records for the same NIE. Retrieval preserves all records,
attaches source and snapshot metadata, and routes ambiguity to human review.

## Preliminary scope

- One synchronous private inference operation consumed only by `SafeCart-API`.
- No authentication, background jobs, crawler, distributed database, or automatic
  takedown.
- `/_internal/baseline/assessments` accepts structured candidates for baseline testing
  only. It is not a public or production contract and must not be called by the PWA.
