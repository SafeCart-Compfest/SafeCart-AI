# Repository strategy

SafeCart uses service-oriented repositories with a consistent `SafeCart-{Purpose}`
naming convention.

## Ownership

- `SafeCart-PWA`: mobile-first frontend and PWA assets.
- `SafeCart-API`: public HTTP contract, request validation, orchestration, and policy
  response mapping.
- `SafeCart-AI`: data preparation, OCR, retrieval, matching, calibration, training,
  evaluation, model export, and private inference runtime.
- `SafeCart-ScrapingData`: offline acquisition experiments and snapshot tooling; never a
  runtime dependency.
- `SafeCart-Deployment`: primary submission repository, pinned service versions, root
  setup guide, and Docker Compose contract.

Only PWA, API, and AI are runtime services. A new repository requires an independent
ownership boundary and release lifecycle; it must not be created for every Python
module or small supporting concern.

## Public boundary

Competition repositories are public. Never commit credentials, raw acquisition data,
private screenshots, generated training pairs, model checkpoints, or unlicensed
content. Publish source code, safe fixtures, manifests, checksums, aggregate metrics,
and reviewed documentation only.

## Submission integration

`SafeCart-Deployment` is the single source-code link submitted to the competition. It
pins compatible releases of each runtime service and links their repositories. The
evaluator path must not depend on `SafeCart-ScrapingData`; prepared catalog and model
artifacts are immutable and checksummed.
