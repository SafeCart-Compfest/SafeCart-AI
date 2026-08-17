# SafeCart AI service agent rules

- Keep this repository limited to data preparation, experiments, OCR, retrieval,
  matching, calibration, model export, and private inference.
- Do not add the public assessment API, frontend, scraping runtime, or deployment
  composition here.
- Work AI-first. Do not add PWA integration until the experiment acceptance gates pass.
- Explain affected files and the current flow before broad edits.
- Keep domain logic independent from FastAPI, OCR, storage, and model frameworks.
- Treat all external records as versioned evidence, not unquestionable ground truth.
- Preserve multiple official candidates for the same NIE and abstain on ambiguity.
- Never output `FAKE`, `ILLEGAL`, or a legal/safety conclusion from listing evidence.
- Do not commit datasets, weights, secrets, generated artifacts, or institution branding.
- Keep Kaggle notebooks thin; tested Python modules are the source of truth.
- Use English for code, comments, identifiers, and commit messages.
- Run format, lint, type-check, tests, and Docker build when practical.
- Never commit, push, pull, merge, rebase, or create branches unless the user explicitly
  requests that Git action.
