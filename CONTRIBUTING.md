# Contributing

## Development principles

- Prefer small, testable changes that preserve the API contract.
- Keep domain logic independent from FastAPI, storage, OCR, and model frameworks.
- Do not commit secrets, personal data, raw datasets, generated artifacts, or model
  weights.
- Do not claim that a result proves a product is authentic, safe, or illegal.
- Do not show educational institution names or branding in competition artifacts.

## Branch workflow

`main` is the protected, reproducible branch. Do not push directly to it.

1. Update local `main` with `git pull --ff-only`.
2. Create a short-lived branch: `feat/...`, `fix/...`, `experiment/...`, `docs/...`, or
   `chore/...`.
3. Commit each coherent change with tests and a descriptive Conventional Commit.
4. Push the branch and open a pull request.
5. Complete the self-review checklist and require every configured quality check to pass.
6. Delete the merged branch. Never force-push a shared branch.

Examples:

```text
feat(matching): add hard-negative pair classifier
fix(data): prevent product-family leakage across splits
experiment(retrieval): compare BM25 and multilingual embeddings
docs(methodology): record OCR ablation results
```

## Pull request checklist

- Scope and motivation are explicit.
- Dataset source, snapshot date, license/permission, and preprocessing are documented.
- Metrics include per-class results and false-positive behavior, not only accuracy.
- Tests cover the changed behavior.
- README or experiment notes are updated when reproducibility changes.
- No institution branding, credentials, large data, or generated files are included.

## Recommended GitHub settings

Configure these in the organization after the team agrees:

- Protect `main`; require pull requests and resolved conversations. Approval count is zero
  for the preliminary sprint, so the author must perform the documented self-review.
- Require the test, lint, and type-check jobs before merge.
- Block force pushes and branch deletion on `main`.
- Enable secret scanning and dependency alerts.
- Keep an immutable tag for the preliminary submission commit.

Repository settings are not changed by this document.
