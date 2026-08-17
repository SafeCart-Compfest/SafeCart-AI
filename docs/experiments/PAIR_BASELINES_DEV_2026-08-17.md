# Pair baselines dev audit — 2026-08-17

## Result

The current generated dev set is saturated by simple baselines and cannot demonstrate
the value of fine-tuning.

| Baseline | Macro-F1 (95% bootstrap CI) | Mismatch precision | Mismatch recall | Errors / 82,167 |
| --- | ---: | ---: | ---: | ---: |
| Exact NIE | 99.9644% (99.9493–99.9768) | 100% | 99.9525% | 26 |
| Deterministic fields | 99.9959% (99.9904–100%) | 99.9945% | 100% | 3 |

Input pair SHA-256:
`a2016a60458c3bed4edcc8ad5902ead2dd2f615ebcc2f0166956829ee4ba42f1`.
The run used the `dev` split only. Confidence intervals use 2,000 seeded multinomial
resamples of the observed confusion categories, equivalent to row bootstrapping for
confusion-derived metrics.

## Error analysis

- Exact NIE misses all 26 `AMBIGUOUS_NIE` mismatches because the identifier is identical
  by construction.
- Deterministic matching catches every generated negative.
- Its three false mismatches are positive multi-pack records where punctuation and
  decimal structure differ between normalized listing text and the official package,
  for example `3 75 3x1 25 g` versus `3.75(3x1.25) g`.

## Decision

The Kaggle smoke run remains useful to validate CUDA, dependencies, Trainer behavior,
and artifact manifests. The 180,000-row training run is **not** authorized by this dev
result: a fine-tuned model has virtually no measurable headroom and a high score would
be misleading.

Before the bounded full run, add manually reviewed real marketplace cases, OCR-degraded
text, same-NIE variant/package contradictions, missing-NIE cases, and edit-similar hard
negatives. Model selection may use the expanded train/dev data; calibration and frozen
test data remain untouched.
