# Experiment plan

Submission cutoff: 2026-08-25 23:55 WIB.

## Hypothesis

A fine-tuned Indonesian product-pair matcher can improve subtle mismatch detection over
exact NIE lookup, fuzzy field matching, and an unfine-tuned multilingual encoder while
maintaining an acceptable false-positive rate on valid listings.

## Baselines

1. Exact NIE existence lookup.
2. Deterministic normalized field comparison (implemented here).
3. Fuzzy/BM25 candidate retrieval.
4. Pretrained multilingual encoder without fine-tuning.
5. Fine-tuned cross-encoder or compact sequence classifier.

## Metrics

- OCR: NIE and entity exact-match accuracy.
- Retrieval: Recall@1 and Recall@5.
- Pair classification: macro-F1 and per-class precision/recall.
- Safety: valid-listing false-positive rate and abstention coverage.
- Calibration: Brier score or ECE on a held-out calibration split.
- Operations: p50/p95 end-to-end latency on the evaluator machine.

## Acceptance gates before PWA integration

- Provenance and permitted use are documented for every training/evaluation source.
- Product-family split and duplicate checks show no leakage.
- Real, manually reviewed test data is included alongside synthetic hard negatives.
- Retrieval Recall@5 is at least 95% on the test set.
- Mismatch recall is at least 80% at at least 85% precision on real hard negatives.
- False-positive rate on valid listings is at most 10%.
- Fine-tuning materially outperforms the deterministic and unfine-tuned baselines.
- Low-evidence and official-record ambiguity cases abstain instead of auto-passing.

Thresholds are competition targets, not current measured results. If they are missed,
report the measured result and failure analysis rather than tuning on the test set.

## 17 August baseline audit

Exact-NIE and deterministic rules reach 99.96% and 100.00% macro-F1 (rounded) on the
generated dev pairs. This saturation means the synthetic split is useful for leakage and
pipeline testing but not for proving model lift. The bounded full fine-tuning run is held
until real hard negatives and OCR-degraded dev cases create measurable headroom. See
`docs/experiments/PAIR_BASELINES_DEV_2026-08-17.md`.

The Kaggle T4 training path also passed a pinned 96/48-row technical smoke test. Its
metrics are deliberately excluded from acceptance-gate evidence; see
`docs/experiments/MATCHER_SMOKE_KAGGLE_V4_2026-08-17.md`.

## Eight-day critical path

### 17 August

Freeze the problem statement, audit data, document provenance gaps, and implement the
deterministic baseline.

### 18 August

Create the canonical official-product table, reconcile duplicate/ambiguous records, and
write pair-generation rules. Manually label the first real hard-negative set.

### 19 August

Build leakage-safe train/dev/test splits and run exact, fuzzy, and pretrained baselines.

### 20 August

Fine-tune the smallest viable multilingual pair model. Log configuration, seed, dataset
hash, metrics, latency, and failures.

### 21 August

Run hard-negative ablations, calibration, error analysis, and select one frozen model.

### 22 August

Integrate OCR and entity extraction only after structured-text matching passes. Measure
end-to-end degradation.

### 23 August

Freeze the API contract, integrate the single-screen PWA, and validate Docker Compose on
a clean machine.

### 24 August

Record the uncut proof-of-work flow, produce the promotion video, finalize the proposal,
and perform a judge-style review against every rubric item.

### 25 August

Fix only submission-blocking issues, tag the tested commit after approval, and submit
before the cutoff with buffer time.
