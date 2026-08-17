# Retrieval dev baseline — 2026-08-17

## Purpose

Measure whether hybrid retrieval returns the source BPOM record in the reranking set,
and verify that exact NIE lookup is not hiding weak lexical retrieval.

These are **synthetic-dev** results. They are engineering checks only and are not the
headline real-listing evaluation result.

## Reproducibility

- Split: `dev`; seed: `42`; sampled positive queries: 1,000.
- Normalized BPOM records: 275,582.
- Catalog SHA-256: `298e5fc5b08ae7901dd0b7912bdffd764cd505d71b2a25104cce7faa0d2014ca`.
- Pair SHA-256: `a2016a60458c3bed4edcc8ad5902ead2dd2f615ebcc2f0166956829ee4ba42f1`.
- CI: percentile bootstrap, 2,000 resamples.
- Machine: Windows 11, Intel Core Ultra 5 225H, 14 cores/logical processors,
  15.5 GiB memory.

## Results

| Mode | Recall@1 (95% CI) | Recall@5 (95% CI) | Recall@20 | MRR (95% CI) | Mean query |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact NIE + lexical | 93.1% (91.5–94.6) | 99.9% (99.7–100) | 100% | 0.9627 (0.9540–0.9712) | 80.1 ms |
| Lexical only | 92.3% (90.7–94.0) | 99.8% (99.5–100) | 100% | 0.9590 (0.9500–0.9681) | 57.1 ms |

The global sparse TF-IDF index took 18.0 seconds for the normal run and 16.7 seconds for
the lexical-only run. It is startup work, not per-request work.

## Interpretation

The target Recall@5 of at least 95% is met on synthetic dev with a lower confidence
bound above 99%. Lexical-only Recall@5 remains 99.8%, so missing NIE does not collapse
candidate generation on these normalized pairs. The earlier per-query TF-IDF fit had
higher Recall@1 (94.6%) but equivalent Recall@5 (99.9%) and materially slower execution;
the global index is retained because the cross-encoder consumes top-5 candidates.

## Limitations and next action

- Generated positives are cleaner than marketplace OCR and must not stand in for real
  screenshots.
- Mean timing is not p95 end-to-end latency and excludes OCR/model inference.
- The final synthetic-test split and future real-listing evaluation set were not inspected.
- Next, benchmark exact, deterministic, pretrained, and fine-tuned pair classification
  on the same leakage-safe dev contract.
