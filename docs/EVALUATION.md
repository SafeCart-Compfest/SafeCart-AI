# Evaluation

SafeCart evaluates listing-to-record identity matching, not physical authenticity or
product safety. Synthetic data is used for development; final claims require reviewed
marketplace screenshots that were not used for model or threshold selection.

## Current results

| Check | Result |
| --- | ---: |
| Retrieval Recall@5, exact NIE + lexical | 99.9% |
| Retrieval Recall@5, lexical only | 99.8% |
| Exact-NIE baseline macro-F1 | 99.9644% |
| Deterministic baseline macro-F1 | 99.9959% |

These results use the synthetic development split. They confirm the pipeline works but
do not measure marketplace performance. The deterministic baseline leaves too little
headroom on this split to justify selecting a trained model from it.

A small Kaggle T4 smoke run also completed training and artifact export successfully.
Its 96 training rows and 48 development rows validate the training environment only;
its score is not reported as model quality.

## Final evaluation

The reviewed screenshot set must cover readable matches, verified field mismatches,
ambiguous official records, missing fields, and unreadable screenshots. Two reviewers
label each example independently and resolve disagreements. Thresholds are selected on
a calibration split, never on the final test split.

The final report will include retrieval recall, mismatch precision and recall,
false-positive rate, calibration error, OCR NIE accuracy, and end-to-end latency.

Validate the dataset with:

```bash
uv run safecart-ai-validate-evaluation-dataset private/evaluation-dataset.csv --final
```
