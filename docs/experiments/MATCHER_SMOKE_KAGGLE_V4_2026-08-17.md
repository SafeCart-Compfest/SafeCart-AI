# Kaggle matcher smoke v4 — 17 August 2026

## Purpose

This private Kaggle run validates that the pinned training pipeline can execute on a
Tesla T4, verify its input dataset, train the multilingual cross-encoder, and export a
hashed run manifest and model files. It is not a model-selection or acceptance-gate run.

## Run details

- Kernel: `ryuorlandotamin/safecart-matcher-smoke-v1`, version 4.
- Source commit: `4c406c7e7db005c1eb15fd0bef3de5f676ebe1ef`.
- Pair dataset SHA-256:
  `a2016a60458c3bed4edcc8ad5902ead2dd2f615ebcc2f0166956829ee4ba42f1`.
- Config SHA-256:
  `d856b40cff0585ae13190a2204ccb45f906d842b1d862d9e4d217f71147e0614`.
- Downloaded run-manifest SHA-256:
  `329f230d4856be12f6f094ef840175b03d1787ac53f75c2c0a22e008b0ee28b6`.
- Runtime: Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 4.57.6,
  Datasets 4.8.5, and Tesla T4.

The run used 96 train rows, 48 dev rows, seed 42, and one epoch. Training itself took
7.31 seconds after environment and data setup. The exported manifest records individual
SHA-256 hashes for every model and tokenizer file. Weights and checkpoints remain outside
Git.

## Result and interpretation

The pipeline completed and produced the expected artifacts. The tiny dev split scored
macro-F1 0.2381 with confusion matrix `[[15, 0], [33, 0]]` in label order
`MATCH`, `MISMATCH`. This weak result is expected from a one-epoch, 96-row technical
smoke and must not be presented as a model quality result.

The full synthetic dev set is already saturated by deterministic rules, so a larger
fine-tuning run would currently spend GPU budget without answering the competition
hypothesis. The next valid step is collecting two independent reviews for real marketplace
mismatch and insufficient-evidence cases. Model selection remains blocked until that
evaluation set exists and is finalized.
