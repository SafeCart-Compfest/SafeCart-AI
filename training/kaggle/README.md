# Kaggle launcher contract

Kaggle kernels are thin, private launchers. They may clone an exact merged Git SHA,
install the `ml` extra, verify the private pair-dataset SHA-256, invoke
`safecart-ai-train-matcher`, and expose run outputs. They must not contain hidden
preprocessing, splitting, metric, or model logic.

## Inputs

- Private dataset: `ryuorlandotamin/safecart-product-pairs-v1`.
- Expected file SHA-256:
  `a2016a60458c3bed4edcc8ad5902ead2dd2f615ebcc2f0166956829ee4ba42f1`.
- Versioned config: `training/configs/distilmbert-v1.toml`.
- `SAFECART_GIT_SHA`: exact merged commit to clone and record.

## Accelerator

Use an NVIDIA T4. Current Kaggle CLI documentation warns that the default image's
PyTorch build does not include kernels for the older P100 architecture. A kernel must
fail before training if CUDA is unavailable; silently falling back to CPU would waste
the run budget and make timing incomparable.

## Run sequence

1. Run a tiny train/dev smoke configuration and inspect the manifest.
2. Only after the smoke run succeeds, launch `distilmbert-v1.toml`.
3. Download and review `run-manifest.json` and `failure-samples.jsonl`.
4. Do not publish weights until evaluation, calibration, model card, and checksum review
   are complete.

`smoke/run.py` is the submitted private-kernel launcher. It pins the merged source SHA,
verifies the attached dataset before installing dependencies, fails if CUDA is absent,
and checks the output manifest SHA before reporting success.
