# Training

Kaggle provides GPU compute. Training logic remains in the tested `safecart_ai`
package; the notebook only installs the package, verifies inputs, and calls the CLI.

The training command:

1. loads only the leakage-safe `train` and `dev` rows;
2. applies a versioned sample cap and seed;
3. serializes structured listing/official pairs;
4. fine-tunes `distilbert-base-multilingual-cased` with class-weighted loss;
5. selects the best checkpoint by dev macro-F1 with early stopping;
6. writes the configuration, Git SHA, dataset hash, dependency versions, metrics,
   confusion matrix, and up to 100 failure references.

It never uses calibration or synthetic-test rows during model selection.

```bash
uv sync --extra ml
uv run safecart-ai-train-matcher /path/to/product-pairs.csv \
  --config training/configs/distilmbert-v1.toml \
  --output /path/to/run-output
```

Use `distilmbert-smoke.toml` to verify the Kaggle environment before running
`distilmbert-v1.toml`. Each run records its config, Git SHA, dataset checksum, seed,
dependencies, metrics, failures, and output checksums. Model weights remain outside Git.
