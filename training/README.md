# Training

Kaggle provides compute; the tested `safecart_ai` package remains the source of truth.
The training CLI:

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

The Transformers adapter is excluded from local line coverage because it requires the
external model and accelerator stack. Pure serialization, configuration, and metric
contracts are unit-tested in CI; the full adapter receives a Kaggle smoke run before a
long training run is accepted.
