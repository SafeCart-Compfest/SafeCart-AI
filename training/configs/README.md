# Experiment configurations

Commit immutable TOML configurations here. A valid result records its configuration,
merged Git SHA, dataset SHA-256, random seed, dependency versions, metrics, and artifact
hashes together.

`distilmbert-v1.toml` is the first bounded Kaggle run: at most 180,000 train and 20,000
dev pairs, effective batch size 32, learning rate `2e-5`, maximum length 192, and at most
three epochs with macro-F1 early stopping. Sample caps protect the 12-hour notebook
budget; increase them only through a documented dev-set experiment.
