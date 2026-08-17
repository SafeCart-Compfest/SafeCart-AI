# Training

Training code must be executable as a regular Python CLI. Kaggle notebooks and kernels
are thin launchers only; preprocessing, splitting, fitting, calibration, evaluation, and
export logic belong in the tested `safecart_ai` package.

Every run records its git revision, configuration, random seed, source hashes, dependency
versions, metrics, thresholds, and artifact hashes. Raw datasets and generated outputs are
never committed.
