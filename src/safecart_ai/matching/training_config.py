from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MatcherTrainingConfig:
    model_name: str
    seed: int
    max_length: int
    learning_rate: float
    epochs: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int
    max_train_samples: int | None
    max_dev_samples: int | None
    mixed_precision: bool

    @classmethod
    def from_toml(cls, path: Path) -> MatcherTrainingConfig:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
        model = _table(payload, "model")
        training = _table(payload, "training")
        config = cls(
            model_name=str(model["name"]),
            seed=int(training["seed"]),
            max_length=int(model["max_length"]),
            learning_rate=float(training["learning_rate"]),
            epochs=float(training["epochs"]),
            train_batch_size=int(training["train_batch_size"]),
            eval_batch_size=int(training["eval_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            weight_decay=float(training["weight_decay"]),
            warmup_ratio=float(training["warmup_ratio"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            max_train_samples=_optional_positive_int(training.get("max_train_samples")),
            max_dev_samples=_optional_positive_int(training.get("max_dev_samples")),
            mixed_precision=bool(training["mixed_precision"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model.name must not be empty")
        if self.max_length < 8:
            raise ValueError("model.max_length must be at least 8")
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if not 0 < self.epochs <= 3:
            raise ValueError("training.epochs must be in (0, 3]")
        for name, value in (
            ("train_batch_size", self.train_batch_size),
            ("eval_batch_size", self.eval_batch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("early_stopping_patience", self.early_stopping_patience),
        ):
            if value < 1:
                raise ValueError(f"training.{name} must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("training.warmup_ratio must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must not be negative")


def _table(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing [{name}] configuration table")
    return value


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    parsed = int(str(value))
    if parsed < 1:
        raise ValueError("sample limits must be positive when provided")
    return parsed
