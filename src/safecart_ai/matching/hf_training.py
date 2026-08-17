from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from safecart_ai.matching.metrics import classification_metrics
from safecart_ai.matching.pair_text import ID_TO_LABEL, LABEL_TO_ID, format_pair, label_id
from safecart_ai.matching.training_config import MatcherTrainingConfig


class _WeightedLoss(Protocol):
    def __call__(self, outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    configured_sha = os.environ.get("SAFECART_GIT_SHA")
    if configured_sha:
        return configured_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _file_manifest(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _limit(dataset: Any, maximum: int | None, seed: int) -> Any:
    if maximum is None or len(dataset) <= maximum:
        return dataset
    return dataset.shuffle(seed=seed).select(range(maximum))


def _class_weights(labels: list[int]) -> tuple[float, float]:
    counts = Counter(labels)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("Training sample must contain MATCH and MISMATCH labels")
    total = len(labels)
    return total / (2 * counts[0]), total / (2 * counts[1])


def _trainer_metrics(prediction: Any) -> dict[str, float]:
    logits = np.asarray(prediction.predictions)
    labels = np.asarray(prediction.label_ids).astype(int).tolist()
    predicted = np.argmax(logits, axis=-1).astype(int).tolist()
    metrics = classification_metrics(labels, predicted)
    per_class = cast(dict[str, dict[str, float | int]], metrics["per_class"])
    mismatch = per_class["MISMATCH"]
    match = per_class["MATCH"]
    return {
        "macro_f1": cast(float, metrics["macro_f1"]),
        "match_precision": float(match["precision"]),
        "match_recall": float(match["recall"]),
        "mismatch_precision": float(mismatch["precision"]),
        "mismatch_recall": float(mismatch["recall"]),
    }


def _make_weighted_loss(functional: Any, weight_tensor: Any) -> _WeightedLoss:
    def weighted_loss(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
        del num_items_in_batch
        return functional.cross_entropy(
            outputs.logits,
            labels,
            weight=weight_tensor.to(outputs.logits.device),
        )

    return weighted_loss


def _softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return cast(
        np.ndarray[Any, Any],
        exponentials / np.sum(exponentials, axis=1, keepdims=True),
    )


def run_training(pairs_path: Path, config_path: Path, output_dir: Path) -> None:
    """Fine-tune and evaluate the pair classifier on train/dev only."""
    import torch
    import torch.nn.functional as functional
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = MatcherTrainingConfig.from_toml(config_path)
    started_at = datetime.now(UTC)
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("csv", data_files=str(pairs_path), split="train")
    train_source = _limit(
        raw.filter(lambda row: row["split"] == "train"),
        config.max_train_samples,
        config.seed,
    )
    dev_source = _limit(
        raw.filter(lambda row: row["split"] == "dev"),
        config.max_dev_samples,
        config.seed,
    )
    if not len(train_source) or not len(dev_source):
        raise ValueError("Pair dataset must contain non-empty train and dev splits")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize(batch: dict[str, list[object]]) -> dict[str, object]:
        count = len(batch["label"])
        texts = [
            format_pair({field: values[index] for field, values in batch.items()})
            for index in range(count)
        ]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_length,
        )
        encoded["labels"] = [label_id(str(label)) for label in batch["label"]]
        return cast(dict[str, object], encoded)

    train_dataset = train_source.map(
        tokenize,
        batched=True,
        remove_columns=train_source.column_names,
        desc="Tokenizing train pairs",
    )
    dev_dataset = dev_source.map(
        tokenize,
        batched=True,
        remove_columns=dev_source.column_names,
        desc="Tokenizing dev pairs",
    )
    train_labels = [label_id(str(label)) for label in train_source["label"]]
    weights = _class_weights(train_labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    def model_init() -> Any:
        return AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        eval_on_start=True,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.mixed_precision and torch.cuda.is_available(),
        seed=config.seed,
        data_seed=config.seed,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_loss_func=_make_weighted_loss(functional, weight_tensor),
        compute_metrics=_trainer_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
            )
        ],
    )
    train_result = trainer.train()
    prediction = trainer.predict(dev_dataset)
    logits = np.asarray(prediction.predictions)
    labels = np.asarray(prediction.label_ids).astype(int)
    predicted = np.argmax(logits, axis=-1).astype(int)
    probabilities = _softmax(logits)
    metrics = classification_metrics(labels.tolist(), predicted.tolist())

    model_dir = output_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    failures = []
    for index, (actual, predicted_label) in enumerate(zip(labels, predicted, strict=True)):
        if actual == predicted_label:
            continue
        source = dev_source[int(index)]
        failures.append(
            {
                "pair_id": source["pair_id"],
                "mutation_type": source["mutation_type"],
                "actual": ID_TO_LABEL[int(actual)],
                "predicted": ID_TO_LABEL[int(predicted_label)],
                "predicted_probability": float(probabilities[index, predicted_label]),
            }
        )
        if len(failures) == 100:
            break

    failure_path = output_dir / "failure-samples.jsonl"
    with failure_path.open("w", encoding="utf-8") as stream:
        for failure in failures:
            stream.write(f"{json.dumps(failure, sort_keys=True)}\n")

    manifest = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": pairs_path.name,
            "sha256": _sha256(pairs_path),
            "train_rows": len(train_source),
            "dev_rows": len(dev_source),
        },
        "config": {
            "path": config_path.name,
            "sha256": _sha256(config_path),
            "values": asdict(config),
        },
        "class_weights": {"MATCH": weights[0], "MISMATCH": weights[1]},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "datasets": version("datasets"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train_metrics": train_result.metrics,
        "dev_metrics": metrics,
        "model_files": _file_manifest(model_dir),
        "failure_samples": {
            "count": len(failures),
            "sha256": _sha256(failure_path),
        },
    }
    (output_dir / "run-manifest.json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
