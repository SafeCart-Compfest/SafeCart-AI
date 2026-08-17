import sys
from pathlib import Path

import pytest

from safecart_ai.cli.train_matcher import main as train_matcher_main
from safecart_ai.matching.hf_training import WeightedCrossEntropy
from safecart_ai.matching.metrics import bootstrap_classification_ci, classification_metrics
from safecart_ai.matching.pair_text import format_pair, label_id
from safecart_ai.matching.training_config import MatcherTrainingConfig


def valid_config() -> str:
    return """
[model]
name = "distilbert-base-multilingual-cased"
max_length = 192

[training]
seed = 42
learning_rate = 2e-5
epochs = 3
train_batch_size = 16
eval_batch_size = 32
gradient_accumulation_steps = 2
weight_decay = 0.01
warmup_ratio = 0.1
early_stopping_patience = 1
max_train_samples = 100
max_dev_samples = 20
mixed_precision = true
"""


def test_pair_text_uses_stable_structured_boundary() -> None:
    text = format_pair(
        {
            "listing_nie": "NA123",
            "listing_brand": "Lumi Glow",
            "listing_product_name": "Day Cream",
            "listing_package": None,
            "official_nie": "NA123",
            "official_brand": "Lumi Glow",
            "official_product_name": "Night Cream",
            "official_package": "30 g",
            "official_registrant": "Example, PT",
        }
    )

    assert text == (
        "[LISTING] nie=NA123 | brand=Lumi Glow | name=Day Cream | package=\n"
        "[OFFICIAL] nie=NA123 | brand=Lumi Glow | name=Night Cream | "
        "package=30 g | registrant=Example, PT"
    )
    assert label_id("MATCH") == 0
    assert label_id("MISMATCH") == 1
    with pytest.raises(ValueError, match="Unsupported"):
        label_id("UNKNOWN")


def test_training_config_loads_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(valid_config(), encoding="utf-8")

    config = MatcherTrainingConfig.from_toml(path)

    assert config.model_name == "distilbert-base-multilingual-cased"
    assert config.seed == 42
    assert config.max_train_samples == 100
    assert config.mixed_precision


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("epochs = 3", "epochs = 4", "epochs"),
        ("max_length = 192", "max_length = 4", "max_length"),
        ("learning_rate = 2e-5", "learning_rate = 0", "learning_rate"),
        ("warmup_ratio = 0.1", "warmup_ratio = 1", "warmup_ratio"),
        ("weight_decay = 0.01", "weight_decay = -1", "weight_decay"),
        ("train_batch_size = 16", "train_batch_size = 0", "train_batch_size"),
        ("max_train_samples = 100", "max_train_samples = 0", "sample limits"),
    ],
)
def test_training_config_rejects_invalid_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(valid_config().replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MatcherTrainingConfig.from_toml(path)


def test_training_config_requires_tables_and_model_name(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[model]\nname = 'x'\nmax_length = 192\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training"):
        MatcherTrainingConfig.from_toml(path)

    path.write_text(
        valid_config().replace('name = "distilbert-base-multilingual-cased"', 'name = ""'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"model\.name"):
        MatcherTrainingConfig.from_toml(path)


def test_classification_metrics_report_both_classes() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert metrics["macro_f1"] == pytest.approx(0.7333333333)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["label_order"] == ["MATCH", "MISMATCH"]
    with pytest.raises(ValueError, match="equal length"):
        classification_metrics([0], [0, 1])
    with pytest.raises(ValueError, match="at least one"):
        classification_metrics([], [])


def test_bootstrap_classification_interval_is_deterministic() -> None:
    interval = bootstrap_classification_ci([0, 0, 1, 1], [0, 1, 1, 1], samples=100)

    assert interval == bootstrap_classification_ci([0, 0, 1, 1], [0, 1, 1, 1], samples=100)
    macro = interval["macro_f1"]
    assert isinstance(macro, dict)
    assert 0.0 <= macro["low"] <= macro["high"] <= 1.0
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_classification_ci([0], [0, 1])
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_classification_ci([], [])
    with pytest.raises(ValueError, match="positive"):
        bootstrap_classification_ci([0], [0], samples=0)


def test_training_cli_rejects_missing_pair_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(valid_config(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safecart-ai-train-matcher",
            str(tmp_path / "missing.csv"),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        train_matcher_main()


def test_weighted_loss_accepts_trainer_batch_size_keyword() -> None:
    class Logits:
        device = "cuda:0"

    class Outputs:
        logits = Logits()

    class WeightTensor:
        def to(self, device: str) -> tuple[str, str]:
            return ("weights", device)

    class Functional:
        def cross_entropy(
            self, logits: object, labels: object, *, weight: object
        ) -> tuple[object, object, object]:
            return logits, labels, weight

    labels = object()
    weighted_loss = WeightedCrossEntropy(Functional(), WeightTensor())

    assert weighted_loss(Outputs(), labels, num_items_in_batch=8) == (
        Outputs.logits,
        labels,
        ("weights", "cuda:0"),
    )
