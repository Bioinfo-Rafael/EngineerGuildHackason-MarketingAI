from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    data_dir: Path
    vocab_path: Path
    split_ratio: float
    latent_dim: int
    batch_size: int
    epochs: int
    train_ratio: float
    validation_split: float
    early_stopping_patience: int
    checkpoint_dir: Path
    log_dir: Path
    metrics_dir: Path
    figures_dir: Path
    pretrained_path: Path

    @classmethod
    def from_yaml(cls, path: Path, root: Path | None = None) -> Config:
        root = root or path.parent.parent
        with path.open() as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        def p(key: str) -> Path:
            return (root / raw[key]).resolve()

        return cls(
            data_dir=p("data_dir"),
            vocab_path=p("vocab_path"),
            split_ratio=float(raw["split_ratio"]),
            latent_dim=int(raw["latent_dim"]),
            batch_size=int(raw["batch_size"]),
            epochs=int(raw["epochs"]),
            train_ratio=float(raw["train_ratio"]),
            validation_split=float(raw["validation_split"]),
            early_stopping_patience=int(raw["early_stopping_patience"]),
            checkpoint_dir=p("checkpoint_dir"),
            log_dir=p("log_dir"),
            metrics_dir=p("metrics_dir"),
            figures_dir=p("figures_dir"),
            pretrained_path=p("pretrained_path"),
        )

    def ensure_dirs(self) -> None:
        for d in (
            self.checkpoint_dir,
            self.log_dir,
            self.metrics_dir,
            self.figures_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
