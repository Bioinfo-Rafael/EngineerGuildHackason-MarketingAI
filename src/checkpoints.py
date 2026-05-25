from __future__ import annotations

from pathlib import Path

from src.config import Config


def resolve_checkpoint(cfg: Config, explicit: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return explicit

    candidates = [
        cfg.checkpoint_dir / "best.weights.h5",
        cfg.checkpoint_dir / "last.weights.h5",
        cfg.checkpoint_dir / "best.h5",
        cfg.checkpoint_dir / "last.h5",
        cfg.pretrained_path,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found. Run: python -m src.train "
        f"(expected under {cfg.checkpoint_dir})"
    )
