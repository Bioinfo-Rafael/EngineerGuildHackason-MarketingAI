from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TerminateOnNaN,
)

from src.config import Config
from src.dataset import build_tensors
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train APM (GRU seq2seq)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(args.config, root=root)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.ensure_dirs()

    bundle = build_tensors(cfg)
    model, _encoder, _decoder = build_model(
        bundle.num_tokens, cfg.latent_dim
    )

    n = bundle.input_s.shape[0]
    train_n = int(n * cfg.train_ratio)

    x_enc = bundle.input_s[:train_n]
    x_dec = bundle.output_s[:train_n]
    y = bundle.output_s_one_shot[:train_n]

    print("Shapes:", bundle.input_s.shape, bundle.output_s.shape, y.shape)
    print("Training samples:", train_n, "epochs:", cfg.epochs)

    # One-batch forward smoke test
    model.predict([x_enc[:1], x_dec[:1]], verbose=0)

    best_path = cfg.checkpoint_dir / "best.weights.h5"
    last_path = cfg.checkpoint_dir / "last.weights.h5"
    history = model.fit(
        [x_enc, x_dec],
        y,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        validation_split=cfg.validation_split,
        shuffle=True,
        callbacks=[
            TerminateOnNaN(),
            ModelCheckpoint(
                filepath=str(best_path),
                monitor="val_loss",
                verbose=1,
                save_weights_only=True,
                save_best_only=True,
            ),
            ModelCheckpoint(
                filepath=str(last_path),
                monitor="val_loss",
                verbose=0,
                save_weights_only=True,
                save_best_only=False,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=False,
            ),
        ],
    )

    val_losses = history.history.get("val_loss", [])
    if val_losses:
        plt.figure()
        plt.plot(val_losses, label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("val_loss")
        plt.legend()
        loss_fig = cfg.figures_dir / "val_loss.png"
        plt.savefig(loss_fig)
        plt.close()
        print(f"Saved {loss_fig}")

    metrics_path = cfg.metrics_dir / "train_history.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "epochs": cfg.epochs,
                "split_ratio": cfg.split_ratio,
                "train_ratio": cfg.train_ratio,
                "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
                "best_checkpoint": str(best_path) if best_path.exists() else None,
                "last_checkpoint": str(last_path) if last_path.exists() else None,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {metrics_path}")
    print(f"Best weights: {best_path} (exists={best_path.exists()})")
    print(f"Last weights: {last_path} (exists={last_path.exists()})")


if __name__ == "__main__":
    main()
