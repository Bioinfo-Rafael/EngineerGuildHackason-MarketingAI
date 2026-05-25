from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.checkpoints import resolve_checkpoint
from src.config import Config
from src.dataset import build_tensors
from src.inference import decode_sequence
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate APM checkpoint")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=None,
        help="Override split_ratio (e.g. 0.97 for paper-style prediction eval)",
    )
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(args.config, root=root)
    cfg.ensure_dirs()

    split_ratio = args.split_ratio if args.split_ratio is not None else cfg.split_ratio
    ckpt = resolve_checkpoint(cfg, args.checkpoint)

    bundle = build_tensors(cfg, split_ratio=split_ratio)
    model, encoder, decoder = build_model(
        bundle.num_tokens, cfg.latent_dim, weights_path=ckpt
    )

    n = bundle.input_s.shape[0]
    train_n = int(n * cfg.train_ratio)

    score = model.evaluate(
        [
            bundle.input_s[train_n:],
            bundle.output_s[train_n:],
        ],
        bundle.output_s_one_shot[train_n:],
        verbose=0,
    )
    loss, accuracy = float(score[0]), float(score[1])

    sample_decodes = []
    for i in range(min(args.samples, n - train_n)):
        idx = train_n + i
        pred = decode_sequence(
            encoder,
            decoder,
            bundle.input_s[idx : idx + 1],
            bundle.id_vocab,
            bundle.max_length,
            split_ratio,
        )
        sample_decodes.append(
            {
                "index": int(idx),
                "predicted": pred[:10],
            }
        )

    report = {
        "checkpoint": str(ckpt),
        "split_ratio": split_ratio,
        "holdout_loss": loss,
        "holdout_accuracy": accuracy,
        "train_ratio": cfg.train_ratio,
        "samples": sample_decodes,
    }

    out_path = cfg.metrics_dir / "eval_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
