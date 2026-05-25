from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from src import tokens
from src.checkpoints import resolve_checkpoint
from src.config import Config
from src.dataset import build_tensors, load_a_sentence, load_vocabs
from src.inference import decode_sequence
from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next URLs for one task")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--split-ratio", type=float, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = Config.from_yaml(args.config, root=root)
    split_ratio = args.split_ratio if args.split_ratio is not None else cfg.split_ratio
    ckpt = resolve_checkpoint(cfg, args.checkpoint)

    vocabs, id_vocab, num_tokens = load_vocabs(cfg.vocab_path)
    bundle = build_tensors(cfg, split_ratio=split_ratio)
    _model, encoder, decoder = build_model(
        num_tokens, cfg.latent_dim, weights_path=ckpt
    )

    sentence, task_type, _ = load_a_sentence(
        cfg.data_dir, args.user_id, args.task_id
    )
    training_length = int(len(sentence) * split_ratio)
    input_sentence = sentence[:training_length]

    if task_type == 0:
        tokenized_input = (
            [tokens.SOA]
            + [vocabs.get(w, tokens.MIS) for w in input_sentence]
            + [tokens.COI]
        )
    elif task_type == 1:
        tokenized_input = (
            [tokens.SOA]
            + [vocabs.get(w, tokens.MIS) for w in input_sentence]
            + [tokens.COI]
        )
    else:
        tokenized_input = [tokens.SOA] + [
            vocabs.get(w, tokens.MIS) for w in input_sentence
        ]

    input_max = int(bundle.max_length * split_ratio)
    input_seq = pad_sequences(
        [tokenized_input], value=tokens.PAD, maxlen=input_max
    )

    predicted = decode_sequence(
        encoder,
        decoder,
        input_seq,
        id_vocab,
        bundle.max_length,
        split_ratio,
    )

    print(f"checkpoint: {ckpt}")
    print(f"user_id={args.user_id} task_id={args.task_id} split_ratio={split_ratio}")
    print(f"context URLs ({len(input_sentence)}):")
    for url in input_sentence[-5:]:
        print(f"  - {url}")
    print("predicted continuation:")
    for url in predicted:
        print(f"  -> {url}")


if __name__ == "__main__":
    main()
