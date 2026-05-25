from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from src import tokens
from src.config import Config


@dataclass
class TensorBundle:
    input_s: np.ndarray
    output_s: np.ndarray
    output_s_one_shot: np.ndarray
    max_length: int
    vocabs: dict[str, int]
    id_vocab: dict[int, str]
    num_tokens: int


def load_vocabs(vocab_path: Path) -> tuple[dict[str, int], dict[int, str], int]:
    vocabs: dict[str, int] = {}
    with vocab_path.open() as f:
        for idx, line in enumerate(f):
            vocabs[line.strip()] = idx
    id_vocab = {value: key for key, value in vocabs.items()}
    return vocabs, id_vocab, len(vocabs)


def load_clickstream(data_dir: Path, user_id: int, task_id: int) -> list[dict]:
    with (data_dir / f"{user_id}.json").open() as f:
        return json.load(f)[task_id - 1]["clickstream"]


def load_a_sentence(
    data_dir: Path, user_id: int, task_id: int
) -> tuple[list[str], int, list[float]]:
    clickstream = load_clickstream(data_dir, user_id, task_id)
    urls = [obj["previous_url"] for obj in clickstream]
    durations = [obj["stay_seconds"] for obj in clickstream]
    return urls, (task_id - 1) % 3, durations


def build_tensors(cfg: Config, split_ratio: float | None = None) -> TensorBundle:
    split_ratio = split_ratio if split_ratio is not None else cfg.split_ratio
    vocabs, id_vocab, num_tokens = load_vocabs(cfg.vocab_path)

    input_sentences: list[list[int]] = []
    output_sentences: list[list[int]] = []
    max_length = 0
    min_length = 10**9

    for user_id in range(1, 22):
        for task_id in range(1, 10):
            sentence, task_type, _duration = load_a_sentence(
                cfg.data_dir, user_id, task_id
            )
            max_length = max(max_length, len(sentence))
            min_length = min(min_length, len(sentence))
            training_length = int(len(sentence) * split_ratio)

            input_sentence = sentence[:training_length]
            output_sentence = sentence[training_length:]

            if task_type == 0:
                tokenized_input = (
                    [tokens.SOA]
                    + [vocabs.get(w, tokens.MIS) for w in input_sentence]
                    + [tokens.COI]
                )
                tokenized_output = (
                    [vocabs.get(w, tokens.MIS) for w in output_sentence]
                    + [tokens.EOA_GOAL]
                )
            elif task_type == 1:
                tokenized_input = (
                    [tokens.SOA]
                    + [vocabs.get(w, tokens.MIS) for w in input_sentence]
                    + [tokens.COI]
                )
                tokenized_output = (
                    [vocabs.get(w, tokens.MIS) for w in output_sentence]
                    + [tokens.EOA_FUZZY]
                )
            else:
                tokenized_input = [tokens.SOA] + [
                    vocabs.get(w, tokens.MIS) for w in input_sentence
                ]
                tokenized_output = (
                    [vocabs.get(w, tokens.MIS) for w in output_sentence]
                    + [tokens.EOA_EXPLORE]
                )

            input_sentences.append(tokenized_input)
            output_sentences.append(tokenized_output)

    input_max = int(max_length * split_ratio)
    output_max = max_length - int(max_length * split_ratio)

    input_s = pad_sequences(
        input_sentences, value=tokens.PAD, maxlen=input_max
    )
    output_s = pad_sequences(
        output_sentences,
        value=tokens.PAD,
        maxlen=output_max,
        padding="post",
    )
    output_s_one_shot = np.array(
        [to_categorical(line, num_classes=num_tokens) for line in output_s]
    )

    return TensorBundle(
        input_s=input_s,
        output_s=output_s,
        output_s_one_shot=output_s_one_shot,
        max_length=max_length,
        vocabs=vocabs,
        id_vocab=id_vocab,
        num_tokens=num_tokens,
    )
