from __future__ import annotations

import numpy as np
from keras.models import Model

from src import tokens
from src.dataset import TensorBundle


def decode_sequence(
    encoder: Model,
    decoder: Model,
    input_seq: np.ndarray,
    id_vocab: dict[int, str],
    max_length: int,
    split_ratio: float,
) -> list[str]:
    states_value = encoder.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokens.SOP

    decoded: list[str] = []
    max_steps = max_length - int(max_length * split_ratio) + 1

    while len(decoded) <= max_steps:
        output_tokens, h = decoder.predict(
            [target_seq, states_value], verbose=0
        )
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        predicted_url = id_vocab[sampled_token_index]
        decoded.append(predicted_url)

        if sampled_token_index in tokens.EOA_TOKENS:
            break

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = h

    return decoded
