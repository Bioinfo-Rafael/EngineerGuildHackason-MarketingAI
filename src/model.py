from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from keras.layers import Dense, Embedding, GRU, Input, TimeDistributed
from keras.models import Model
from keras.regularizers import l2

if TYPE_CHECKING:
    pass


def build_model(
    num_tokens: int,
    latent_dim: int,
    weights_path: Path | None = None,
) -> tuple[Model, Model, Model]:
    encoder_input = Input(shape=(None,), name="ContextEncoderInputRaw")
    embedded_encoder_input = Embedding(
        num_tokens, latent_dim, mask_zero=True
    )(encoder_input)
    encoder = GRU(latent_dim, return_state=True)
    _, hidden_state = encoder(embedded_encoder_input)
    context_state = hidden_state

    decoder_input = Input(shape=(None,), name="ContextDecoderInputRaw")
    embedded_decoder_input = Embedding(
        num_tokens, latent_dim, mask_zero=True
    )(decoder_input)
    decoder = GRU(
        latent_dim,
        return_sequences=True,
        return_state=True,
        kernel_regularizer=l2(1e-7),
        activity_regularizer=l2(1e-7),
    )
    x, _ = decoder(embedded_decoder_input, initial_state=context_state)
    decoder_dense = TimeDistributed(
        Dense(num_tokens, activation="softmax")
    )
    decoder_outputs = decoder_dense(x)

    model = Model([encoder_input, decoder_input], decoder_outputs)
    if weights_path is not None and weights_path.exists():
        model.load_weights(str(weights_path))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    encoder_model = Model(encoder_input, context_state)

    decoder_state_input_h = Input(shape=(None,), name="ContextDecoderInput")
    dec_out, decoder_state_h = decoder(
        embedded_decoder_input, initial_state=decoder_state_input_h
    )
    dec_out = decoder_dense(dec_out)
    decoder_model = Model(
        [decoder_input, decoder_state_input_h],
        [dec_out, decoder_state_h],
    )

    return model, encoder_model, decoder_model
