from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    levels: int,
    level_input_features: int,
    aux_input_features: int,
    level_target_features: int,
    scalar_target_features: int,
    rnn_units: int = 64,
) -> keras.Model:
    """Two-headed LSTM emulator over one atmospheric column.

    The vertical levels form the sequence axis (adjacent levels are physically
    coupled by convection/mixing), so an LSTM reads bottom-to-top per column.
    Surface/whole-column scalars (pressure, insolation, surface fluxes) don't
    vary by level, so they're broadcast into the sequence input and additionally
    fed into a small dense head off the LSTM's final state for the scalar
    (surface-flux) outputs, which aren't naturally per-level quantities.
    """
    level_input = layers.Input(shape=(levels, level_input_features), name="level_sequence")
    aux_input = layers.Input(shape=(aux_input_features,), name="surface_aux")

    aux_repeated = layers.RepeatVector(levels)(aux_input)
    x = layers.Concatenate(axis=-1)([level_input, aux_repeated])

    x = layers.LSTM(rnn_units, return_sequences=True)(x)
    x, state_h, _ = layers.LSTM(rnn_units, return_sequences=True, return_state=True)(x)

    level_output = layers.TimeDistributed(layers.Dense(level_target_features), name="level_tendencies")(x)

    scalar_branch = layers.Concatenate()([state_h, aux_input])
    scalar_branch = layers.Dense(32, activation="relu")(scalar_branch)
    scalar_output = layers.Dense(scalar_target_features, name="surface_fluxes")(scalar_branch)

    model = keras.Model(inputs=[level_input, aux_input], outputs=[level_output, scalar_output])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"level_tendencies": "mse", "surface_fluxes": "mse"},
    )
    return model
