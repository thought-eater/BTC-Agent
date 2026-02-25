import tensorflow as tf


def build_trade_dqn(input_dim: int = 1, output_dim: int = 3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, activation="linear")(x)
    model = tf.keras.Model(inputs, outputs)
    return model
