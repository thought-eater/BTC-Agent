import tensorflow as tf


def build_predictive_dqn(input_dim: int = 2, output_dim: int = 20001) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)


def build_predictive_continuous_model(input_dim: int = 2) -> tf.keras.Model:
    """Continuous x2 predictor (bounded to [-100, 100])."""
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="tanh")(x)
    out = tf.keras.layers.Lambda(lambda z: z * 100.0)(out)
    return tf.keras.Model(inputs, out)
