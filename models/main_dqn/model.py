import tensorflow as tf

from models.main_dqn.model_dueling_double import build_dueling_main_dqn


def build_main_dqn(input_dim: int = 2, output_dim: int = 3, variant_id: str = "paper") -> tf.keras.Model:
    if variant_id == "dueling_double":
        return build_dueling_main_dqn(input_dim=input_dim, output_dim=output_dim)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_dim, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)
