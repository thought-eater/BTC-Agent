import tensorflow as tf


def build_dueling_main_dqn(input_dim: int = 2, output_dim: int = 3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    value = tf.keras.layers.Dense(32, activation="relu")(x)
    value = tf.keras.layers.Dense(1, activation="linear")(value)

    advantage = tf.keras.layers.Dense(32, activation="relu")(x)
    advantage = tf.keras.layers.Dense(output_dim, activation="linear")(advantage)

    mean_adv = tf.keras.layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    q_values = tf.keras.layers.Add()([value, tf.keras.layers.Subtract()([advantage, mean_adv])])
    return tf.keras.Model(inputs, q_values)
