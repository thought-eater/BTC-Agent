import time
from typing import Optional

import numpy as np
import tensorflow as tf

from models.trade_dqn.model import build_trade_dqn
from models.trade_dqn.replay_buffer import ReplayBuffer


class TradeDQNAgent:
    def __init__(
        self,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_decay: float,
        epsilon_min: float,
        replay_size: int,
        batch_size: int,
        target_update_freq: int,
    ):
        self.model = build_trade_dqn()
        self.target_model = build_trade_dqn()
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay = ReplayBuffer(replay_size)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, 3))
        q = self.model(np.expand_dims(state, axis=0), training=False).numpy()[0]
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self, global_step: int) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        next_q = self.target_model(next_states, training=False).numpy().max(axis=1)
        target_values = rewards + (1.0 - dones) * self.gamma * next_q

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            action_mask = tf.one_hot(actions, depth=3)
            pred = tf.reduce_sum(q_values * action_mask, axis=1)
            loss = self.loss_fn(target_values, pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        if global_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.numpy())

    def infer_actions(self, states: np.ndarray) -> np.ndarray:
        q = self.model(states.astype(np.float32), training=False).numpy()
        idx = np.argmax(q, axis=1)
        # map index -> {-1,0,1} as x1
        return np.take(np.array([-1, 0, 1], dtype=np.int32), idx)

    def save_replay(self, path: str) -> str:
        return self.replay.dump(path)

    def load_replay(self, path: str) -> bool:
        return self.replay.load(path)

    def save_optimizer_state(self, path: str) -> bool:
        try:
            vars_ = self.optimizer.variables
            np.save(path, np.array([v.numpy() for v in vars_], dtype=object), allow_pickle=True)
            return True
        except Exception:
            return False

    def load_optimizer_state(self, path: str) -> bool:
        try:
            vals = np.load(path, allow_pickle=True)
            if not self.optimizer.variables:
                dummy_x = np.zeros((1, 1), dtype=np.float32)
                with tf.GradientTape() as tape:
                    y = self.model(dummy_x, training=True)
                    loss = tf.reduce_sum(y)
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            for var, val in zip(self.optimizer.variables, vals):
                var.assign(val)
            return True
        except Exception:
            return False
