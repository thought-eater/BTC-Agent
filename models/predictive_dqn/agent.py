from typing import Optional

import numpy as np
import tensorflow as tf

from models.predictive_dqn.model import build_predictive_dqn
from models.predictive_dqn.replay_buffer import ReplayBuffer


class PredictiveDQNAgent:
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
        action_size: int = 20001,
    ):
        self.action_size = action_size
        self.model = build_predictive_dqn(output_dim=action_size)
        self.target_model = build_predictive_dqn(output_dim=action_size)
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
        self.state_mean = np.zeros((2,), dtype=np.float32)
        self.state_std = np.ones((2,), dtype=np.float32)

    def set_state_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean.size == 2 and std.size == 2:
            self.state_mean = mean
            self.state_std = np.maximum(std, 1e-6).astype(np.float32)

    def _norm_states(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float32)
        return (x - self.state_mean) / self.state_std

    def act(self, state: np.ndarray, force_random: bool = False) -> int:
        if force_random or np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.action_size))
        q = self.model(np.expand_dims(self._norm_states(state), axis=0), training=False).numpy()[0]
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self, global_step: int) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = self._norm_states(states)
        next_states = self._norm_states(next_states)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        # Double DQN: online net selects, target net evaluates.
        next_online_q = self.model(next_states, training=False).numpy()
        next_actions = np.argmax(next_online_q, axis=1)
        next_target_q_all = self.target_model(next_states, training=False).numpy()
        next_q = next_target_q_all[np.arange(self.batch_size), next_actions]
        target_values = rewards + (1.0 - dones) * self.gamma * next_q

        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            action_mask = tf.one_hot(actions, depth=self.action_size)
            pred = tf.reduce_sum(q_values * action_mask, axis=1)
            loss = self.loss_fn(target_values, pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        if global_step % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        return float(loss.numpy())

    def decay_epsilon(self) -> float:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(self.epsilon)

    def infer_action_indices(self, states: np.ndarray) -> np.ndarray:
        q = self.model(self._norm_states(states), training=False).numpy()
        return np.argmax(q, axis=1).astype(np.int32)

    def boost_exploration(self, epsilon_floor: float = 0.5) -> None:
        self.epsilon = float(max(self.epsilon, epsilon_floor))

    def perturb_policy_head(self, stddev: float = 5e-3) -> None:
        """Small random perturbation on final Dense layer to escape flat argmax policies."""
        if not self.model.layers:
            return
        head = self.model.layers[-1]
        weights = head.get_weights()
        if len(weights) != 2:
            return
        w, b = weights
        rng = np.random.default_rng()
        w = w + rng.normal(0.0, stddev, size=w.shape).astype(np.float32)
        b = b + rng.normal(0.0, stddev, size=b.shape).astype(np.float32)
        head.set_weights([w, b])

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
                dummy_x = np.zeros((1, 2), dtype=np.float32)
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
