from collections import deque
import pickle
import random
from typing import Iterable


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self._buffer)

    def to_list(self):
        return list(self._buffer)

    def load_list(self, items: Iterable):
        self._buffer = deque(items, maxlen=self.capacity)

    def dump(self, path: str) -> str:
        with open(path, "wb") as fh:
            pickle.dump({"capacity": self.capacity, "items": self.to_list()}, fh)
        return path

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
            self.load_list(payload.get("items", []))
            return True
        except Exception:
            return False
