import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def available_gpu_count() -> int:
    try:
        import tensorflow as tf

        return len(tf.config.list_physical_devices("GPU"))
    except Exception:
        return 0


def configure_gpu(gpu_mode: str = "single") -> str:
    gpu_mode = (gpu_mode or "single").lower()
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return "cpu"

        if gpu_mode in ("single", "auto"):
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            return "single"

        if gpu_mode == "dual" and len(gpus) >= 2:
            for gpu in gpus[:2]:
                tf.config.experimental.set_memory_growth(gpu, True)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            return "dual"

        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return "single"
    except Exception:
        return "cpu"


def resolve_parallel_workers(parallel_policy: str, gpu_mode: str, max_workers: int = 2) -> int:
    policy = (parallel_policy or "safe_adaptive").lower()
    if policy == "always_single":
        return 1
    if policy == "always_dual":
        return min(2, max_workers)

    # safe_adaptive
    if gpu_mode in ("dual", "auto") and available_gpu_count() >= 2:
        return min(2, max_workers)
    return 1


@dataclass
class StopConfig:
    budget_sec: float
    episode_cap: int
    eval_every_n_episodes: int
    early_stop_patience: int
    min_improvement_delta: float
    early_stop_metric: str = "roi"


@dataclass
class TrainResult:
    elapsed_sec: float
    budget_sec: float
    stopped_reason: str
    best_metric: float
    checkpoint_path: Optional[str]
    episodes_completed: int
    metrics: Dict[str, float]
    job_id: str = ""
    variant_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    is_deadline_eligible: bool = True
