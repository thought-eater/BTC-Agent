from dataclasses import dataclass


@dataclass(frozen=True)
class EmergencyProfile:
    """Fixed emergency-time settings for 2-hour completion mode."""

    MAIN_EPISODE_CAP: int = 3
    MAIN_EPISODE_CAP_FALLBACK: int = 4
    EVAL_EVERY_N_EPISODES: int = 1
    EARLY_STOP_PATIENCE: int = 3
    MIN_EPISODES_BEFORE_EARLY_STOP: int = 3
    MIN_TRADES_FOR_BEST: int = 5
    MIN_IMPROVEMENT_DELTA: float = 0.0
    PER_JOB_MINUTES: int = 8
    TOTAL_HOURS: float = 2.0
    GPU_MODE: str = "single"
    SAFETY_MARGIN_SEC: int = 300
