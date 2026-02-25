import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List


@dataclass
class HyperParameters:
    """Centralized configuration for deadline-driven M-DQN replication."""

    LEARNING_RATE: float = 0.001
    DISCOUNT_FACTOR: float = 0.95
    EPSILON_START: float = 1.0
    EPSILON_DECAY: float = 0.995
    EPSILON_MIN: float = 0.01
    BATCH_SIZE: int = 64
    TARGET_UPDATE_FREQ: int = 400
    REPLAY_BUFFER_SIZE: int = 10_000

    TRANSACTION_FEE: float = 0.015
    INITIAL_INVESTMENT: float = 1_000_000.0
    HOLD_PENALTY_LIMIT: int = 20
    BUY_PENALTY_LIMIT: int = 20

    TRADE_DQN_INPUT_DIM: int = 1
    TRADE_DQN_LAYERS: List[int] = field(default_factory=lambda: [64, 32, 8])
    TRADE_DQN_OUTPUT_DIM: int = 3

    PREDICTIVE_DQN_INPUT_DIM: int = 2
    PREDICTIVE_DQN_LAYERS: List[int] = field(default_factory=lambda: [64, 64, 64])
    PREDICTIVE_DQN_OUTPUT_DIM: int = 20_001
    PREDICTIVE_DQN_ACTION_MIN: float = -100.0
    PREDICTIVE_DQN_ACTION_MAX: float = 100.0
    PREDICTIVE_DQN_ACTION_STEP: float = 0.01

    MAIN_DQN_INPUT_DIM: int = 2
    MAIN_DQN_LAYERS: List[int] = field(default_factory=lambda: [64, 64, 64])
    MAIN_DQN_OUTPUT_DIM: int = 3

    RISK_LEVELS: List[float] = field(default_factory=lambda: [0.30, 0.55, 0.80])
    ACTIVE_TRADE_THRESHOLDS: List[int] = field(default_factory=lambda: [8, 16, 24])

    OVERLAP_DAYS_TARGET: int = 1505
    TOTAL_HOURS_TARGET: int = 36_120
    TEST_HOURS: int = 720

    TOTAL_BUDGET_HOURS: int = 24
    DEADLINE_HOURS_DEFAULT: int = 20
    STAGE_BUDGET_MIN: Dict[str, int] = field(
        default_factory=lambda: {
            "prep": 120,
            "trade": 180,
            "pred": 240,
            "main": 600,
            "eval": 240,
        }
    )

    MAX_WALL_TIME_PER_JOB_MIN: int = 65
    MAX_STEPS_PER_EPISODE: int = 2000
    EARLY_STOP_PATIENCE: int = 8
    MIN_IMPROVEMENT_DELTA: float = 1e-4
    EVAL_EVERY_N_EPISODES: int = 2

    BTC_RAW_PATH: str = "data/raw/bitcoin_price/btc_hourly.csv"
    TWEETS_RAW_PATH: str = "data/raw/twitter/engtweetsbtc_clean.csv"
    TWEETS_VADER_PATH: str = "data/raw/twitter/engtweetsbtc_vader_sentiment.csv"

    DATA_PROCESSED_DIR: str = "data/processed"
    BTC_CLEAN_PATH: str = "data/processed/btc_price_clean.csv"
    SENTIMENT_SCORES_PATH: str = "data/processed/tweet_sentiment_scores.csv"
    INTEGRATED_BASE_PATH: str = "data/processed/integrated_base.csv"
    X1_OUTPUT_PATH: str = "data/processed/x1_trade_recommendations.csv"
    X2_OUTPUT_PATH: str = "data/processed/x2_price_predictions.csv"
    X2_OUTPUT_CONT_PATH: str = "data/processed/x2_price_predictions_continuous.csv"
    INTEGRATED_DATASET_PATH: str = "data/processed/integrated_dataset.csv"

    CHECKPOINTS_DIR: str = "utils/checkpoints"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    RUNTIME_AUDIT_LOG: str = "logs/runtime_audit.jsonl"
    RUN_MANIFEST_PATH: str = "results/run_manifest.json"
    RUN_STATE_PATH: str = "results/run_state.json"

    def __post_init__(self) -> None:
        os.makedirs(self.DATA_PROCESSED_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.LOGS_DIR, exist_ok=True)

    def parse_stage_budget(self, stage_budget: str) -> Dict[str, int]:
        parsed = dict(self.STAGE_BUDGET_MIN)
        if not stage_budget:
            return parsed

        for item in stage_budget.split(","):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip().lower()
            if k in parsed:
                parsed[k] = int(v.strip())
        return parsed

    def parse_variant_list(self, variant_list: str) -> List[str]:
        if not variant_list:
            return ["paper", "dueling_double"]
        out = [x.strip() for x in variant_list.split(",") if x.strip()]
        if "paper" not in out:
            out.insert(0, "paper")
        return out

    def parse_predictive_variant_list(self, variant_list: str) -> List[str]:
        if not variant_list:
            return ["paper", "continuous"]
        out = [x.strip() for x in variant_list.split(",") if x.strip()]
        if "paper" not in out:
            out.insert(0, "paper")
        return out

    def get_all_alpha_omega_combinations(self) -> List[tuple]:
        return [(alpha, omega) for alpha in self.RISK_LEVELS for omega in self.ACTIVE_TRADE_THRESHOLDS]

    def get_main_dqn_weights_name(self, alpha: float, omega: int, method: str = "proposed", variant_id: str = "paper") -> str:
        return f"main_dqn_{method}_{variant_id}_alpha{int(alpha*100)}_omega{omega}"

    @property
    def train_hours(self) -> int:
        return self.TOTAL_HOURS_TARGET - self.TEST_HOURS

    def validate(self) -> bool:
        assert self.TOTAL_HOURS_TARGET == self.OVERLAP_DAYS_TARGET * 24
        assert self.TEST_HOURS == 720
        assert self.PREDICTIVE_DQN_OUTPUT_DIM == 20001
        assert self.BATCH_SIZE == 64
        assert self.TARGET_UPDATE_FREQ == 400
        return True

    @staticmethod
    def hours_to_timedelta(hours: int) -> timedelta:
        return timedelta(hours=hours)
