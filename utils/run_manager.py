import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class StageStatus:
    stage: str
    elapsed_sec: float
    budget_sec: float
    stopped_reason: str


class RunManager:
    def __init__(self, total_budget_hours: int, stage_budget_min: Dict[str, int], audit_log_path: str):
        self.total_budget_sec = float(total_budget_hours) * 3600.0
        self.stage_budget_sec = {k: float(v) * 60.0 for k, v in stage_budget_min.items()}
        self.start_time = time.monotonic()
        self.stage_start: Dict[str, float] = {}
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def elapsed_total_sec(self) -> float:
        return time.monotonic() - self.start_time

    def remaining_total_sec(self) -> float:
        return max(0.0, self.total_budget_sec - self.elapsed_total_sec())

    def stage_budget(self, stage: str) -> float:
        return self.stage_budget_sec.get(stage, 0.0)

    def start_stage(self, stage: str) -> None:
        self.stage_start[stage] = time.monotonic()
        self._log_event("stage_start", stage, {"remaining_total_sec": self.remaining_total_sec()})

    def stage_elapsed_sec(self, stage: str) -> float:
        if stage not in self.stage_start:
            return 0.0
        return time.monotonic() - self.stage_start[stage]

    def should_stop_stage(self, stage: str) -> bool:
        return self.stage_elapsed_sec(stage) >= self.stage_budget(stage)

    def can_start_job(self, expected_job_sec: float) -> bool:
        return self.remaining_total_sec() >= expected_job_sec

    def end_stage(self, stage: str, stopped_reason: str = "completed") -> StageStatus:
        elapsed = self.stage_elapsed_sec(stage)
        budget = self.stage_budget(stage)
        status = StageStatus(
            stage=stage,
            elapsed_sec=elapsed,
            budget_sec=budget,
            stopped_reason=stopped_reason,
        )
        self._log_event("stage_end", stage, {
            "elapsed_sec": elapsed,
            "budget_sec": budget,
            "stopped_reason": stopped_reason,
            "remaining_total_sec": self.remaining_total_sec(),
        })
        return status

    def _log_event(self, event: str, stage: str, payload: Optional[dict] = None) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "stage": stage,
            "payload": payload or {},
        }
        with self.audit_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_payload(self, stage: str, payload: dict) -> None:
        self._log_event("payload", stage, payload)
