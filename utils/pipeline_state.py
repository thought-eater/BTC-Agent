import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class NodeState:
    node_id: str
    status: str
    updated_at: str
    outputs: list
    metadata: Dict[str, Any]


class PipelineState:
    def __init__(self, manifest_path: str, state_path: str):
        self.manifest_path = Path(manifest_path)
        self.state_path = Path(state_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.manifest: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"nodes": {}}

        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"jobs": {}}

    def initialize_run(self, run_id: str, config_obj: Dict[str, Any], deadline_utc: str) -> None:
        config_hash = hashlib.sha256(json.dumps(config_obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        self.manifest.setdefault("run_id", run_id)
        self.manifest.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
        self.manifest["deadline_utc"] = deadline_utc
        self.manifest["config_hash"] = config_hash
        self._save()

    def node_completed(self, node_id: str) -> bool:
        n = self.manifest.get("nodes", {}).get(node_id)
        if not n:
            return False
        if n.get("status") != "completed":
            return False

        for out in n.get("outputs", []):
            if out and not Path(out).exists():
                return False
        return True

    def set_node(self, node_id: str, status: str, outputs: Optional[list] = None, metadata: Optional[dict] = None) -> None:
        self.manifest.setdefault("nodes", {})
        self.manifest["nodes"][node_id] = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "outputs": outputs or [],
            "metadata": metadata or {},
        }
        self._save()

    def set_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.state.setdefault("jobs", {})
        self.state["jobs"][job_id] = payload
        self._save()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.state.get("jobs", {}).get(job_id)

    def _save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
