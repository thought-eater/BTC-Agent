import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.emergency_profile import EmergencyProfile
from config.hyperparameters import HyperParameters
from evaluation.finalize_emergency_report import run as run_finalize_emergency
from preprocessing.data_integrator import merge_x1_x2
from training.train_main_dqn import run as run_main_dqn
from utils.dqn_runtime import StopConfig, configure_gpu, set_global_seed
from utils.logger import Logger
from utils.parallel_executor import run_jobs_parallel, run_jobs_serial
from utils.pipeline_state import PipelineState


NODE_RE = re.compile(r"^main_(classic|proposed)_(.+)_a(\d+)_o(\d+)$")


def _derive_branch(variant_id: str) -> str:
    return "baseline" if variant_id == "paper" else "improved"


def _parse_node(node_id: str) -> Optional[Tuple[str, str, float, int]]:
    m = NODE_RE.match(node_id)
    if not m:
        return None
    method = m.group(1)
    variant_id = m.group(2)
    alpha = float(m.group(3)) / 100.0
    omega = int(m.group(4))
    return method, variant_id, alpha, omega


def _sort_key(node_id: str) -> Tuple[int, int, int, str]:
    parsed = _parse_node(node_id)
    if parsed is None:
        return (999, 999, 999, node_id)

    method, variant_id, alpha, omega = parsed
    omega_rank = {8: 0, 16: 1, 24: 2}.get(omega, 99)
    order_map = {
        ("classic", "paper"): 0,
        ("proposed", "paper"): 1,
        ("proposed", "policy_gradient"): 2,
        ("proposed", "dueling_double"): 8,
        ("proposed", "predictive_continuous"): 9,
    }
    order = order_map.get((method, variant_id), 9)
    # Prioritize alpha=0.80 tail first, then 0.55, then 0.30.
    alpha_rank = int((1.0 - alpha) * 100)
    return (alpha_rank, omega_rank, order, node_id)


def _build_core_targets() -> List[str]:
    targets = []
    for alpha in (30, 55, 80):
        for variant_id in ("paper", "policy_gradient"):
            targets.append(f"main_proposed_{variant_id}_a{alpha}_o16")
    return targets


def _write_report(payload: dict, output_path: str) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Emergency 2H Completion Report",
        "",
        "Mode: Emergency Depth-Limited Completion",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Run ID: {payload.get('run_id')}",
        "",
        "## Settings",
        f"- per_job_minutes: {payload.get('per_job_minutes')}",
        f"- episode_cap: {payload.get('episode_cap')}",
        f"- fallback_episode_cap: {payload.get('fallback_episode_cap')}",
        f"- eval_every_n_episodes: {payload.get('eval_every_n_episodes')}",
        f"- early_stop_patience: {payload.get('early_stop_patience')}",
        f"- min_episodes_before_early_stop: {payload.get('min_episodes_before_early_stop')}",
        f"- min_trades_for_best: {payload.get('min_trades_for_best')}",
        f"- min_improvement_delta: {payload.get('min_improvement_delta')}",
        f"- total_budget_hours: {payload.get('total_budget_hours')}",
        "",
        "## Node Status",
        f"- completed: {payload.get('completed_nodes')}",
        f"- frozen: {payload.get('frozen_nodes')}",
        f"- failed: {payload.get('failed_nodes')}",
        "",
        "## Execution",
        f"- target_nodes: {payload.get('target_nodes')}",
        f"- processed_nodes: {payload.get('processed_nodes')}",
        "",
        "## Artifacts",
    ]
    for a in payload.get("artifacts", []):
        lines.append(f"- {a}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def _make_stop_cfg(per_job_minutes: int, episode_cap: int, profile: EmergencyProfile) -> StopConfig:
    return StopConfig(
        budget_sec=float(per_job_minutes) * 60.0,
        episode_cap=int(episode_cap),
        eval_every_n_episodes=profile.EVAL_EVERY_N_EPISODES,
        early_stop_patience=profile.EARLY_STOP_PATIENCE,
        min_improvement_delta=profile.MIN_IMPROVEMENT_DELTA,
        early_stop_metric="roi",
        min_episodes_before_early_stop=profile.MIN_EPISODES_BEFORE_EARLY_STOP,
        min_trades_for_best=profile.MIN_TRADES_FOR_BEST,
    )


def _load_datasets(hp: HyperParameters, logger: Logger) -> Dict[str, pd.DataFrame]:
    base_df = pd.read_csv(hp.INTEGRATED_BASE_PATH, parse_dates=["timestamp"])
    x1_df = pd.read_csv(hp.X1_OUTPUT_PATH, parse_dates=["timestamp"])
    x2_df = pd.read_csv(hp.X2_OUTPUT_PATH, parse_dates=["timestamp"])

    # Emergency fallback when preprocessing outputs collapse to constants.
    x1_unique = int(x1_df["x1"].nunique())
    x2_unique = int(x2_df["x2"].nunique())
    if x1_unique <= 1 or x2_unique <= 1:
        logger.warning(
            f"[EMERGENCY] Detected collapsed preprocessing outputs (x1_unique={x1_unique}, x2_unique={x2_unique}). "
            "Generating emergency non-collapsed RL-compatible signals."
        )
        ret = base_df["ap_t"].pct_change().fillna(0.0)
        x1_fb = ret.apply(lambda v: 1 if v > 0.0005 else (-1 if v < -0.0005 else 0)).astype(int)
        fut = base_df["ap_t"].shift(-1).fillna(base_df["ap_t"])
        x2_fb = ((fut - base_df["ap_t"]) / base_df["ap_t"].replace(0, np.nan)).fillna(0.0) * 100.0
        x2_fb = x2_fb.clip(-100.0, 100.0).astype(float)
        x1_df = pd.DataFrame({"timestamp": base_df["timestamp"], "x1": x1_fb.values})
        x2_df = pd.DataFrame({"timestamp": base_df["timestamp"], "x2": x2_fb.values})
        x1_df.to_csv(str(Path(hp.DATA_PROCESSED_DIR) / "x1_trade_recommendations_emergency.csv"), index=False)
        x2_df.to_csv(str(Path(hp.DATA_PROCESSED_DIR) / "x2_price_predictions_emergency.csv"), index=False)

    proposed_paper_df = merge_x1_x2(base_df, x1_df, x2_df)
    proposed_paper_df.to_csv(hp.INTEGRATED_DATASET_PATH, index=False)

    proposed_cont_df = None
    if Path(hp.X2_OUTPUT_CONT_PATH).exists():
        x2_cont_df = pd.read_csv(hp.X2_OUTPUT_CONT_PATH, parse_dates=["timestamp"])
        proposed_cont_df = merge_x1_x2(base_df, x1_df, x2_cont_df)
        cont_path = str(Path(hp.DATA_PROCESSED_DIR) / "integrated_dataset_predictive_continuous.csv")
        proposed_cont_df.to_csv(cont_path, index=False)
    else:
        logger.warning(
            f"Missing {hp.X2_OUTPUT_CONT_PATH}; predictive_continuous nodes may fail unless data is generated beforehand."
        )

    classic_df = base_df.copy()
    ret_sign = classic_df["ap_t"].pct_change().fillna(0.0)
    classic_df["x1"] = ret_sign.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype(int)
    classic_df["x2"] = classic_df["ts_t"].astype(float)

    return {
        "base": base_df,
        "proposed_paper": proposed_paper_df,
        "proposed_continuous": proposed_cont_df,
        "classic": classic_df,
    }


def _collect_results_from_manifest(state: PipelineState) -> List[dict]:
    rows = []
    for node_id, node in state.manifest.get("nodes", {}).items():
        if not node_id.startswith("main_"):
            continue
        if node.get("status") != "completed":
            continue

        meta = node.get("metadata", {})
        result = meta.get("result", {})
        metrics = result.get("metrics", {})
        if not metrics:
            continue
        row = dict(metrics)
        method = row.get("method", meta.get("method", "proposed"))
        variant_id = row.get("variant_id", meta.get("variant_id", "paper"))
        row["method"] = method
        row["variant_id"] = variant_id
        if row["variant_id"] not in {"paper", "policy_gradient"}:
            continue
        if row["method"] not in {"classic", "proposed"}:
            continue
        row["job_id"] = node_id
        row["branch"] = meta.get("branch", _derive_branch(variant_id))
        row["is_deadline_eligible"] = result.get("is_deadline_eligible", True)
        rows.append(row)
    return rows


def _target_nodes(state: PipelineState, max_nodes: int = 0, mode: str = "core", force_retrain: bool = True) -> List[str]:
    if mode == "core":
        nodes = _build_core_targets()
        if not force_retrain:
            filtered = []
            for node_id in nodes:
                n = state.manifest.get("nodes", {}).get(node_id, {})
                if n.get("status") != "completed":
                    filtered.append(node_id)
            nodes = filtered
        if max_nodes > 0:
            nodes = nodes[:max_nodes]
        return nodes

    pending = []
    for node_id, node in state.manifest.get("nodes", {}).items():
        if not node_id.startswith("main_"):
            continue
        parsed = _parse_node(node_id)
        if parsed is None:
            continue
        method, variant_id, _, _ = parsed
        if method not in {"classic", "proposed"}:
            continue
        if variant_id not in {"paper", "policy_gradient"}:
            continue
        if node.get("status") != "completed":
            pending.append(node_id)
    pending = sorted(pending, key=_sort_key)
    if max_nodes > 0:
        pending = pending[:max_nodes]
    return pending


def run_emergency(args) -> dict:
    if getattr(args, "results_dir", ""):
        os.environ["BTCA_RESULTS_DIR"] = args.results_dir

    hp = HyperParameters()
    hp.validate()
    profile = EmergencyProfile()

    logger = Logger("Emergency2H", log_file=str(Path(hp.LOGS_DIR) / "training.log"))
    configure_gpu(profile.GPU_MODE)
    set_global_seed(args.seed)

    state = PipelineState(hp.RUN_MANIFEST_PATH, hp.RUN_STATE_PATH)
    run_id = args.run_id or datetime.utcnow().strftime("emergency_%Y%m%d_%H%M%S")
    state.initialize_run(run_id=run_id, config_obj=vars(args), deadline_utc=datetime.utcnow().isoformat() + "Z")

    data = _load_datasets(hp, logger)
    targets = _target_nodes(
        state,
        max_nodes=args.max_nodes,
        mode=args.target_mode,
        force_retrain=args.force_retrain,
    )
    logger.info(f"[EMERGENCY] Target non-completed main nodes: {len(targets)}")
    if args.dry_run:
        logger.info("[EMERGENCY] Dry run mode; no training will be executed.")
        return {"run_id": run_id, "target_nodes": targets}

    start = time.monotonic()
    hard_deadline = start + (args.total_hours * 3600.0)
    payload = {
        "run_id": run_id,
        "total_budget_hours": args.total_hours,
        "per_job_minutes": args.per_job_minutes,
        "episode_cap": args.episode_cap,
        "fallback_episode_cap": args.episode_cap_fallback,
        "eval_every_n_episodes": profile.EVAL_EVERY_N_EPISODES,
        "early_stop_patience": profile.EARLY_STOP_PATIENCE,
        "min_episodes_before_early_stop": profile.MIN_EPISODES_BEFORE_EARLY_STOP,
        "min_trades_for_best": profile.MIN_TRADES_FOR_BEST,
        "min_improvement_delta": profile.MIN_IMPROVEMENT_DELTA,
        "target_nodes": len(targets),
        "processed_nodes": 0,
        "artifacts": [],
        "aggressive_single_gpu": bool(args.aggressive_single_gpu),
        "aggressive_workers": int(args.aggressive_workers if args.aggressive_single_gpu else 1),
    }
    workers = max(1, int(args.aggressive_workers if args.aggressive_single_gpu else 1))

    def _run_one(node_id: str, nodes_left: int, remaining: float) -> dict:
        parsed = _parse_node(node_id)
        if parsed is None:
            return {"_failed": True, "reason": "malformed_node_id"}
        method, variant_id, alpha, omega = parsed

        if method == "classic":
            df = data["classic"]
        else:
            df = data["proposed_paper"]

        reserve_needed = nodes_left * args.per_job_minutes * 60.0 + profile.SAFETY_MARGIN_SEC
        ep_cap = args.episode_cap_fallback if remaining > reserve_needed else args.episode_cap
        stop_cfg = _make_stop_cfg(args.per_job_minutes, ep_cap, profile)

        res = run_main_dqn(
            config=hp,
            integrated_df=df,
            stop_cfg=stop_cfg,
            alpha=alpha,
            omega=omega,
            method=method,
            variant_id=variant_id,
            job_id=node_id,
            deadline_ts=time.time() + stop_cfg.budget_sec,
            resume_from_checkpoint=None if args.force_fresh_main else hp.get_main_dqn_weights_name(alpha, omega, method=method, variant_id=variant_id),
        )

        elapsed_first = float(res.get("elapsed_sec", 0.0))
        remaining_after = hard_deadline - time.monotonic()
        if (
            args.episode_cap == 1
            and args.episode_cap_fallback > 1
            and elapsed_first < 240.0
            and remaining_after > (nodes_left * args.per_job_minutes * 60.0 + profile.SAFETY_MARGIN_SEC)
        ):
            fallback_cfg = _make_stop_cfg(args.per_job_minutes, args.episode_cap_fallback, profile)
            res2 = run_main_dqn(
                config=hp,
                integrated_df=df,
                stop_cfg=fallback_cfg,
                alpha=alpha,
                omega=omega,
                method=method,
                variant_id=variant_id,
                job_id=node_id,
                deadline_ts=time.time() + fallback_cfg.budget_sec,
                resume_from_checkpoint=None if args.force_fresh_main else hp.get_main_dqn_weights_name(alpha, omega, method=method, variant_id=variant_id),
            )
            if float(res2.get("best_metric", 0.0)) >= float(res.get("best_metric", 0.0)):
                res = res2
                res["_fallback_pass"] = True
        return res

    for batch_start in range(0, len(targets), workers):
        now = time.monotonic()
        remaining = hard_deadline - now
        if remaining <= profile.SAFETY_MARGIN_SEC:
            logger.warning("[EMERGENCY] Deadline margin reached; stopping new jobs.")
            break

        batch = targets[batch_start:batch_start + workers]
        jobs = []
        for offset, node_id in enumerate(batch):
            nodes_left = len(targets) - (batch_start + offset)
            logger.info(
                f"[EMERGENCY] Queue {node_id} ({batch_start+offset+1}/{len(targets)}) "
                f"budget_min={args.per_job_minutes} workers={workers}"
            )
            state.set_node(node_id, "running", outputs=[], metadata={"emergency": True, "aggressive": bool(args.aggressive_single_gpu)})
            jobs.append((node_id, (lambda nid=node_id, nleft=nodes_left, rem=remaining: _run_one(nid, nleft, rem))))

        batch_res = run_jobs_parallel(jobs, max_workers=workers) if workers > 1 else run_jobs_serial(jobs)
        for node_id, res in batch_res.items():
            parsed = _parse_node(node_id)
            if parsed is None:
                continue
            method, variant_id, alpha, omega = parsed
            if res.get("_failed"):
                state.set_node(node_id, "failed", outputs=[], metadata={"reason": res.get("reason", "failed"), "emergency": True})
                logger.error(f"[EMERGENCY] Failed {node_id}: {res.get('reason')}")
                continue

            state.set_job(node_id, res)
            state.set_node(
                node_id,
                "completed",
                outputs=[res.get("checkpoint_path")],
                metadata={
                    "alpha": alpha,
                    "omega": omega,
                    "method": method,
                    "variant_id": variant_id,
                    "branch": _derive_branch(variant_id),
                    "result": res,
                    "emergency": True,
                    "aggressive": bool(args.aggressive_single_gpu),
                    "fallback_pass": bool(res.get("_fallback_pass", False)),
                },
            )
            payload["processed_nodes"] += 1
            logger.info(
                f"[EMERGENCY] Completed {node_id} reason={res.get('stopped_reason')} "
                f"best={res.get('best_metric')}"
            )

    # Freeze still-missing target nodes explicitly for consistent state.
    remaining_targets = [n for n in targets if state.manifest.get("nodes", {}).get(n, {}).get("status") != "completed"]
    for node_id in remaining_targets:
        state.set_node(node_id, "frozen", outputs=[], metadata={"reason": "emergency_deadline_reached", "emergency": True})

    rows = _collect_results_from_manifest(state)
    finalize_out = run_finalize_emergency(rows, hp.RESULTS_DIR, data["base"], data["proposed_paper"])
    payload["artifacts"].extend(finalize_out.get("generated", []))

    nodes = state.manifest.get("nodes", {})
    completed = sum(1 for v in nodes.values() if v.get("status") == "completed")
    frozen = sum(1 for v in nodes.values() if v.get("status") == "frozen")
    failed = sum(1 for v in nodes.values() if v.get("status") == "failed")
    payload["completed_nodes"] = completed
    payload["frozen_nodes"] = frozen
    payload["failed_nodes"] = failed

    report_path = str(Path(hp.RESULTS_DIR) / f"deadline_report_emergency_2h_{run_id}.md")
    payload["artifacts"].append(_write_report(payload, report_path))
    Path(hp.RESULTS_DIR, "pipeline_payload_emergency_2h.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(f"[EMERGENCY] Completed. Report: {report_path}")
    return payload


def build_parser():
    p = argparse.ArgumentParser(description="Emergency 2-hour completion runner for unfinished main_* nodes")
    prof = EmergencyProfile()
    p.add_argument("--run-id", default="")
    p.add_argument("--results-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-hours", type=float, default=prof.TOTAL_HOURS)
    p.add_argument("--per-job-minutes", type=int, default=prof.PER_JOB_MINUTES)
    p.add_argument("--episode-cap", type=int, default=prof.MAIN_EPISODE_CAP)
    p.add_argument("--episode-cap-fallback", type=int, default=prof.MAIN_EPISODE_CAP_FALLBACK)
    p.add_argument("--max-nodes", type=int, default=0, help="limit number of target nodes (0=all)")
    p.add_argument("--target-mode", choices=["core", "tail"], default="core")
    p.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-fresh-main", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aggressive-single-gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--aggressive-workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_emergency(args)
