import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd

from config.hyperparameters import HyperParameters
from evaluation.evaluate_improvement import run as run_improvement_eval
from evaluation.evaluate_thresholds import run as run_threshold_eval
from preprocessing.btc_cleaner import clean_btc_hourly_csv, save_btc_clean
from preprocessing.data_integrator import build_base_dataset, merge_x1_x2, resolve_adapted_window
from preprocessing.sentiment_analyzer import aggregate_hourly_sentiment
from training.train_main_dqn import run as run_main_dqn
from training.train_predictive_dqn import run as run_predictive_dqn
from training.train_trade_dqn import run as run_trade_dqn
from utils.dqn_runtime import StopConfig, configure_gpu, resolve_parallel_workers, set_global_seed
from utils.logger import Logger
from utils.parallel_executor import run_jobs_parallel, run_jobs_serial
from utils.pipeline_state import PipelineState
from utils.run_manager import RunManager
from visualization.plot_improvement_comparison import plot_improvement_comparison
from visualization.plot_price_history import plot_price_history
from visualization.plot_trading_signals import plot_trading_signals


def _make_stop_cfg(
    cfg: HyperParameters,
    minutes: int,
    episode_cap: int,
    metric: str = "roi",
    eval_every_n_episodes: int | None = None,
    early_stop_patience: int | None = None,
    min_improvement_delta: float | None = None,
    min_episodes_before_early_stop: int = 1,
    min_trades_for_best: int = 0,
) -> StopConfig:
    return StopConfig(
        budget_sec=minutes * 60.0,
        episode_cap=episode_cap,
        eval_every_n_episodes=eval_every_n_episodes if eval_every_n_episodes is not None else cfg.EVAL_EVERY_N_EPISODES,
        early_stop_patience=early_stop_patience if early_stop_patience is not None else cfg.EARLY_STOP_PATIENCE,
        min_improvement_delta=min_improvement_delta if min_improvement_delta is not None else cfg.MIN_IMPROVEMENT_DELTA,
        early_stop_metric=metric,
        min_episodes_before_early_stop=min_episodes_before_early_stop,
        min_trades_for_best=min_trades_for_best,
    )


def _deadline_ts(deadline_hours: float) -> float:
    return time.time() + float(deadline_hours) * 3600.0


def _is_deadline_reached(deadline_ts: float) -> bool:
    return time.time() >= deadline_ts


def _report_path(results_dir: str, run_id: str) -> str:
    return str(Path(results_dir) / f"deadline_report_{run_id}.md")


def _parse_int_list(value: str, default: List[int]) -> List[int]:
    if not value:
        return list(default)
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return list(dict.fromkeys(out))


def _parse_alpha_list(value: str, default: List[float]) -> List[float]:
    if not value:
        return list(default)
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        v = float(item)
        if v > 1.0:
            v = v / 100.0
        out.append(v)
    return list(dict.fromkeys(out))


def _quick_validate_prep_outputs(hp: HyperParameters) -> bool:
    """Fast sanity check for cached prep artifacts."""
    specs = [
        (hp.BTC_CLEAN_PATH, {"timestamp", "close"}),
        (hp.SENTIMENT_SCORES_PATH, {"timestamp", "sentiment"}),
        (hp.INTEGRATED_BASE_PATH, {"timestamp", "ap_t", "ts_t", "split"}),
    ]
    try:
        for path, required_cols in specs:
            p = Path(path)
            if not p.exists() or p.stat().st_size == 0:
                return False
            sample = pd.read_csv(p, nrows=64)
            if sample.empty:
                return False
            if not required_cols.issubset(set(sample.columns)):
                return False

        base = pd.read_csv(hp.INTEGRATED_BASE_PATH, usecols=["split"], nrows=2048)
        splits = set(base["split"].dropna().astype(str).unique())
        if not {"train", "test"}.issubset(splits):
            return False
        return True
    except Exception:
        return False


def _build_table8(results_dir: str = "results") -> str:
    rows = [
        {"study": "DNA-S (Betancourt et al., 2021)", "roi": ">24%", "sr": "N/A"},
        {"study": "SharpeD-DQN (Lucarelli et al., 2019)", "roi": "26.14%", "sr": "N/A"},
        {"study": "Double Q-network + Boltzmann (Bu et al., 2018)", "roi": "27.87%", "sr": "N/A"},
        {"study": "DQN (Theate et al., 2021)", "roi": "29.4%", "sr": "N/A"},
        {"study": "TD3 (Majidi et al., 2022)", "roi": "57.5%", "sr": "1.53"},
        {"study": "M-DQN (this run)", "roi": "runtime_output", "sr": "runtime_output"},
    ]
    out = Path(results_dir) / "table8_sota_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def _write_deadline_report(payload: dict, output_path: str) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Deadline Report (20h mode)",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Run ID: {payload.get('run_id')}",
        f"Deadline UTC: {payload.get('deadline_utc')}",
        f"Freeze policy: {payload.get('freeze_policy')}",
        "",
        "## Runtime",
        f"- total_budget_hours: {payload.get('total_budget_hours')}",
        f"- deadline_hours: {payload.get('deadline_hours')}",
        f"- gpu_mode_selected: {payload.get('gpu_mode_selected')}",
        f"- parallel_workers: {payload.get('parallel_workers')}",
        "",
        "## Stage Results",
    ]

    for stage_name, stage_payload in payload.get("stages", {}).items():
        lines.append(f"- {stage_name}: {json.dumps(stage_payload)}")

    lines.extend(["", "## Job Summary"])
    for job in payload.get("jobs", []):
        lines.append(f"- {job['job_id']}: status={job['status']} variant={job.get('variant_id')} method={job.get('method')}")

    lines.extend(["", "## Artifacts"])
    for item in payload.get("artifacts", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Validation"])
    lines.append(f"- min_improvements_required: {payload.get('min_improvements')} ")
    lines.append(f"- improvements_completed: {payload.get('improvements_completed')} ")
    lines.append(f"- requirement_met: {payload.get('improvement_requirement_met')} ")

    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def _maybe_run_node(
    state: PipelineState,
    node_id: str,
    resume_mode: str,
    outputs: List[str],
    fn: Callable[[], dict],
    metadata: dict,
) -> Tuple[dict, bool]:
    """Returns (result, skipped)."""
    if resume_mode in ("auto", "force") and state.node_completed(node_id):
        return {"stopped_reason": "resume_skip", "outputs": outputs}, True

    state.set_node(node_id, "running", outputs=outputs, metadata=metadata)
    res = fn()
    state.set_node(node_id, "completed", outputs=outputs, metadata={**metadata, "result": res})
    return res, False


def _build_main_jobs(
    hp: HyperParameters,
    main_variants: List[str],
    per_job_minutes: int,
    args,
    dataset_map: Dict[str, pd.DataFrame],
    deadline_ts: float,
    alpha_list: List[float],
    omega_list: List[int],
) -> List[Tuple[str, Callable[[], dict], dict]]:
    jobs = []
    for alpha in alpha_list:
        for omega in omega_list:
        # Baselines: classic + proposed, both paper variant.
            for method in ("classic", "proposed"):
                variant_id = "paper"
                branch = "baseline"
                node_id = f"main_{method}_{variant_id}_a{int(alpha*100)}_o{omega}"

                def _make_baseline_fn(alpha=alpha, omega=omega, method=method, variant_id=variant_id, node_id=node_id):
                    return lambda: run_main_dqn(
                        config=hp,
                        integrated_df=dataset_map["proposed_paper"] if method == "proposed" else dataset_map["classic"],
                        stop_cfg=_make_stop_cfg(
                            hp,
                            per_job_minutes,
                            args.main_episode_cap,
                            metric=args.early_stop_metric,
                            eval_every_n_episodes=args.eval_every_n_episodes,
                            early_stop_patience=args.early_stop_patience,
                            min_improvement_delta=args.min_improvement_delta,
                            min_episodes_before_early_stop=args.min_episodes_before_early_stop,
                            min_trades_for_best=args.min_trades_for_best,
                        ),
                        alpha=alpha,
                        omega=omega,
                        method=method,
                        variant_id=variant_id,
                        job_id=node_id,
                        deadline_ts=deadline_ts,
                        resume_from_checkpoint=hp.get_main_dqn_weights_name(alpha, omega, method=method, variant_id=variant_id),
                    )

                jobs.append(
                    (
                        node_id,
                        _make_baseline_fn(),
                        {
                            "alpha": alpha,
                            "omega": omega,
                            "method": method,
                            "variant_id": variant_id,
                            "branch": branch,
                        },
                    )
                )

        # Improvement: policy-gradient Main on proposed paper x2.
            if "policy_gradient" in main_variants:
                variant_id = "policy_gradient"
                method = "proposed"
                branch = "improved"
                node_id = f"main_{method}_{variant_id}_a{int(alpha*100)}_o{omega}"

                def _make_pg_fn(alpha=alpha, omega=omega, method=method, variant_id=variant_id, node_id=node_id):
                    return lambda: run_main_dqn(
                        config=hp,
                        integrated_df=dataset_map["proposed_paper"],
                        stop_cfg=_make_stop_cfg(
                            hp,
                            per_job_minutes,
                            args.main_episode_cap,
                            metric=args.early_stop_metric,
                            eval_every_n_episodes=args.eval_every_n_episodes,
                            early_stop_patience=args.early_stop_patience,
                            min_improvement_delta=args.min_improvement_delta,
                            min_episodes_before_early_stop=args.min_episodes_before_early_stop,
                            min_trades_for_best=args.min_trades_for_best,
                        ),
                        alpha=alpha,
                        omega=omega,
                        method=method,
                        variant_id=variant_id,
                        job_id=node_id,
                        deadline_ts=deadline_ts,
                        resume_from_checkpoint=hp.get_main_dqn_weights_name(alpha, omega, method=method, variant_id=variant_id),
                    )

                jobs.append(
                    (
                        node_id,
                        _make_pg_fn(),
                        {
                            "alpha": alpha,
                            "omega": omega,
                            "method": method,
                            "variant_id": variant_id,
                            "branch": branch,
                        },
                    )
                )
    return jobs


def run_pipeline(args):
    if getattr(args, "results_dir", ""):
        os.environ["BTCA_RESULTS_DIR"] = args.results_dir

    hp = HyperParameters()
    if args.predictive_action_step is not None:
        hp.PREDICTIVE_DQN_ACTION_STEP = float(args.predictive_action_step)
        hp.PREDICTIVE_DQN_OUTPUT_DIM = hp.compute_predictive_output_dim()
    hp.validate()

    logger = Logger("MainPipeline", log_file=str(Path(hp.LOGS_DIR) / "training.log"))
    logger.info(
        "[PHASE B] Predictive action grid: "
        f"min={hp.PREDICTIVE_DQN_ACTION_MIN} max={hp.PREDICTIVE_DQN_ACTION_MAX} "
        f"step={hp.PREDICTIVE_DQN_ACTION_STEP} size={hp.PREDICTIVE_DQN_OUTPUT_DIM}"
    )
    gpu_mode_selected = configure_gpu(args.gpu_mode)
    set_global_seed(args.seed)

    stage_budget = hp.parse_stage_budget(args.stage_budget)
    manager = RunManager(args.total_budget_hours, stage_budget, hp.RUNTIME_AUDIT_LOG)

    workers = resolve_parallel_workers(args.parallel_policy, gpu_mode_selected, max_workers=args.max_workers)
    deadline_ts = _deadline_ts(args.deadline_hours)

    run_id = args.run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    state = PipelineState(hp.RUN_MANIFEST_PATH, hp.RUN_STATE_PATH)
    state.initialize_run(
        run_id=run_id,
        config_obj=vars(args),
        deadline_utc=datetime.utcfromtimestamp(deadline_ts).isoformat() + "Z",
    )

    payload = {
        "run_id": run_id,
        "total_budget_hours": args.total_budget_hours,
        "deadline_hours": args.deadline_hours,
        "deadline_utc": datetime.utcfromtimestamp(deadline_ts).isoformat() + "Z",
        "freeze_policy": args.freeze_policy,
        "gpu_mode_selected": gpu_mode_selected,
        "parallel_workers": workers,
        "stages": {},
        "artifacts": [],
        "jobs": [],
        "min_improvements": args.min_improvements,
        "improvements_completed": 0,
        "improvement_requirement_met": False,
    }

    # Phase A: prep
    manager.start_stage("prep")
    prep_outputs = [hp.BTC_CLEAN_PATH, hp.SENTIMENT_SCORES_PATH, hp.INTEGRATED_BASE_PATH]
    if not _is_deadline_reached(deadline_ts):
        def _prep_fn():
            logger.info("[PHASE A] Building cached datasets")
            btc_df = clean_btc_hourly_csv(hp.BTC_RAW_PATH)
            save_btc_clean(btc_df, hp.BTC_CLEAN_PATH)
            aggregate_hourly_sentiment(hp.TWEETS_VADER_PATH, hp.SENTIMENT_SCORES_PATH)

            sentiment_df = pd.read_csv(hp.SENTIMENT_SCORES_PATH, parse_dates=["timestamp"])
            btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], utc=True)
            sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], utc=True)

            window = resolve_adapted_window(
                btc_df=btc_df,
                sentiment_df=sentiment_df,
                overlap_days=hp.OVERLAP_DAYS_TARGET,
                test_hours=hp.TEST_HOURS,
            )
            base_df = build_base_dataset(btc_df, sentiment_df, window)
            base_df.to_csv(hp.INTEGRATED_BASE_PATH, index=False)
            return {"rows": len(base_df)}

        if args.force_prep_rebuild:
            prep_res = _prep_fn()
            state.set_node("prep", "completed", outputs=prep_outputs, metadata={"stage": "prep", "result": prep_res, "forced": True})
            payload["jobs"].append({"job_id": "prep", "status": "completed", "variant_id": "n/a", "method": "n/a"})
            payload["artifacts"].extend(prep_outputs)
        else:
            cached_ok = _quick_validate_prep_outputs(hp)
            if cached_ok:
                logger.info("[PHASE A] Using existing cached prep artifacts (validated).")
                state.set_node("prep", "completed", outputs=prep_outputs, metadata={"stage": "prep", "cache_validated": True})
                payload["jobs"].append({"job_id": "prep", "status": "skipped", "variant_id": "n/a", "method": "n/a"})
                payload["artifacts"].extend(prep_outputs)
            else:
                prep_res = _prep_fn()
                state.set_node("prep", "completed", outputs=prep_outputs, metadata={"stage": "prep", "result": prep_res})
                payload["jobs"].append({"job_id": "prep", "status": "completed", "variant_id": "n/a", "method": "n/a"})
                payload["artifacts"].extend(prep_outputs)
    else:
        state.set_node("prep", "frozen", outputs=[], metadata={"reason": "deadline_reached"})
        payload["jobs"].append({"job_id": "prep", "status": "frozen", "variant_id": "n/a", "method": "n/a"})

    payload["stages"]["prep"] = manager.end_stage("prep").__dict__

    if args.prep_only:
        report_path = _write_deadline_report(payload, _report_path(hp.RESULTS_DIR, run_id))
        payload["artifacts"].append(report_path)
        Path(hp.RESULTS_DIR, "pipeline_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Prep-only completed. Report: {report_path}")
        return payload

    base_df = pd.read_csv(hp.INTEGRATED_BASE_PATH, parse_dates=["timestamp"])

    # Phase B: trade/predictive variants
    manager.start_stage("trade")
    logger.info("[PHASE B] Training Trade-DQN / Predictive variants")
    predictive_variants = hp.parse_predictive_variant_list(args.predictive_variant_list)
    pre_jobs: List[Tuple[str, Callable[[], dict], List[str], dict]] = []

    def _trade_fn():
        return run_trade_dqn(
            hp,
            base_df,
            _make_stop_cfg(hp, stage_budget["trade"], args.trade_episode_cap, metric=args.early_stop_metric),
            seed=args.seed,
            job_id="trade",
            variant_id="paper",
            deadline_ts=deadline_ts,
            resume_from_checkpoint="trade_dqn",
            episode_max_steps=args.trade_max_steps,
        )

    pre_jobs.append(("trade", _trade_fn, [hp.X1_OUTPUT_PATH], {"variant_id": "paper", "method": "n/a"}))

    if "paper" in predictive_variants:
        def _pred_paper_fn():
            return run_predictive_dqn(
                hp,
                base_df,
                _make_stop_cfg(hp, stage_budget["pred"], args.predictive_episode_cap, metric=args.early_stop_metric),
                seed=args.seed,
                job_id="pred_paper",
                variant_id="paper",
                deadline_ts=deadline_ts,
                resume_from_checkpoint="predictive_dqn",
                output_path=hp.X2_OUTPUT_PATH,
                episode_max_steps=args.predictive_max_steps,
            )
        pre_jobs.append(("pred_paper", _pred_paper_fn, [hp.X2_OUTPUT_PATH], {"variant_id": "paper", "method": "n/a"}))

    if "continuous" in predictive_variants:
        logger.warning("[PHASE B] predictive variant 'continuous' is deprecated in this run and will be ignored.")

    if not _is_deadline_reached(deadline_ts):
        runnable = []
        for job_id, fn, outputs, meta in pre_jobs:
            if args.resume in ("auto", "force") and state.node_completed(job_id):
                payload["jobs"].append(
                    {"job_id": job_id, "status": "skipped", "variant_id": meta["variant_id"], "method": meta["method"]}
                )
                payload["artifacts"].extend(outputs)
                continue
            state.set_node(job_id, "running", outputs=outputs, metadata=meta)
            runnable.append((job_id, fn, outputs, meta))

        logger.info(
            f"[PHASE B] Jobs queued: total={len(pre_jobs)} runnable={len(runnable)} "
            f"skipped={len(pre_jobs)-len(runnable)}"
        )
        run_pairs = [(jid, fn) for jid, fn, _, _ in runnable]
        if run_pairs:
            pre_res = run_jobs_parallel(run_pairs, max_workers=min(2, max(1, workers))) if workers >= 2 and args.parallel_policy != "always_single" else run_jobs_serial(run_pairs)
            for job_id, res in pre_res.items():
                _, _, outputs, meta = next(x for x in runnable if x[0] == job_id)
                state.set_job(job_id, res)
                state.set_node(job_id, "completed", outputs=outputs, metadata={"result": res, **meta})
                payload["jobs"].append(
                    {"job_id": job_id, "status": "completed", "variant_id": res.get("variant_id", meta["variant_id"]), "method": "n/a"}
                )
                payload["artifacts"].extend(outputs)
                logger.info(
                    f"[PHASE B] Completed {job_id} "
                    f"(reason={res.get('stopped_reason')}, episodes={res.get('episodes_completed')})"
                )
    else:
        for jid, _, _, meta in pre_jobs:
            state.set_node(jid, "frozen", outputs=[], metadata={"reason": "deadline_reached"})
            payload["jobs"].append({"job_id": jid, "status": "frozen", "variant_id": meta["variant_id"], "method": "n/a"})

    payload["stages"]["trade_pred"] = manager.end_stage("trade").__dict__
    logger.info(f"[PHASE B] Finished with status: {payload['stages']['trade_pred']}")

    x1_df = pd.read_csv(hp.X1_OUTPUT_PATH, parse_dates=["timestamp"])
    x2_df = pd.read_csv(hp.X2_OUTPUT_PATH, parse_dates=["timestamp"])
    x1_nunique = int(x1_df["x1"].nunique())
    if x1_nunique < args.x1_min_unique:
        raise RuntimeError(
            "Trade-DQN output collapsed before Main. "
            f"x1_nunique={x1_nunique} (min={args.x1_min_unique}). "
            "Re-run trade from clean checkpoint (e.g., run_10h.sh --restart-trade --restart-main-from-b)."
        )
    x2_nunique = int(x2_df["x2"].nunique())
    x2_std = float(x2_df["x2"].std(ddof=0))
    if x2_nunique < args.x2_min_unique or x2_std <= args.x2_min_std:
        raise RuntimeError(
            "Predictive-DQN output collapsed before Main. "
            f"x2_nunique={x2_nunique} (min={args.x2_min_unique}), "
            f"x2_std={x2_std:.12f} (min>{args.x2_min_std}). "
            "Re-run predictive from clean checkpoint (e.g., run_10h.sh --restart-predictive)."
        )
    proposed_paper_df = merge_x1_x2(base_df, x1_df, x2_df)
    proposed_paper_df.to_csv(hp.INTEGRATED_DATASET_PATH, index=False)
    payload["artifacts"].append(hp.INTEGRATED_DATASET_PATH)

    classic_df = base_df.copy()
    ret_sign = classic_df["ap_t"].pct_change().fillna(0.0)
    classic_df["x1"] = ret_sign.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype(int)
    classic_df["x2"] = classic_df["ts_t"].astype(float)

    variants = hp.parse_variant_list(args.main_variant_list)
    selected_variants = []
    for v in variants:
        if v == "paper":
            selected_variants.append(v)
        if v == "policy_gradient":
            selected_variants.append(v)
    if "paper" not in selected_variants:
        selected_variants.insert(0, "paper")

    # Enforce baseline + single improvement (policy gradient).
    selected_variants = [v for v in selected_variants if v in {"paper", "policy_gradient"}]
    selected_variants = list(dict.fromkeys(selected_variants))

    dataset_map = {"proposed_paper": proposed_paper_df, "classic": classic_df}
    alpha_list = _parse_alpha_list(args.main_alpha_list, hp.RISK_LEVELS)
    omega_list = _parse_int_list(args.main_omega_list, hp.ACTIVE_TRADE_THRESHOLDS)
    logger.info(f"[PHASE C/D] Target grid: alpha={alpha_list} omega={omega_list}")

    # Phase C/D: main jobs
    manager.start_stage("main")
    logger.info("[PHASE C/D] Running Main-DQN baseline + improvements grid")
    results_rows = []
    # Build full job list first so time is allocated per actual run.
    dry_jobs = _build_main_jobs(hp, selected_variants, 10, args, dataset_map, deadline_ts, alpha_list, omega_list)
    per_job_minutes = max(8, int(stage_budget["main"] / max(1, len(dry_jobs))))
    main_jobs = _build_main_jobs(hp, selected_variants, per_job_minutes, args, dataset_map, deadline_ts, alpha_list, omega_list)
    logger.info(
        f"[PHASE C/D] Main jobs prepared: total={len(main_jobs)} "
        f"workers={max(1, workers)} per_job_minutes={per_job_minutes}"
    )

    for i in range(0, len(main_jobs), max(1, workers)):
        if _is_deadline_reached(deadline_ts):
            if args.freeze_policy == "hard":
                break
        batch = main_jobs[i:i + max(1, workers)]

        runnable = []
        for node_id, fn, meta in batch:
            if args.resume in ("auto", "force") and state.node_completed(node_id):
                node_meta = state.manifest.get("nodes", {}).get(node_id, {}).get("metadata", {})
                prev_res = node_meta.get("result", {})
                if prev_res and "metrics" in prev_res:
                    row = dict(prev_res["metrics"])
                    row["branch"] = meta["branch"]
                    row["job_id"] = node_id
                    row["is_deadline_eligible"] = prev_res.get("is_deadline_eligible", True)
                    results_rows.append(row)
                payload["jobs"].append({
                    "job_id": node_id,
                    "status": "skipped",
                    "variant_id": meta["variant_id"],
                    "method": meta["method"],
                })
                continue
            if _is_deadline_reached(deadline_ts) and args.freeze_policy == "graceful":
                state.set_node(node_id, "frozen", outputs=[], metadata={"reason": "deadline_reached"})
                payload["jobs"].append({
                    "job_id": node_id,
                    "status": "frozen",
                    "variant_id": meta["variant_id"],
                    "method": meta["method"],
                })
                continue
            runnable.append((node_id, fn, meta))

        if not runnable:
            continue

        logger.info(f"[PHASE C/D] Running batch {i//max(1,workers)+1}: jobs={len(runnable)}")
        run_pairs = [(nid, fn) for nid, fn, _ in runnable]
        batch_res = run_jobs_parallel(run_pairs, max_workers=max(1, workers)) if workers > 1 else run_jobs_serial(run_pairs)

        for node_id, res in batch_res.items():
            meta = next(m for nid, _, m in runnable if nid == node_id)
            state.set_job(node_id, res)
            state.set_node(node_id, "completed", outputs=[res.get("checkpoint_path")], metadata={**meta, "result": res})

            row = dict(res["metrics"])
            row["branch"] = meta["branch"]
            row["job_id"] = node_id
            row["is_deadline_eligible"] = res.get("is_deadline_eligible", True)
            results_rows.append(row)

            payload["jobs"].append({
                "job_id": node_id,
                "status": "completed",
                "variant_id": meta["variant_id"],
                "method": meta["method"],
            })
            best_metric = res.get("best_metric")
            best_metric_s = f"{float(best_metric):.6f}" if best_metric is not None else "n/a"
            logger.info(
                f"[PHASE C/D] Completed {node_id} "
                f"(reason={res.get('stopped_reason')}, best={best_metric_s})"
            )

    payload["stages"]["main"] = manager.end_stage("main").__dict__
    logger.info(f"[PHASE C/D] Finished with status: {payload['stages']['main']}")

    # Phase E: evaluation/report
    manager.start_stage("eval")
    logger.info("[PHASE E] Generating tables/plots/reports")

    threshold_out = run_threshold_eval(results_rows, hp.RESULTS_DIR)
    improvement_out = run_improvement_eval(results_rows, hp.RESULTS_DIR)
    table8_path = _build_table8(hp.RESULTS_DIR)

    payload["artifacts"].extend(threshold_out.get("generated", []))
    payload["artifacts"].extend(improvement_out.get("generated", []))
    payload["artifacts"].append(table8_path)

    plot1 = plot_price_history(base_df, str(Path(hp.RESULTS_DIR) / "price_history.png"))
    plot2 = plot_trading_signals(proposed_paper_df, str(Path(hp.RESULTS_DIR) / "trading_signals.png"))
    payload["artifacts"].extend([plot1, plot2])

    delta_path = str(Path(hp.RESULTS_DIR) / "table_improvement_delta.csv")
    if Path(delta_path).exists():
        plot3 = plot_improvement_comparison(delta_path, str(Path(hp.RESULTS_DIR) / "improvement_comparison.png"))
        payload["artifacts"].append(plot3)

    improved_completed = {
        r["variant_id"] for r in results_rows if r.get("branch") == "improved" and r.get("is_deadline_eligible", True)
    }
    payload["improvements_completed"] = len(improved_completed)
    payload["improvement_requirement_met"] = len(improved_completed) >= args.min_improvements

    payload["stages"]["eval"] = manager.end_stage("eval").__dict__
    logger.info(f"[PHASE E] Finished with status: {payload['stages']['eval']}")

    report_path = _write_deadline_report(payload, _report_path(hp.RESULTS_DIR, run_id))
    payload["artifacts"].append(report_path)

    Path(hp.RESULTS_DIR, "pipeline_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Pipeline completed. Deadline report: {report_path}")

    if not payload["improvement_requirement_met"]:
        raise RuntimeError(
            f"Minimum improvements not met: required={args.min_improvements}, got={payload['improvements_completed']}"
        )

    return payload


def build_parser():
    parser = argparse.ArgumentParser(description="Deadline-driven M-DQN replication and improvements pipeline")
    parser.add_argument("--total-budget-hours", type=int, default=24)
    parser.add_argument("--deadline-hours", type=float, default=20)
    parser.add_argument("--stage-budget", default="prep=120,trade=180,pred=240,main=600,eval=240")
    parser.add_argument("--gpu-mode", choices=["single", "dual", "auto"], default="single")
    parser.add_argument("--parallel-policy", choices=["safe_adaptive", "always_single", "always_dual"], default="safe_adaptive")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--freeze-policy", choices=["graceful", "hard"], default="graceful")
    parser.add_argument("--resume", choices=["auto", "force", "off"], default="auto")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--prep-only", action="store_true")
    parser.add_argument("--force-prep-rebuild", action="store_true")

    parser.add_argument("--degrade-policy", choices=["fidelity_first"], default="fidelity_first")
    parser.add_argument("--improvement-scope", choices=["main_priority", "any_model"], default="main_priority")
    parser.add_argument("--min-improvements", type=int, default=1)
    parser.add_argument("--main-variant-list", default="paper,policy_gradient")
    parser.add_argument("--main-alpha-list", default="")
    parser.add_argument("--main-omega-list", default="")
    parser.add_argument("--predictive-variant-list", default="paper")
    parser.add_argument("--predictive-action-step", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trade-episode-cap", type=int, default=120)
    parser.add_argument("--trade-max-steps", type=int, default=None)
    parser.add_argument("--predictive-episode-cap", type=int, default=120)
    parser.add_argument("--predictive-max-steps", type=int, default=None)
    parser.add_argument("--main-episode-cap", type=int, default=90)
    parser.add_argument("--early-stop-metric", choices=["roi", "sr"], default="roi")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--eval-every-n-episodes", type=int, default=None)
    parser.add_argument("--min-improvement-delta", type=float, default=None)
    parser.add_argument("--min-episodes-before-early-stop", type=int, default=1)
    parser.add_argument("--min-trades-for-best", type=int, default=0)
    parser.add_argument("--x2-min-unique", type=int, default=2)
    parser.add_argument("--x2-min-std", type=float, default=1e-8)
    parser.add_argument("--x1-min-unique", type=int, default=2)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
