import argparse
import json
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


def _make_stop_cfg(cfg: HyperParameters, minutes: int, episode_cap: int, metric: str = "roi") -> StopConfig:
    return StopConfig(
        budget_sec=minutes * 60.0,
        episode_cap=episode_cap,
        eval_every_n_episodes=cfg.EVAL_EVERY_N_EPISODES,
        early_stop_patience=cfg.EARLY_STOP_PATIENCE,
        min_improvement_delta=cfg.MIN_IMPROVEMENT_DELTA,
        early_stop_metric=metric,
    )


def _deadline_ts(deadline_hours: float) -> float:
    return time.time() + float(deadline_hours) * 3600.0


def _is_deadline_reached(deadline_ts: float) -> bool:
    return time.time() >= deadline_ts


def _report_path(results_dir: str, run_id: str) -> str:
    return str(Path(results_dir) / f"deadline_report_{run_id}.md")


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
) -> List[Tuple[str, Callable[[], dict], dict]]:
    jobs = []
    for alpha, omega in hp.get_all_alpha_omega_combinations():
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

        # Improvement 1: dueling+double on proposed paper x2.
        if "dueling_double" in main_variants:
            variant_id = "dueling_double"
            method = "proposed"
            branch = "improved"
            node_id = f"main_{method}_{variant_id}_a{int(alpha*100)}_o{omega}"

            def _make_dueling_fn(alpha=alpha, omega=omega, method=method, variant_id=variant_id, node_id=node_id):
                return lambda: run_main_dqn(
                    config=hp,
                    integrated_df=dataset_map["proposed_paper"],
                    stop_cfg=_make_stop_cfg(
                        hp,
                        per_job_minutes,
                        args.main_episode_cap,
                        metric=args.early_stop_metric,
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
                    _make_dueling_fn(),
                    {
                        "alpha": alpha,
                        "omega": omega,
                        "method": method,
                        "variant_id": variant_id,
                        "branch": branch,
                    },
                )
            )

        # Improvement 2: predictive continuous x2 with paper Main-DQN (independent test).
        if "predictive_continuous" in main_variants and "proposed_continuous" in dataset_map:
            variant_id = "predictive_continuous"
            method = "proposed"
            branch = "improved"
            node_id = f"main_{method}_{variant_id}_a{int(alpha*100)}_o{omega}"

            def _make_predcont_fn(alpha=alpha, omega=omega, method=method, variant_id=variant_id, node_id=node_id):
                return lambda: run_main_dqn(
                    config=hp,
                    integrated_df=dataset_map["proposed_continuous"],
                    stop_cfg=_make_stop_cfg(
                        hp,
                        per_job_minutes,
                        args.main_episode_cap,
                        metric=args.early_stop_metric,
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
                    _make_predcont_fn(),
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
    hp = HyperParameters()
    hp.validate()

    logger = Logger("MainPipeline", log_file="logs/training.log")
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
    if not _is_deadline_reached(deadline_ts):
        logger.info("[PHASE A] Building cached datasets")

        def _prep_fn():
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

        prep_outputs = [hp.BTC_CLEAN_PATH, hp.SENTIMENT_SCORES_PATH, hp.INTEGRATED_BASE_PATH]
        prep_res, prep_skipped = _maybe_run_node(
            state, "prep", args.resume, prep_outputs, _prep_fn, {"stage": "prep"}
        )
        payload["jobs"].append({"job_id": "prep", "status": "skipped" if prep_skipped else "completed", "variant_id": "n/a", "method": "n/a"})
        payload["artifacts"].extend(prep_outputs)
    else:
        state.set_node("prep", "frozen", outputs=[], metadata={"reason": "deadline_reached"})
        payload["jobs"].append({"job_id": "prep", "status": "frozen", "variant_id": "n/a", "method": "n/a"})

    payload["stages"]["prep"] = manager.end_stage("prep").__dict__

    base_df = pd.read_csv(hp.INTEGRATED_BASE_PATH, parse_dates=["timestamp"])

    # Phase B: trade/predictive variants
    manager.start_stage("trade")
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
            )
        pre_jobs.append(("pred_paper", _pred_paper_fn, [hp.X2_OUTPUT_PATH], {"variant_id": "paper", "method": "n/a"}))

    if "continuous" in predictive_variants:
        def _pred_cont_fn():
            return run_predictive_dqn(
                hp,
                base_df,
                _make_stop_cfg(hp, stage_budget["pred"], args.predictive_episode_cap, metric=args.early_stop_metric),
                seed=args.seed,
                job_id="pred_continuous",
                variant_id="continuous",
                deadline_ts=deadline_ts,
                resume_from_checkpoint="predictive_dqn_continuous",
                output_path=hp.X2_OUTPUT_CONT_PATH,
            )
        pre_jobs.append(("pred_continuous", _pred_cont_fn, [hp.X2_OUTPUT_CONT_PATH], {"variant_id": "continuous", "method": "n/a"}))

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
    else:
        for jid, _, _, meta in pre_jobs:
            state.set_node(jid, "frozen", outputs=[], metadata={"reason": "deadline_reached"})
            payload["jobs"].append({"job_id": jid, "status": "frozen", "variant_id": meta["variant_id"], "method": "n/a"})

    payload["stages"]["trade_pred"] = manager.end_stage("trade").__dict__

    x1_df = pd.read_csv(hp.X1_OUTPUT_PATH, parse_dates=["timestamp"])
    x2_df = pd.read_csv(hp.X2_OUTPUT_PATH, parse_dates=["timestamp"])
    proposed_paper_df = merge_x1_x2(base_df, x1_df, x2_df)
    proposed_paper_df.to_csv(hp.INTEGRATED_DATASET_PATH, index=False)
    payload["artifacts"].append(hp.INTEGRATED_DATASET_PATH)

    proposed_cont_df = None
    if Path(hp.X2_OUTPUT_CONT_PATH).exists():
        x2_cont_df = pd.read_csv(hp.X2_OUTPUT_CONT_PATH, parse_dates=["timestamp"])
        proposed_cont_df = merge_x1_x2(base_df, x1_df, x2_cont_df)
        cont_path = str(Path(hp.DATA_PROCESSED_DIR) / "integrated_dataset_predictive_continuous.csv")
        proposed_cont_df.to_csv(cont_path, index=False)
        payload["artifacts"].append(cont_path)

    classic_df = base_df.copy()
    ret_sign = classic_df["ap_t"].pct_change().fillna(0.0)
    classic_df["x1"] = ret_sign.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).astype(int)
    classic_df["x2"] = classic_df["ts_t"].astype(float)

    variants = hp.parse_variant_list(args.main_variant_list)
    selected_variants = []
    for v in variants:
        if v == "paper":
            selected_variants.append(v)
        if v == "dueling_double":
            selected_variants.append(v)
    if "continuous" in predictive_variants:
        selected_variants.append("predictive_continuous")
    if "paper" not in selected_variants:
        selected_variants.insert(0, "paper")

    # Enforce only baseline + the two requested independent improvements.
    selected_variants = [v for v in selected_variants if v in {"paper", "dueling_double", "predictive_continuous"}]
    selected_variants = list(dict.fromkeys(selected_variants))

    dataset_map = {"proposed_paper": proposed_paper_df, "classic": classic_df}
    if proposed_cont_df is not None:
        dataset_map["proposed_continuous"] = proposed_cont_df

    # Phase C/D: main jobs
    manager.start_stage("main")
    results_rows = []
    # Build full job list first so time is allocated per actual run.
    dry_jobs = _build_main_jobs(hp, selected_variants, 10, args, dataset_map, deadline_ts)
    per_job_minutes = max(8, int(stage_budget["main"] / max(1, len(dry_jobs))))
    main_jobs = _build_main_jobs(hp, selected_variants, per_job_minutes, args, dataset_map, deadline_ts)

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

    payload["stages"]["main"] = manager.end_stage("main").__dict__

    # Phase E: evaluation/report
    manager.start_stage("eval")
    logger.info("[PHASE E] Generating tables/plots/reports")

    threshold_out = run_threshold_eval(results_rows, hp.RESULTS_DIR)
    improvement_out = run_improvement_eval(results_rows, hp.RESULTS_DIR)
    table8_path = _build_table8(hp.RESULTS_DIR)

    payload["artifacts"].extend(threshold_out.get("generated", []))
    payload["artifacts"].extend(improvement_out.get("generated", []))
    payload["artifacts"].append(table8_path)

    plot1 = plot_price_history(base_df, "results/price_history.png")
    plot2 = plot_trading_signals(proposed_paper_df, "results/trading_signals.png")
    payload["artifacts"].extend([plot1, plot2])

    delta_path = str(Path(hp.RESULTS_DIR) / "table_improvement_delta.csv")
    if Path(delta_path).exists():
        plot3 = plot_improvement_comparison(delta_path, "results/improvement_comparison.png")
        payload["artifacts"].append(plot3)

    improved_completed = {
        r["variant_id"] for r in results_rows if r.get("branch") == "improved" and r.get("is_deadline_eligible", True)
    }
    payload["improvements_completed"] = len(improved_completed)
    payload["improvement_requirement_met"] = len(improved_completed) >= args.min_improvements

    payload["stages"]["eval"] = manager.end_stage("eval").__dict__

    report_path = _write_deadline_report(payload, _report_path(hp.RESULTS_DIR, run_id))
    payload["artifacts"].append(report_path)

    Path("results/pipeline_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
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

    parser.add_argument("--degrade-policy", choices=["fidelity_first"], default="fidelity_first")
    parser.add_argument("--improvement-scope", choices=["main_priority", "any_model"], default="main_priority")
    parser.add_argument("--min-improvements", type=int, default=2)
    parser.add_argument("--main-variant-list", default="paper,dueling_double")
    parser.add_argument("--predictive-variant-list", default="paper,continuous")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trade-episode-cap", type=int, default=120)
    parser.add_argument("--predictive-episode-cap", type=int, default=120)
    parser.add_argument("--main-episode-cap", type=int, default=90)
    parser.add_argument("--early-stop-metric", choices=["roi", "sr"], default="roi")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
