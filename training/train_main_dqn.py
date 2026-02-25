import argparse
import json
import time
from typing import Dict, Optional

import pandas as pd

from config.hyperparameters import HyperParameters
from evaluation.metrics import roi, sharpe_ratio
from models.main_dqn import MainDQNAgent, MainDQNEnvironment
from utils.checkpoint import CheckpointManager
from utils.dqn_runtime import StopConfig, TrainResult, utcnow_iso


def _evaluate(
    agent: MainDQNAgent,
    df: pd.DataFrame,
    config: HyperParameters,
    alpha: float,
    omega: int,
    reward_mode: str,
) -> Dict[str, float]:
    env = MainDQNEnvironment(
        x1=df["x1"].values,
        x2=df["x2"].values,
        prices=df["ap_t"].values,
        initial_investment=config.INITIAL_INVESTMENT,
        transaction_fee=config.TRANSACTION_FEE,
        alpha=alpha,
        omega=omega,
        reward_mode=reward_mode,
    )
    state = env.reset()
    profits = []
    done = False

    while not done:
        action = int(agent.model(state[None, :], training=False).numpy()[0].argmax())
        next_state, reward, done, info = env.step(action)
        profits.append(reward)
        state = next_state

    final_cash = info["portfolio_value"]
    return {
        "roi": roi(config.INITIAL_INVESTMENT, final_cash),
        "sr": sharpe_ratio(profits),
        "trades": float(len([t for t in env.trades if t[0] in ("buy", "sell")])),
        "final_cash": float(final_cash),
    }


def run(
    config: HyperParameters,
    integrated_df: pd.DataFrame,
    stop_cfg: StopConfig,
    alpha: float,
    omega: int,
    method: str = "proposed",
    variant_id: str = "paper",
    job_id: str = "",
    deadline_ts: Optional[float] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> dict:
    reward_mode_map = {
        "paper": "paper",
        "dueling_double": "paper",
        "predictive_continuous": "paper",
    }
    reward_mode = reward_mode_map.get(variant_id, "paper")

    train_df = integrated_df[integrated_df["split"] == "train"].copy()
    test_df = integrated_df[integrated_df["split"] == "test"].copy()

    env = MainDQNEnvironment(
        x1=train_df["x1"].values,
        x2=train_df["x2"].values,
        prices=train_df["ap_t"].values,
        initial_investment=config.INITIAL_INVESTMENT,
        transaction_fee=config.TRANSACTION_FEE,
        alpha=alpha,
        omega=omega,
        reward_mode=reward_mode,
    )

    agent = MainDQNAgent(
        lr=config.LEARNING_RATE,
        gamma=config.DISCOUNT_FACTOR,
        epsilon_start=config.EPSILON_START,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
        replay_size=config.REPLAY_BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        variant_id=variant_id,
    )

    ckpt = CheckpointManager(config.CHECKPOINTS_DIR)
    ckpt_name = config.get_main_dqn_weights_name(alpha, omega, method=method, variant_id=variant_id)
    progress_path = f"{config.CHECKPOINTS_DIR}/{ckpt_name}_progress.json"
    replay_path = f"{config.CHECKPOINTS_DIR}/{ckpt_name}_replay.pkl"
    optimizer_path = f"{config.CHECKPOINTS_DIR}/{ckpt_name}_optimizer.npy"

    if resume_from_checkpoint and ckpt.exists(resume_from_checkpoint):
        ckpt.load(agent.model, resume_from_checkpoint)
        agent.target_model.set_weights(agent.model.get_weights())
        agent.load_replay(replay_path)
        agent.load_optimizer_state(optimizer_path)

    started_at = utcnow_iso()
    start = time.monotonic()

    best_metric = -float("inf")
    best_path = None
    no_improve = 0
    global_step = 0
    episodes_done = 0

    # Restore progress cursor for stronger resume continuity.
    try:
        with open(progress_path, "r", encoding="utf-8") as fh:
            prog = json.load(fh)
        if prog.get("stopped_reason") != "completed":
            best_metric = float(prog.get("best_metric", best_metric))
            no_improve = int(prog.get("no_improve", no_improve))
            global_step = int(prog.get("global_step", global_step))
            episodes_done = int(prog.get("episodes_done", episodes_done))
            agent.epsilon = float(prog.get("epsilon", agent.epsilon))
    except Exception:
        pass

    while episodes_done < stop_cfg.episode_cap:
        elapsed = time.monotonic() - start
        if elapsed >= stop_cfg.budget_sec:
            stopped = "time_budget"
            break
        if deadline_ts is not None and time.time() >= deadline_ts:
            stopped = "deadline_freeze"
            break

        state = env.reset()
        done = False
        for _ in range(min(len(train_df) - 2, config.MAX_STEPS_PER_EPISODE)):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step(global_step)
            global_step += 1
            state = next_state
            if done:
                break

        episodes_done += 1

        if episodes_done % stop_cfg.eval_every_n_episodes == 0:
            eval_res = _evaluate(agent, test_df, config, alpha, omega, reward_mode=reward_mode)
            metric = eval_res.get(stop_cfg.early_stop_metric, eval_res["roi"])

            if metric > (best_metric + stop_cfg.min_improvement_delta):
                best_metric = metric
                no_improve = 0
                best_path = ckpt.save(
                    model=agent.model,
                    name=ckpt_name,
                    metadata={
                        "episode": episodes_done,
                        "alpha": alpha,
                        "omega": omega,
                        "method": method,
                        "variant_id": variant_id,
                        "roi": eval_res["roi"],
                        "sr": eval_res["sr"],
                    },
                )
                agent.save_replay(replay_path)
                agent.save_optimizer_state(optimizer_path)
            else:
                no_improve += 1

            if no_improve >= stop_cfg.early_stop_patience:
                stopped = "early_stop"
                break

            # Persist job cursor periodically for resume.
            try:
                with open(progress_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "episodes_done": episodes_done,
                            "global_step": global_step,
                            "best_metric": best_metric,
                            "no_improve": no_improve,
                            "epsilon": agent.epsilon,
                            "stopped_reason": "running",
                        },
                        fh,
                        indent=2,
                    )
            except Exception:
                pass
    else:
        stopped = "episode_cap"

    final_eval = _evaluate(agent, test_df, config, alpha, omega, reward_mode=reward_mode)

    finished_at = utcnow_iso()
    result = TrainResult(
        elapsed_sec=time.monotonic() - start,
        budget_sec=stop_cfg.budget_sec,
        stopped_reason=stopped,
        best_metric=float(best_metric if best_metric > -float("inf") else final_eval.get("roi", 0.0)),
        checkpoint_path=best_path,
        episodes_completed=episodes_done,
        metrics={
            "roi": final_eval["roi"],
            "sr": final_eval["sr"],
            "trades": final_eval["trades"],
            "final_cash": final_eval["final_cash"],
            "alpha": alpha,
            "omega": omega,
            "method": method,
            "variant_id": variant_id,
        },
        job_id=job_id,
        variant_id=variant_id,
        started_at=started_at,
        finished_at=finished_at,
        is_deadline_eligible=(deadline_ts is None or time.time() <= deadline_ts),
    )
    try:
        with open(progress_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "episodes_done": episodes_done,
                    "global_step": global_step,
                    "best_metric": result.best_metric,
                    "no_improve": no_improve,
                    "epsilon": agent.epsilon,
                    "stopped_reason": "completed",
                },
                fh,
                indent=2,
            )
        agent.save_replay(replay_path)
        agent.save_optimizer_state(optimizer_path)
    except Exception:
        pass
    return result.__dict__


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/integrated_dataset.csv")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--omega", type=int, default=16)
    parser.add_argument("--method", choices=["proposed", "classic"], default="proposed")
    parser.add_argument("--variant-id", default="paper")
    parser.add_argument("--time-budget-min", type=int, default=65)
    parser.add_argument("--episode-cap", type=int, default=120)
    parser.add_argument("--eval-every-n-episodes", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-improvement-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-metric", choices=["roi", "sr"], default="roi")
    args = parser.parse_args()

    hp = HyperParameters()
    df = pd.read_csv(args.dataset, parse_dates=["timestamp"])
    stop_cfg = StopConfig(
        budget_sec=args.time_budget_min * 60.0,
        episode_cap=args.episode_cap,
        eval_every_n_episodes=args.eval_every_n_episodes,
        early_stop_patience=args.early_stop_patience,
        min_improvement_delta=args.min_improvement_delta,
        early_stop_metric=args.early_stop_metric,
    )
    print(run(hp, df, stop_cfg, alpha=args.alpha, omega=args.omega, method=args.method, variant_id=args.variant_id))


if __name__ == "__main__":
    _cli()
