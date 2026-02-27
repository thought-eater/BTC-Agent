import argparse
import json
import time

import numpy as np
import pandas as pd

from config.hyperparameters import HyperParameters
from models.trade_dqn import TradeDQNAgent, TradeDQNEnvironment
from utils.checkpoint import CheckpointManager
from utils.dqn_runtime import StopConfig, TrainResult, utcnow_iso
from utils.logger import Logger


def run(
    config: HyperParameters,
    base_df: pd.DataFrame,
    stop_cfg: StopConfig,
    seed: int = 42,
    job_id: str = "trade",
    variant_id: str = "paper",
    deadline_ts: float = None,
    resume_from_checkpoint: str = None,
    episode_max_steps: int | None = None,
) -> dict:
    logger = Logger("TradeDQNTrain", log_file=f"{config.LOGS_DIR}/training.log")
    train_df = base_df[base_df["split"] == "train"].copy()

    env = TradeDQNEnvironment(
        train_df["ap_t"].values,
        hold_penalty_limit=config.HOLD_PENALTY_LIMIT,
        buy_penalty_limit=config.BUY_PENALTY_LIMIT,
    )

    agent = TradeDQNAgent(
        lr=config.LEARNING_RATE,
        gamma=config.DISCOUNT_FACTOR,
        epsilon_start=config.EPSILON_START,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
        replay_size=config.REPLAY_BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
    )
    state_train = train_df[["ap_t"]].values.astype(np.float32)
    agent.set_state_normalization(state_train.mean(axis=0), state_train.std(axis=0))

    ckpt = CheckpointManager(config.CHECKPOINTS_DIR)
    progress_path = f"{config.CHECKPOINTS_DIR}/trade_dqn_progress.json"
    replay_path = f"{config.CHECKPOINTS_DIR}/trade_dqn_replay.pkl"
    optimizer_path = f"{config.CHECKPOINTS_DIR}/trade_dqn_optimizer.npy"
    if resume_from_checkpoint and ckpt.exists(resume_from_checkpoint):
        ckpt.load(agent.model, resume_from_checkpoint)
        agent.target_model.set_weights(agent.model.get_weights())
        agent.load_replay(replay_path)
        agent.load_optimizer_state(optimizer_path)

    started_at = utcnow_iso()
    start = time.monotonic()

    best_metric = -float("inf")
    best_path = None
    best_div_path = None
    no_improve = 0
    global_step = 0
    episodes_done = 0
    collapse_streak = 0
    best_div_unique = -1
    best_div_metric = -float("inf")
    rng = np.random.default_rng(int(seed))
    max_steps = int(episode_max_steps) if episode_max_steps is not None else int(config.MAX_STEPS_PER_EPISODE)
    episode_steps = int(min(len(train_df) - 2, max(512, max_steps)))
    max_start = max(0, len(train_df) - episode_steps - 2)
    warmup_episodes = max(8, min(30, int(0.3 * max(1, stop_cfg.episode_cap))))
    min_unique_actions = 2
    min_action_std = 0.1
    diversity_states = train_df[["ap_t"]].values.astype(np.float32)[:: max(1, len(train_df) // 1024)]
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
        if time.monotonic() - start >= stop_cfg.budget_sec:
            stopped = "time_budget"
            break
        if deadline_ts is not None and time.time() >= deadline_ts:
            stopped = "deadline_freeze"
            break

        start_idx = int(rng.integers(0, max_start + 1))
        state = env.reset(start_idx=start_idx, episode_max_steps=episode_steps)
        total_reward = 0.0
        force_random = episodes_done < warmup_episodes or collapse_streak > 0
        for step_i in range(episode_steps):
            action = agent.act(state)
            if force_random:
                action = int(rng.integers(0, 3))
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if global_step % 20 == 0:
                agent.train_step(global_step)
            global_step += 1
            state = next_state
            total_reward += reward
            if done:
                break

        episodes_done += 1

        if episodes_done % stop_cfg.eval_every_n_episodes == 0:
            metric = total_reward
            sample_actions = agent.infer_actions(diversity_states)
            unique_actions = int(np.unique(sample_actions).size)
            action_std = float(np.std(sample_actions.astype(np.float32)))
            logger.info(
                f"[TRADE-EVAL] job={job_id} variant={variant_id} ep={episodes_done} "
                f"reward={total_reward:.6f} metric={metric:.6f} "
                f"best={best_metric if best_metric > -float('inf') else float('nan'):.6f} "
                f"no_improve={no_improve} eps={agent.epsilon:.6f} "
                f"unique_actions={unique_actions} action_std={action_std:.6f}"
            )
            if unique_actions > 1 and action_std >= min_action_std and (
                unique_actions > best_div_unique or (unique_actions == best_div_unique and metric > best_div_metric)
            ):
                best_div_unique = unique_actions
                best_div_metric = metric
                best_div_path = ckpt.save(
                    agent.model,
                    "trade_dqn_best_diverse",
                    {"episode": episodes_done, "metric": metric, "unique_actions": unique_actions},
                )
                agent.save_replay(replay_path)
                agent.save_optimizer_state(optimizer_path)

            if unique_actions <= 1 or action_std < min_action_std:
                collapse_streak += 1
                no_improve = 0
                if collapse_streak >= 3:
                    agent.perturb_policy_head(stddev=1e-2)
                    agent.boost_exploration(epsilon_floor=0.5)
                    logger.warning(
                        f"[TRADE-RESCUE] job={job_id} ep={episodes_done} "
                        f"collapse_streak={collapse_streak} -> perturb_head+boost_eps({agent.epsilon:.3f})"
                    )
            else:
                collapse_streak = 0

            if (
                unique_actions >= min_unique_actions
                and action_std >= min_action_std
                and metric > (best_metric + stop_cfg.min_improvement_delta)
            ):
                best_metric = metric
                no_improve = 0
                best_path = ckpt.save(agent.model, "trade_dqn", {"episode": episodes_done, "metric": metric})
                agent.save_replay(replay_path)
                agent.save_optimizer_state(optimizer_path)
            else:
                no_improve += 1

            min_eps_for_stop = max(stop_cfg.min_episodes_before_early_stop, warmup_episodes)
            if episodes_done >= min_eps_for_stop and collapse_streak == 0 and no_improve >= stop_cfg.early_stop_patience:
                stopped = "early_stop"
                break
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
                            "best_div_unique": best_div_unique,
                            "best_div_metric": best_div_metric,
                        },
                        fh,
                        indent=2,
                    )
            except Exception:
                pass
    else:
        stopped = "episode_cap"

    path_to_load = best_path or best_div_path
    if path_to_load:
        try:
            ckpt_name = "trade_dqn" if best_path else "trade_dqn_best_diverse"
            ckpt.load(agent.model, ckpt_name)
            logger.info(f"[TRADE-INF] Loaded checkpoint for inference: {ckpt_name}")
        except Exception as e:
            logger.warning(f"[TRADE-INF] Could not load checkpoint ({path_to_load}): {e}")

    full_states = base_df[["ap_t"]].values.astype(np.float32)
    x1 = agent.infer_actions(full_states)
    x1_nunique = int(pd.Series(x1).nunique())
    x1_std = float(np.std(x1.astype(np.float32)))
    collapsed_output = x1_nunique <= 1 or x1_std <= 1e-12
    if collapsed_output:
        logger.warning(
            f"[TRADE-COLLAPSE] job={job_id} variant={variant_id} x1_nunique={x1_nunique} x1_std={x1_std:.12f}"
        )
        stopped = "collapsed_output"
    pd.DataFrame({"timestamp": base_df["timestamp"], "x1": x1}).to_csv(config.X1_OUTPUT_PATH, index=False)

    result = TrainResult(
        elapsed_sec=time.monotonic() - start,
        budget_sec=stop_cfg.budget_sec,
        stopped_reason=stopped,
        best_metric=float(best_metric if best_metric > -float("inf") else 0.0),
        checkpoint_path=best_path,
        episodes_completed=episodes_done,
        metrics={
            "train_reward": float(best_metric if best_metric > -float("inf") else 0.0),
            "x1_nunique": x1_nunique,
            "x1_std": x1_std,
            "collapsed_output": bool(collapsed_output),
        },
        job_id=job_id,
        variant_id=variant_id,
        started_at=started_at,
        finished_at=utcnow_iso(),
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
                    "x1_nunique": x1_nunique,
                    "x1_std": x1_std,
                    "collapsed_output": bool(collapsed_output),
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
    parser.add_argument("--base-dataset", default="data/processed/integrated_base.csv")
    parser.add_argument("--time-budget-min", type=int, default=180)
    parser.add_argument("--episode-cap", type=int, default=200)
    parser.add_argument("--eval-every-n-episodes", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-improvement-delta", type=float, default=1e-4)
    args = parser.parse_args()

    hp = HyperParameters()
    df = pd.read_csv(args.base_dataset, parse_dates=["timestamp"])
    stop_cfg = StopConfig(
        budget_sec=args.time_budget_min * 60.0,
        episode_cap=args.episode_cap,
        eval_every_n_episodes=args.eval_every_n_episodes,
        early_stop_patience=args.early_stop_patience,
        min_improvement_delta=args.min_improvement_delta,
        early_stop_metric="roi",
    )
    print(run(hp, df, stop_cfg))


if __name__ == "__main__":
    _cli()
