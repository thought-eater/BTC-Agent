import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from config.hyperparameters import HyperParameters
from models.predictive_dqn import PredictiveDQNAgent, PredictiveDQNEnvironment
from models.predictive_dqn.model import build_predictive_continuous_model
from utils.checkpoint import CheckpointManager
from utils.dqn_runtime import StopConfig, TrainResult, utcnow_iso
from utils.logger import Logger


def run(
    config: HyperParameters,
    base_df: pd.DataFrame,
    stop_cfg: StopConfig,
    seed: int = 42,
    job_id: str = "predictive",
    variant_id: str = "paper",
    deadline_ts: float = None,
    resume_from_checkpoint: str = None,
    output_path: str = None,
    episode_max_steps: int | None = None,
) -> dict:
    logger = Logger("PredictiveDQNTrain", log_file=f"{config.LOGS_DIR}/training.log")
    train_df = base_df[base_df["split"] == "train"].copy()

    env = PredictiveDQNEnvironment(
        prices=train_df["ap_t"].values,
        sentiments=train_df["ts_t"].values,
        action_min=config.PREDICTIVE_DQN_ACTION_MIN,
        action_step=config.PREDICTIVE_DQN_ACTION_STEP,
    )

    agent = PredictiveDQNAgent(
        lr=config.LEARNING_RATE,
        gamma=config.DISCOUNT_FACTOR,
        epsilon_start=config.EPSILON_START,
        epsilon_decay=config.EPSILON_DECAY,
        epsilon_min=config.EPSILON_MIN,
        replay_size=config.REPLAY_BUFFER_SIZE,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        action_size=config.PREDICTIVE_DQN_OUTPUT_DIM,
    )
    state_train = train_df[["ap_t", "ts_t"]].values.astype(np.float32)
    agent.set_state_normalization(state_train.mean(axis=0), state_train.std(axis=0))

    ckpt = CheckpointManager(config.CHECKPOINTS_DIR)
    ckpt_base = "predictive_dqn" if variant_id == "paper" else f"predictive_dqn_{variant_id}"
    progress_path = f"{config.CHECKPOINTS_DIR}/{ckpt_base}_progress.json"
    replay_path = f"{config.CHECKPOINTS_DIR}/predictive_dqn_replay.pkl"
    optimizer_path = f"{config.CHECKPOINTS_DIR}/predictive_dqn_optimizer.npy"
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

    if variant_id == "continuous":
        # Continuous regression improvement: predict next-hour % return directly.
        model = build_predictive_continuous_model(input_dim=2)
        cont_weights_path = f"{config.CHECKPOINTS_DIR}/{ckpt_base}.weights.h5"
        if resume_from_checkpoint and Path(cont_weights_path).exists():
            try:
                model.load_weights(cont_weights_path)
            except Exception:
                pass
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        loss_fn = tf.keras.losses.MeanSquaredError()
        x_all = base_df[["ap_t", "ts_t"]].values.astype(np.float32)
        train_x = train_df[["ap_t", "ts_t"]].values.astype(np.float32)
        ap = train_df["ap_t"].values.astype(np.float32)
        y = ((np.roll(ap, -1) - ap) / np.maximum(np.abs(ap), 1e-8)) * 100.0
        y = np.clip(y, -100.0, 100.0)
        y = y.reshape(-1, 1)
        train_x = train_x[:-1]
        y = y[:-1]

        best_loss = float("inf")
        while episodes_done < stop_cfg.episode_cap:
            if time.monotonic() - start >= stop_cfg.budget_sec:
                stopped = "time_budget"
                break
            if deadline_ts is not None and time.time() >= deadline_ts:
                stopped = "deadline_freeze"
                break

            with tf.GradientTape() as tape:
                pred = model(train_x, training=True)
                loss = loss_fn(y, pred)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            episodes_done += 1
            cur_loss = float(loss.numpy())
            metric = -cur_loss

            if metric > (best_metric + stop_cfg.min_improvement_delta):
                best_metric = metric
                no_improve = 0
                best_loss = cur_loss
                # Persist using keras save_weights for compatibility.
                model.save_weights(cont_weights_path)
                with open(f"{config.CHECKPOINTS_DIR}/{ckpt_base}_metadata.json", "w", encoding="utf-8") as fh:
                    json.dump({"episode": episodes_done, "loss": cur_loss, "variant_id": variant_id}, fh, indent=2)
                best_path = cont_weights_path
            else:
                no_improve += 1

            if no_improve >= stop_cfg.early_stop_patience:
                stopped = "early_stop"
                break
            logger.info(
                f"[PRED-EVAL] job={job_id} variant={variant_id} ep={episodes_done} "
                f"metric={metric:.6f} best={best_metric:.6f} no_improve={no_improve}"
            )
        else:
            stopped = "episode_cap"

        if best_path and Path(best_path).exists():
            try:
                model.load_weights(best_path)
                logger.info(f"[PRED-INF] Loaded best continuous weights for inference: {best_path}")
            except Exception as e:
                logger.warning(f"[PRED-INF] Could not load best continuous weights ({best_path}): {e}")
        x2_pred = model(x_all, training=False).numpy().reshape(-1)
        x2 = np.clip(x2_pred, -100.0, 100.0)
    else:
        warmup_episodes = max(10, min(40, int(0.35 * max(1, stop_cfg.episode_cap))))
        # 20,001-way action space is very prone to single-action collapse with short budgets.
        # Use a realistic diversity floor for deadline-constrained training.
        min_unique_actions = max(2, min(12, int(0.002 * config.PREDICTIVE_DQN_OUTPUT_DIM)))
        min_action_std = max(config.PREDICTIVE_DQN_ACTION_STEP * 2.0, 0.2)
        diversity_states = train_df[["ap_t", "ts_t"]].values.astype(np.float32)[:: max(1, len(train_df) // 1024)]
        best_div_unique = -1
        best_div_metric = -float("inf")
        best_div_path = None
        collapse_streak = 0
        rng = np.random.default_rng(int(seed))
        max_steps_cfg = int(episode_max_steps) if episode_max_steps is not None else int(config.MAX_STEPS_PER_EPISODE)
        episode_steps = int(min(len(train_df) - 2, max(256, max_steps_cfg)))
        max_start = max(1, len(train_df) - episode_steps - 2)

        while episodes_done < stop_cfg.episode_cap:
            if time.monotonic() - start >= stop_cfg.budget_sec:
                stopped = "time_budget"
                break
            if deadline_ts is not None and time.time() >= deadline_ts:
                stopped = "deadline_freeze"
                break

            start_idx = int(rng.integers(1, max_start + 1))
            state = env.reset(start_idx=start_idx, episode_max_steps=episode_steps)
            total_reward = 0.0
            # If policy collapsed in the previous eval, force one more random episode to recover diversity.
            force_random = episodes_done < warmup_episodes or collapse_streak > 0

            for _ in range(episode_steps):
                action = agent.act(state, force_random=force_random)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_step(global_step)
                global_step += 1
                state = next_state
                total_reward += reward
                if done:
                    break

            episodes_done += 1
            eps_now = agent.decay_epsilon()

            if episodes_done % stop_cfg.eval_every_n_episodes == 0:
                metric = total_reward
                sample_actions = agent.infer_action_indices(diversity_states)
                unique_actions = int(np.unique(sample_actions).size)
                sample_values = config.PREDICTIVE_DQN_ACTION_MIN + sample_actions * config.PREDICTIVE_DQN_ACTION_STEP
                action_std = float(np.std(sample_values))
                logger.info(
                    f"[PRED-EVAL] job={job_id} variant={variant_id} ep={episodes_done} "
                    f"reward={total_reward:.6f} metric={metric:.6f} "
                    f"best={best_metric if best_metric > -float('inf') else float('nan'):.6f} "
                    f"no_improve={no_improve} eps={eps_now:.6f} "
                    f"unique_actions={unique_actions} action_std={action_std:.6f}"
                )
                if unique_actions > 1 and action_std >= min_action_std and (
                    unique_actions > best_div_unique
                    or (unique_actions == best_div_unique and metric > best_div_metric)
                ):
                    best_div_unique = unique_actions
                    best_div_metric = metric
                    best_div_path = ckpt.save(
                        agent.model,
                        f"{ckpt_base}_best_diverse",
                        {"episode": episodes_done, "metric": metric, "unique_actions": unique_actions},
                    )
                    agent.save_replay(replay_path)
                    agent.save_optimizer_state(optimizer_path)

                if unique_actions <= 1 or action_std < min_action_std:
                    collapse_streak += 1
                    # Do not count collapsed evaluations against patience.
                    no_improve = 0
                    if collapse_streak >= 3:
                        agent.perturb_policy_head(stddev=5e-3)
                        agent.boost_exploration(epsilon_floor=0.6)
                        logger.warning(
                            f"[PRED-RESCUE] job={job_id} variant={variant_id} ep={episodes_done} "
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
                    best_path = ckpt.save(
                        agent.model,
                        ckpt_base,
                        {"episode": episodes_done, "metric": metric, "unique_actions": unique_actions},
                    )
                    agent.save_replay(replay_path)
                    agent.save_optimizer_state(optimizer_path)
                else:
                    no_improve += 1

                # If diversity is already healthy, reduce exploration faster to improve reward stability.
                if unique_actions >= min_unique_actions and action_std >= min_action_std and agent.epsilon > 0.25:
                    prev_eps = float(agent.epsilon)
                    agent.epsilon = max(0.25, float(agent.epsilon) * 0.90)
                    logger.info(
                        f"[PRED-EPS] job={job_id} variant={variant_id} ep={episodes_done} "
                        f"epsilon_adjust {prev_eps:.6f}->{agent.epsilon:.6f}"
                    )

                min_eps_for_stop = max(stop_cfg.min_episodes_before_early_stop, warmup_episodes)
                if (
                    episodes_done >= min_eps_for_stop
                    and collapse_streak == 0
                    and no_improve >= stop_cfg.early_stop_patience
                ):
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
        if path_to_load and Path(path_to_load).exists():
            try:
                ckpt_name = ckpt_base if best_path else f"{ckpt_base}_best_diverse"
                ckpt.load(agent.model, ckpt_name)
                logger.info(f"[PRED-INF] Loaded discrete checkpoint for inference: {ckpt_name}")
            except Exception as e:
                logger.warning(f"[PRED-INF] Could not load discrete checkpoint ({path_to_load}): {e}")
        full_states = base_df[["ap_t", "ts_t"]].values.astype(np.float32)
        action_idx = agent.infer_action_indices(full_states)
        x2 = config.PREDICTIVE_DQN_ACTION_MIN + action_idx * config.PREDICTIVE_DQN_ACTION_STEP

    x2_nunique = int(pd.Series(x2).nunique())
    x2_std = float(np.std(x2))
    collapsed_output = x2_nunique <= 1 or x2_std <= 1e-12
    if collapsed_output:
        logger.warning(
            f"[PRED-COLLAPSE] job={job_id} variant={variant_id} "
            f"x2_nunique={x2_nunique} x2_std={x2_std:.12f}"
        )
        stopped = "collapsed_output"

    out_csv = output_path or config.X2_OUTPUT_PATH
    pd.DataFrame({"timestamp": base_df["timestamp"], "x2": x2}).to_csv(out_csv, index=False)

    result = TrainResult(
        elapsed_sec=time.monotonic() - start,
        budget_sec=stop_cfg.budget_sec,
        stopped_reason=stopped,
        best_metric=float(best_metric if best_metric > -float("inf") else 0.0),
        checkpoint_path=best_path,
        episodes_completed=episodes_done,
        metrics={
            "train_reward": float(best_metric if best_metric > -float("inf") else 0.0),
            "output_path": out_csv,
            "x2_nunique": x2_nunique,
            "x2_std": x2_std,
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
                    "epsilon": float(getattr(agent, "epsilon", 0.0)),
                    "stopped_reason": "completed",
                    "x2_nunique": x2_nunique,
                    "x2_std": x2_std,
                    "collapsed_output": bool(collapsed_output),
                },
                fh,
                indent=2,
            )
        if variant_id != "continuous":
            agent.save_replay(replay_path)
            agent.save_optimizer_state(optimizer_path)
    except Exception:
        pass
    return result.__dict__


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", default="data/processed/integrated_base.csv")
    parser.add_argument("--time-budget-min", type=int, default=240)
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
