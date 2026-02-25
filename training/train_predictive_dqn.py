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
) -> dict:
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
        else:
            stopped = "episode_cap"

        x2_pred = model(x_all, training=False).numpy().reshape(-1)
        x2 = np.clip(x2_pred, -100.0, 100.0)
    else:
        while episodes_done < stop_cfg.episode_cap:
            if time.monotonic() - start >= stop_cfg.budget_sec:
                stopped = "time_budget"
                break
            if deadline_ts is not None and time.time() >= deadline_ts:
                stopped = "deadline_freeze"
                break

            state = env.reset()
            total_reward = 0.0

            for _ in range(min(len(train_df) - 2, config.MAX_STEPS_PER_EPISODE)):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_step(global_step)
                global_step += 1
                state = next_state
                total_reward += reward
                if done:
                    break

            episodes_done += 1

            if episodes_done % stop_cfg.eval_every_n_episodes == 0:
                metric = total_reward
                if metric > (best_metric + stop_cfg.min_improvement_delta):
                    best_metric = metric
                    no_improve = 0
                    best_path = ckpt.save(agent.model, ckpt_base, {"episode": episodes_done, "metric": metric})
                    agent.save_replay(replay_path)
                    agent.save_optimizer_state(optimizer_path)
                else:
                    no_improve += 1

                if no_improve >= stop_cfg.early_stop_patience:
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
                            },
                            fh,
                            indent=2,
                        )
                except Exception:
                    pass
        else:
            stopped = "episode_cap"

        full_states = base_df[["ap_t", "ts_t"]].values.astype(np.float32)
        action_idx = agent.infer_action_indices(full_states)
        x2 = config.PREDICTIVE_DQN_ACTION_MIN + action_idx * config.PREDICTIVE_DQN_ACTION_STEP
    out_csv = output_path or config.X2_OUTPUT_PATH
    pd.DataFrame({"timestamp": base_df["timestamp"], "x2": x2}).to_csv(out_csv, index=False)

    result = TrainResult(
        elapsed_sec=time.monotonic() - start,
        budget_sec=stop_cfg.budget_sec,
        stopped_reason=stopped,
        best_metric=float(best_metric if best_metric > -float("inf") else 0.0),
        checkpoint_path=best_path,
        episodes_completed=episodes_done,
        metrics={"train_reward": float(best_metric if best_metric > -float("inf") else 0.0), "output_path": out_csv},
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
