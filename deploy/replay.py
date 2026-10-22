"""Offline real-data replay and target recording; never opens camera or DDS."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", help="cpu or cuda:N, no fallback")
    parser.add_argument("--source", choices=("zarr", "raw"))
    parser.add_argument("--episodes", default="0", help="Zero-based comma-separated episode indices")
    parser.add_argument("--max-frames", type=int, default=32, help="Per episode; 0 means entire episode")
    parser.add_argument("--execute-steps", type=int, help="Replan after 1..15 steps")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("max-frames must be nonnegative")
    # Heavy dependencies are imported only after --help exits.
    import numpy as np
    from deploy.g1dp.adapters import RecordingRobot
    from deploy.g1dp.config import load_config, seed_once
    from deploy.g1dp.control import ActionQueue
    from deploy.g1dp.observations import GROUPS, History, JOINT_NAMES
    from deploy.g1dp.policy import Policy
    from deploy.g1dp.sources import make_source
    cfg = load_config(args.config)
    for key in ("device", "source", "execute_steps"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    episodes = [int(x) for x in args.episodes.split(",")]
    source = make_source(cfg)
    if not episodes or any(e < 0 or e >= source.n_episodes for e in episodes):
        parser.error("Episode index outside source")
    args.output.mkdir(parents=True, exist_ok=False)
    policy = Policy(cfg["checkpoint"], cfg["training_root"], cfg["device"], cfg["threads"])
    from omegaconf import OmegaConf
    (args.output / "checkpoint_config.yaml").write_text(OmegaConf.to_yaml(policy.cfg, resolve=False))
    (args.output / "policy_config.json").write_text(json.dumps(policy.policy_config, indent=2)+"\n")
    seed_once(cfg["seed"])  # Once per run; not once per inference.
    history, sink = History(), RecordingRobot()
    queue = ActionQueue(cfg["execute_steps"], cfg["target_hz"], cfg["max_lateness_s"])
    errors, latencies, jumps = [], [], []
    predictions = 0
    try:
        with (args.output / "targets.jsonl").open("x") as stream:
            for episode in episodes:
                queue.pause()
                queue.start()
                history.reset()
                previous = None
                limit = source.length(episode)
                if args.max_frames:
                    limit = min(limit, args.max_frames)
                for index in range(limit):
                    frame, demonstration = source.read(episode, index)
                    arrays = history.push(frame)
                    sink.update_measurement(frame.q, frame.timestamp_s)
                    row = {"episode": frame.episode, "index": index, "timestamp_s": frame.timestamp_s,
                           "wall_timestamp_ns": frame.wall_timestamp_ns, "timestamp_kind": frame.timestamp_kind,
                           "measured_state": frame.q.tolist(), "demonstration_action": demonstration.tolist(),
                           "history_indices": [f.index for f in history.frames],
                           "history_timestamps_s": [f.timestamp_s for f in history.frames],
                           "actual_sent_target": None, "mode": "offline_recording_exogenous_feedback"}
                    if not queue.queue:
                        prediction, latency = policy.predict(arrays)
                        predictions += 1
                        latencies.append(latency)
                        queue.replace(prediction["action"], frame.timestamp_s)
                        row.update(predicted_action=prediction["action"].tolist(),
                                   predicted_action_full=prediction["action_pred"].tolist(), inference_s=latency)
                    item = queue.pop(frame.timestamp_s)
                    if item:
                        due, action_index, target = item
                        sink.send(target, frame.timestamp_s)
                        row.update(recorded_target=target.tolist(), action_index=action_index,
                                   due_timestamp_s=due, lateness_s=frame.timestamp_s-due)
                        errors.append(target - demonstration)
                        if previous is not None:
                            jumps.append(float(np.max(np.abs(target - previous))))
                        previous = target
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
    finally:
        queue.stop()
        sink.close()
    error = np.stack(errors)
    summary = {"command": [sys.executable, "-m", "deploy.replay"] + sys.argv[1:],
               "cwd": os.getcwd(), "config": cfg, "checkpoint": policy.metadata, "episodes": episodes,
               "max_frames_per_episode": args.max_frames, "seed": cfg["seed"],
               "predictions": predictions, "recorded_targets": len(errors), "expired_slots": queue.dropped,
               "mae_rad": float(np.abs(error).mean()), "rmse_rad": float(np.sqrt(np.mean(error ** 2))),
               "group_mae_rad": {g: float(np.abs(error[:, 7*i:7*i+7]).mean()) for i,g in enumerate(GROUPS)},
               "max_consecutive_target_jump_rad": max(jumps) if jumps else None,
               "latency_s": {"first": latencies[0], "mean": float(np.mean(latencies)),
                             "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
                             "max": max(latencies)}, "joint_names_source_candidate": JOINT_NAMES,
               "time_semantics": "virtual recorded timestamps; inference cost excluded; no wall-clock pacing",
               "limits": "raw predictions recorded; no physical targets sent or hardware limits assumed",
               "scope": "training demonstrations; not a robot simulation, closed loop, or generalization test"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({k: summary[k] for k in ("predictions", "recorded_targets", "mae_rad", "latency_s")}))


if __name__ == "__main__":
    main()
