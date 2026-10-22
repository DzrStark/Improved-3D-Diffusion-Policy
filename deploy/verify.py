"""Verify formal EMA, original inference, real preprocessing and dataset boundaries."""
import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    args = parser.parse_args()
    from deploy.g1dp.config import load_config, seed_once
    cfg = load_config(args.config)
    if cfg["device"] == "cpu":
        # Only the original reference path uses auto-detecting DataParallel.
        # Hide CUDA before imports, as in the training repository's CPU checks.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import copy
    import gc
    import random
    from importlib.metadata import version, PackageNotFoundError
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import xml.etree.ElementTree as ET
    from deploy.g1dp.observations import GROUPS, History, JOINT_NAMES
    from deploy.g1dp.policy import Policy, sha256, strict_restore
    from deploy.g1dp.sources import RawSource, ZarrSource

    def installed_version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    args.output.mkdir(parents=True, exist_ok=False)
    policy = Policy(cfg["checkpoint"], cfg["training_root"], cfg["device"], cfg["threads"])
    seed_once(cfg["seed"])
    source = ZarrSource(cfg["zarr"], cfg["raw_root"])
    raw = RawSource(cfg["raw_root"])
    report = {"command": [sys.executable, "-m", "deploy.verify"] + sys.argv[1:],
              "cwd": os.getcwd(), "config": cfg, "checkpoint": policy.metadata,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "cuda_available": torch.cuda.is_available(), "cuda_count": torch.cuda.device_count(),
                              "versions": {name: installed_version(name) for name in (
                                  "torch", "torchvision", "numpy", "opencv-python", "zarr", "numcodecs", "hydra-core",
                                  "omegaconf", "diffusers", "timm", "dill", "r3m", "einops", "transformers")}}}
    (args.output / "checkpoint_config.yaml").write_text(OmegaConf.to_yaml(policy.cfg, resolve=False))
    (args.output / "policy_config.json").write_text(json.dumps(policy.policy_config, indent=2))
    # Directly invoke the actual converter, independently of deploy preprocessing.
    converter_path = Path(cfg["training_root"]).parent / "tsingtec_raw_data/tools/convert_unitree_to_zarr.py"
    spec = importlib.util.spec_from_file_location("original_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    preprocessed = 0
    info_coordinates, empty_names, timestamp_gaps = {}, 0, []
    action_stats, state_stats = [], []
    for e in range(source.n_episodes):
        rows = raw.frames(e)
        data = json.loads(raw.paths[e].read_text())
        coordinate = json.dumps(data["info"].get("hand_coordinates"), sort_keys=True)
        info_coordinates[coordinate] = info_coordinates.get(coordinate, 0) + 1
        empty_names += int(all(not v for v in data["info"]["joint_names"].values()))
        stamps = np.array([row["record_timestamp_mono_ns"] for row in rows], dtype=np.int64)
        if not np.all(np.diff(stamps) > 0):
            raise AssertionError("Nonmonotonic recorded timestamps")
        timestamp_gaps.extend({"episode": raw.paths[e].parent.name, "index": int(i+1), "dt_s": float(np.diff(stamps)[i]/1e9)}
                              for i in np.flatnonzero(np.diff(stamps) > 100_000_000))
        a = np.stack([converter.flatten_qpos(row, "actions", str(raw.paths[e])) for row in rows])
        s = np.stack([converter.flatten_qpos(row, "states", str(raw.paths[e])) for row in rows])
        start, end = int(source.starts[e]), int(source.ends[e])
        np.testing.assert_array_equal(a, source.store["data/action"][start:end])
        np.testing.assert_array_equal(s, source.store["data/state"][start:end])
        action_stats.append(a); state_stats.append(s)
        for i in sorted(set([0, len(rows)//2, len(rows)-1])):
            zf, za = source.read(e, i)
            rf, ra = raw.read(e, i)
            expected = converter.load_rgb(raw.paths[e].parent / rows[i]["colors"]["color_1"])
            np.testing.assert_array_equal(zf.image, expected)
            np.testing.assert_array_equal(rf.image, expected)
            np.testing.assert_array_equal(zf.q, rf.q)
            np.testing.assert_array_equal(za, ra)
            preprocessed += 1
    actions, states = np.concatenate(action_stats), np.concatenate(state_stats)
    report["data_checks"] = {"episodes": source.n_episodes, "frames": len(actions),
                             "all_raw_state_action_exact": True, "exact_rgb_samples": preprocessed,
                             "empty_joint_names_episodes": empty_names, "hand_coordinate_metadata": info_coordinates,
                             "timestamp_gaps_over_100ms": timestamp_gaps}
    print("Real data and preprocessing checked", flush=True)

    # Original sampler + Dataset conversion, using two actual episodes in a small
    # in-memory ReplayBuffer, so no all-image eager allocation is necessary.
    from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
    from diffusion_policy_3d.common.sampler import SequenceSampler, create_indices
    from diffusion_policy_3d.dataset.gr1_dex_dataset_image import GR1DexDatasetImage
    indices = create_indices(episode_ends=source.ends, sequence_length=16,
                             episode_mask=np.ones(source.n_episodes, dtype=bool), pad_before=1, pad_after=15)
    for bs, be, ss, se in indices:
        e = int(np.searchsorted(source.ends, bs, side="right"))
        assert source.starts[e] <= bs < be <= source.ends[e] and be-bs == se-ss
    replay = ReplayBuffer.create_empty_numpy()
    for e in (0, 1):
        start, end = int(source.starts[e]), int(source.ends[e])
        replay.add_episode({k: np.asarray(source.store["data/"+k][start:end]) for k in ("img", "state", "action")})
    sampler = SequenceSampler(replay, sequence_length=16, pad_before=1, pad_after=15)
    dataset = GR1DexDatasetImage.__new__(GR1DexDatasetImage)
    dataset.use_img, dataset.use_depth = True, False
    for e in (0, 1):
        history = History()
        local = np.flatnonzero((sampler.indices[:,0] >= replay.episode_ends[e]-source.length(e)) &
                               (sampler.indices[:,0] < replay.episode_ends[e]))
        for index in (0, 1, source.length(e)-1):
            history.reset()
            for i in range(max(0,index-1), index+1):
                frame, _ = source.read(e, i)
                arrays = history.push(frame)
            # Window starting at latest index - 1; first sample is left padded.
            sample = dataset._sample_to_data(sampler.sample_sequence(int(local[index])))
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], sample["obs"][key][:2])
    report["sampler"] = {"all_windows_within_episode": int(len(indices)),
                         "startup_padding": "[first,first] matches pad_before=1",
                         "edge_history_checks": 6, "order": "oldest_to_newest"}
    del replay, sampler
    gc.collect()

    # The native pretrained loading path is an independent reference. Require
    # existing cache files before constructing it, so this check cannot download.
    cache = Path.home() / ".r3m/r3m_18"
    if not all((cache/name).is_file() for name in ("model.pt", "config.yaml")):
        raise FileNotFoundError("Original reference comparison requires existing R3M cache: " + str(cache))
    reference = hydra.utils.instantiate(OmegaConf.create(policy.policy_config))
    reference.load_state_dict(policy.model.state_dict(), strict=True)
    reference.to(policy.device).eval().requires_grad_(False)
    reference.obs_encoder.key_model_map["image"].module.device = str(policy.device)
    for k, v in reference.state_dict().items():
        torch.testing.assert_close(v, policy.model.state_dict()[k], rtol=0, atol=0)
    history = History()
    for i in (39, 40):
        f, _ = source.read(0, i)
        arrays = history.push(f)
    seed_once(cfg["seed"])
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    actual, latency = policy.predict(arrays)
    random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(torch_state)
    if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
    with torch.inference_mode():
        expected = reference.predict_action(policy.tensors(arrays))
    diffs = {}
    for key in actual:
        expected_np = expected[key].cpu().numpy()[0]
        np.testing.assert_allclose(actual[key], expected_np, atol=1e-6, rtol=0)
        diffs[key] = float(np.max(np.abs(actual[key]-expected_np)))
    # Same input on a continuing RNG stream should change augmentation/sampling.
    repeated, _ = policy.predict(arrays)
    variation = float(np.max(np.abs(repeated["action"] - actual["action"])))
    assert variation > 0
    report["original_policy_comparison"] = {"max_abs_diff": diffs, "action_shape": list(actual["action"].shape),
                                            "full_shape": list(actual["action_pred"].shape),
                                            "continuing_rng_max_action_change": variation,
                                            "reference_r3m_cache_sha256": sha256(cache/"model.pt"),
                                            "single_inference_s": latency}
    demonstrated = np.stack([source.read(0, i)[1] for i in range(40,55)])
    error = actual["action"] - demonstrated
    report["aligned_15_step_error"] = {"episode_index": 0, "newest_observation_index": 40,
                                       "demonstration_indices": list(range(40,55)),
                                       "mae_rad": float(np.abs(error).mean()),
                                       "rmse_rad": float(np.sqrt(np.mean(error**2))),
                                       "group_mae_rad": {g: float(np.abs(error[:,7*i:7*i+7]).mean()) for i,g in enumerate(GROUPS)},
                                       "max_adjacent_prediction_jump_rad": float(np.abs(np.diff(actual["action"], axis=0)).max())}
    report["normalizer"] = {k: v.cpu().tolist() for k,v in policy.model.normalizer.state_dict().items()}
    np.testing.assert_array_equal(policy.model.normalizer["action"].get_input_stats()["min"].cpu().numpy(), actions.min(axis=0))
    np.testing.assert_array_equal(policy.model.normalizer["action"].get_input_stats()["max"].cpu().numpy(), actions.max(axis=0))
    del reference, expected
    gc.collect()
    print("Original pretrained policy comparison passed", flush=True)

    # Verify rejection paths against real checkpoint tensor schemas.
    state = policy.model.state_dict()
    malformed = dict(state)
    del malformed["normalizer.params_dict.action.scale"]
    try: strict_restore(policy.model, malformed)
    except ValueError: pass
    else: raise AssertionError("Missing normalizer was accepted")
    malformed = dict(state)
    malformed["model.diffusion_step_encoder.1.weight"] = torch.zeros(1)
    try: strict_restore(policy.model, malformed)
    except (ValueError, RuntimeError): pass
    else: raise AssertionError("Unexpected weight was accepted")
    report["invalid_weights_rejected"] = True

    # Compare the three distinct limit notions without silently clipping the data.
    root = ET.parse(args.urdf).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    limits = [joints[n].find("limit") for n in JOINT_NAMES]
    lower = np.array([float(j.get("lower")) for j in limits])
    upper = np.array([float(j.get("upper")) for j in limits])
    velocity = np.array([float(j.get("velocity")) for j in limits])
    limits_report = {"joint_names_candidate": JOINT_NAMES, "arm_dds_indices_candidate": list(range(15,29)),
                     "urdf": str(args.urdf.resolve()), "urdf_sha256": sha256(args.urdf),
                     "urdf_lower_rad": lower.tolist(), "urdf_upper_rad": upper.tolist(),
                     "urdf_velocity_rad_s": velocity.tolist(), "hardware_verified": False,
                     "hardware_lower_rad": None, "hardware_upper_rad": None, "hardware_velocity_rad_s": None,
                     "dataset_action_min_rad": actions.min(axis=0).tolist(), "dataset_action_max_rad": actions.max(axis=0).tolist(),
                     "dataset_state_min_rad": states.min(axis=0).tolist(), "dataset_state_max_rad": states.max(axis=0).tolist(),
                     "action_urdf_violation_count_per_joint": ((actions<lower)|(actions>upper)).sum(axis=0).tolist(),
                     "state_urdf_violation_count_per_joint": ((states<lower)|(states>upper)).sum(axis=0).tolist(),
                     "note": "Source candidate order, not verified hardware equivalence. URDF is reference only; no auto limit selection."}
    (args.output / "limits_comparison.json").write_text(json.dumps(limits_report, indent=2)+"\n")
    report["limits"] = {"action_violations": int(((actions<lower)|(actions>upper)).sum()),
                        "state_violations": int(((states<lower)|(states>upper)).sum()),
                        "hardware_verified": False}
    report["passed"] = True
    report["scope"] = "Formal training checkpoint, training demonstrations, CPU offline. No hardware, dynamics, closed loop, generalization, or image-use claim."
    (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({k: report[k] for k in ("passed", "sampler", "original_policy_comparison", "limits")}))


if __name__ == "__main__":
    main()
