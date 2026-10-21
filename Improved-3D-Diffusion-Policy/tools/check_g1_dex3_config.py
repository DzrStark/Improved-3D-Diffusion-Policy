#!/usr/bin/env python3
"""Check step-2 config, all episode windows and the actual 28-D policy."""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    torch.set_num_threads(4)
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "diffusion_policy_3d/config"),
                               version_base="1.1"):
        cfg = compose(config_name="dp_g1_dex3_rgb", return_hydra_config=True)
        base = compose(config_name="dp_224x224_r3m",
                       overrides=["task=g1_dex3_pick_right-image"])
    for key in ("policy", "optimizer", "ema", "dataloader"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(base[key], resolve=True), key
    for key in base.training:
        if key != "resume":
            assert cfg.training[key] == base.training[key], key
    assert (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps) == (16, 2, 16)
    assert cfg.logging.mode == "online" and not cfg.logging.resume
    assert not cfg.training.resume and cfg.checkpoint.save_ckpt and cfg.checkpoint.save_last_ckpt
    assert not cfg.hydra.job.chdir
    assert cfg.hydra.run.dir.startswith("data/outputs/g1_dex3_rgb/")
    assert cfg.task.dataset.val_ratio == 0.0 and cfg.task.dataset.max_train_episodes is None
    assert list(cfg.shape_meta.obs.image.shape) == [3, 224, 224]
    assert list(cfg.shape_meta.obs.agent_pos.shape) == list(cfg.shape_meta.action.shape) == [28]
    print("Hydra composition and inherited settings: PASS", flush=True)

    # Absolute path only for this standalone check; the training config stays relative.
    dataset = hydra.utils.instantiate(cfg.task.dataset,
        zarr_path=str(PROJECT_ROOT / cfg.task.dataset.zarr_path))
    assert dataset.replay_buffer.n_episodes == 60 and dataset.replay_buffer.n_steps == 10344
    assert dataset.train_mask.all() and int(dataset.train_mask.sum()) == 60
    assert len(dataset.get_validation_dataset()) == 0
    assert dataset.pad_before == 1 and dataset.pad_after == 15
    ends = dataset.replay_buffer.episode_ends
    starts = np.r_[0, ends[:-1]]
    indices = dataset.sampler.indices
    buffer_starts, buffer_ends, sample_starts, sample_ends = indices.T
    episode_ids = np.searchsorted(ends, buffer_starts, side="right")
    assert np.all(buffer_starts >= starts[episode_ids])
    assert np.all(buffer_ends <= ends[episode_ids])
    assert np.all(buffer_ends > buffer_starts)
    assert np.all(buffer_ends - buffer_starts == sample_ends - sample_starts)
    assert np.all(sample_starts >= 0) and np.all(sample_ends <= cfg.horizon)
    # Original rule yields episode_length + 1 windows with padding 1/15 and horizon 16.
    np.testing.assert_array_equal(np.bincount(episode_ids, minlength=60), ends - starts + 1)
    sample_count = 0
    for episode_id in range(60):
        candidates = np.flatnonzero(episode_ids == episode_id)
        for sample_id in (candidates[0], candidates[len(candidates) // 2], candidates[-1]):
            sample = dataset[int(sample_id)]
            assert sample["obs"]["image"].shape == (16, 224, 224, 3)
            assert sample["obs"]["agent_pos"].shape == sample["action"].shape == (16, 28)
            for tensor in (sample["obs"]["image"], sample["obs"]["agent_pos"], sample["action"]):
                assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            bs, be, ss, se = indices[sample_id]
            source_rows = np.clip(np.arange(16) - ss + bs, bs, be - 1)
            for key, tensor in (("state", sample["obs"]["agent_pos"]), ("action", sample["action"]),
                                ("img", sample["obs"]["image"])):
                np.testing.assert_array_equal(tensor.numpy(), dataset.replay_buffer[key][source_rows])
            sample_count += 1
    normalizer = dataset.get_normalizer()
    actual = torch.from_numpy(dataset.replay_buffer["action"])
    torch.testing.assert_close(normalizer["action"].get_input_stats()["min"], actual.amin(dim=0))
    torch.testing.assert_close(normalizer["action"].get_input_stats()["max"], actual.amax(dim=0))
    torch.testing.assert_close(normalizer["agent_pos"].normalize(sample["obs"]["agent_pos"]),
                               sample["obs"]["agent_pos"])
    print(f"All 60 episodes, {len(dataset)} windows, {sample_count} sampled sequences: PASS", flush=True)

    torch.manual_seed(cfg.training.seed)
    policy = hydra.utils.instantiate(cfg.policy).cpu()
    policy.set_normalizer(normalizer)
    assert policy.action_dim == policy.mask_generator.action_dim == 28
    assert policy.model.down_modules[0][0].blocks[0].block[0].in_channels == 28
    assert policy.model.final_conv[-1].out_channels == 28
    assert (policy.horizon, policy.n_obs_steps, policy.n_action_steps) == (16, 2, 16)
    assert policy.obs_encoder.key_shape_map["agent_pos"] == (28,)
    assert policy.obs_feature_dim == 2 * (512 + 28)
    result = {"passed": True, "n_episodes": 60, "n_frames": 10344,
              "n_windows": len(dataset), "sampled_sequences": sample_count,
              "all_windows_within_episode": True, "validation_sequences": 0,
              "policy_action_dim": policy.action_dim, "observation_feature_dim": int(policy.obs_feature_dim),
              "logging_mode": cfg.logging.mode, "resume": cfg.training.resume,
              "checkpoint_enabled": cfg.checkpoint.save_ckpt}
    output = PROJECT_ROOT / "data/g1_dex3_config_check"
    output.mkdir(parents=True, exist_ok=True)
    job_cfg = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
    OmegaConf.save(job_cfg, str(output / "resolved_config.yaml"), resolve=True)
    (output / "check_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
