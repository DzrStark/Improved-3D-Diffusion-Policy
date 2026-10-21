#!/usr/bin/env python3
"""Check step-3 state transitions and updates using the real workspace loop."""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import dill
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.workspace.dp_workspace import DPWorkspace


def check_buffer_updates():
    source = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    source.register_buffer("float_state", torch.tensor([1.0]))
    source.register_buffer("int_state", torch.tensor(1, dtype=torch.int64))
    source.register_parameter("fixed", nn.Parameter(torch.tensor([1.0]), requires_grad=False))
    averaged = copy.deepcopy(source)
    ema = EMAModel(averaged, power=0.75)
    with torch.no_grad():
        for i, expected_decay in enumerate([0.0, 0.0, 1 - 2 ** -0.75]):
            previous_weight = averaged[0].weight.clone()
            for param in source.parameters():
                param.add_(0.25)
            source[1].running_mean.fill_(3.0 + i)
            source[1].running_var.fill_(4.0 + i)
            source[1].num_batches_tracked.fill_(5 + i)
            source.float_state.add_(2.0)
            source.int_state.add_(1)
            ema.step(source)
            assert ema.decay == expected_decay
            torch.testing.assert_close(averaged[0].weight,
                expected_decay * previous_weight + (1 - expected_decay) * source[0].weight)
            torch.testing.assert_close(averaged[1].weight, source[1].weight, rtol=0, atol=0)
            torch.testing.assert_close(averaged.fixed, source.fixed, rtol=0, atol=0)
            for name, buffer in source.named_buffers():
                torch.testing.assert_close(dict(averaged.named_buffers())[name], buffer, rtol=0, atol=0)
    assert not averaged.training and not any(p.requires_grad for p in averaged.parameters())
    print("EMA warmup, parameter exceptions, float/integer/BN buffers: PASS", flush=True)


def main():
    torch.set_num_threads(4)
    check_buffer_updates()
    output = PROJECT_ROOT / "data/g1_dex3_ema_check"
    output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "diffusion_policy_3d/config"),
                               version_base="1.1"):
        cfg = compose(config_name="dp_g1_dex3_rgb", overrides=[
            "training.device=cpu", "training.num_epochs=3", "training.max_train_steps=1",
            "training.lr_warmup_steps=0", "training.sample_every=1", "training.checkpoint_every=100",
            "dataloader.batch_size=2", "dataloader.num_workers=0", "dataloader.pin_memory=false",
            "val_dataloader.num_workers=0", "checkpoint.topk.k=0", "logging.mode=disabled"])
    cfg.task.dataset.zarr_path = str(PROJECT_ROOT / cfg.task.dataset.zarr_path)
    OmegaConf.resolve(cfg)
    workspace = DPWorkspace(cfg, output_dir=str(output))
    source, averaged = workspace.model, workspace.ema_model
    optimizer_ids = {id(p) for group in workspace.optimizer.param_groups for p in group["params"]}
    assert not optimizer_ids.intersection(id(p) for p in averaged.parameters())
    source_buffers = {name: value.clone() for name, value in source.named_buffers()}
    rgb = source.obs_encoder.key_model_map["image"].module.convnet
    tracked_params = {"r3m_conv": rgb.conv1.weight, "r3m_bn_affine": rgb.bn1.weight,
                      "unet_output": source.model.final_conv[-1].weight}
    initial_params = {name: value.detach().clone() for name, value in tracked_params.items()}
    losses, gradient_checks, decays, saves, sampling_shapes = [], [], [], [], []

    def assert_modes(training):
        assert source.training == training and source.model.training == training
        assert not any(m.training for rgb_model in source.obs_encoder.key_model_map.values()
                       for m in rgb_model.modules())
        assert not any(m.training for m in averaged.modules())
        assert not any(p.requires_grad or p.grad is not None for p in averaged.parameters())
        assert all(p.requires_grad for p in rgb.parameters())

    # Mode restoration also works when the configured policy is used without EMA.
    workspace.ema_model = None
    workspace._set_model_modes(training=False)
    assert not source.training
    workspace._set_model_modes(training=True)
    assert source.training and not rgb.training
    workspace.ema_model = averaged
    assert_modes(True)

    original_loss = source.compute_loss
    original_ema_step = EMAModel.step
    original_predict = averaged.predict_action
    original_save = workspace.save_checkpoint

    def observed_loss(batch):
        assert_modes(True)
        assert batch["action"].shape == (2, 16, 28)
        assert batch["obs"]["agent_pos"].shape == (2, 16, 28)
        loss = original_loss(batch)
        assert torch.isfinite(loss)
        losses.append(loss.item())
        for name, value in source.named_buffers():
            torch.testing.assert_close(value, source_buffers[name], rtol=0, atol=0)
        return loss

    before_step = {}

    def before_optimizer_step(optimizer, args, kwargs):
        count = 0
        for param in source.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                count += 1
        before_step.update({name: p.detach().clone() for name, p in tracked_params.items()})
        for param in tracked_params.values():
            assert param.grad is not None and torch.count_nonzero(param.grad) > 0
        gradient_checks.append(count)

    def after_optimizer_step(optimizer, args, kwargs):
        for name, param in tracked_params.items():
            assert not torch.equal(before_step[name], param)

    def observed_ema_step(ema, new_model):
        assert_modes(True)
        previous = averaged.model.final_conv[-1].weight.detach().clone()
        original_ema_step(ema, new_model)
        expected_decay = [0.0, 0.0, 1 - 2 ** -0.75][len(decays)]
        assert ema.decay == expected_decay
        torch.testing.assert_close(averaged.model.final_conv[-1].weight,
            expected_decay * previous + (1 - expected_decay) * source.model.final_conv[-1].weight)
        averaged_buffers = dict(averaged.named_buffers())
        for name, buffer in source.named_buffers():
            torch.testing.assert_close(averaged_buffers[name], buffer, rtol=0, atol=0)
        for name, param in source.normalizer.state_dict().items():
            torch.testing.assert_close(averaged.normalizer.state_dict()[name], param, rtol=0, atol=0)
        decays.append(ema.decay)

    def observed_predict(obs):
        assert_modes(False)
        before = {name: value.clone() for name, value in averaged.named_buffers()}
        # Random augmentations still run in eval, independently of BN mode.
        transform = averaged.obs_encoder.key_transform_map["image"]
        image = obs["image"][:, :2].permute(0, 1, 4, 2, 3).reshape(-1, 3, 224, 224) / 255.0
        first, second = transform(image), transform(image)
        assert not torch.equal(first, second)
        result = original_predict(obs)
        assert result["action_pred"].shape == (2, 16, 28)
        assert result["action"].shape == (2, 15, 28)
        assert torch.isfinite(result["action_pred"]).all()
        for name, buffer in averaged.named_buffers():
            torch.testing.assert_close(buffer, before[name], rtol=0, atol=0)
        sampling_shapes.append(list(result["action"].shape))
        return result

    def observed_save(*args, **kwargs):
        saves.append({"epoch": workspace.epoch, "global_step": workspace.global_step,
                      "use_thread": kwargs.get("use_thread", False)})
        return original_save(*args, **kwargs)

    # Run the actual workspace loop; only WandB's external I/O is replaced.
    pre_hook = workspace.optimizer.register_step_pre_hook(before_optimizer_step)
    post_hook = workspace.optimizer.register_step_post_hook(after_optimizer_step)
    local_run = MagicMock()
    with patch("diffusion_policy_3d.workspace.dp_workspace.wandb.init", return_value=local_run), \
         patch("diffusion_policy_3d.workspace.dp_workspace.wandb.config", MagicMock()), \
         patch.object(source, "compute_loss", new=observed_loss), \
         patch.object(EMAModel, "step", new=observed_ema_step), \
         patch.object(averaged, "predict_action", new=observed_predict), \
         patch.object(workspace, "save_checkpoint", new=observed_save):
        workspace.run()
    pre_hook.remove()
    post_hook.remove()
    assert len(losses) == len(gradient_checks) == len(decays) == len(sampling_shapes) == 3
    assert len(saves) == 2
    assert saves[0]["epoch"] == 0 and saves[1]["epoch"] == 3
    assert saves[1]["global_step"] == workspace.global_step
    assert not any(save["use_thread"] for save in saves)
    assert_modes(True)
    for name, param in tracked_params.items():
        assert not torch.equal(initial_params[name], param)
    assert local_run.finish.call_count == 1
    for call in local_run.log.call_args_list:
        assert "val_loss" not in call.args[0]

    checkpoint = workspace.get_checkpoint_path()
    payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill)
    assert dill.loads(payload["pickles"]["epoch"]) == workspace.epoch == 3
    assert dill.loads(payload["pickles"]["global_step"]) == workspace.global_step
    assert OmegaConf.to_container(payload["cfg"], resolve=True) == OmegaConf.to_container(cfg, resolve=True)
    for key, model in (("model", source), ("ema_model", averaged)):
        actual, saved = model.state_dict(), payload["state_dicts"][key]
        assert actual.keys() == saved.keys() and any(name.startswith("normalizer.") for name in saved)
        for name, value in actual.items():
            torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
    assert len(payload["state_dicts"]["optimizer"]["state"]) > 0
    assert all(int(state["step"]) == 3 for state in payload["state_dicts"]["optimizer"]["state"].values())
    result = {"passed": True, "device": "cpu", "epochs": 3, "batches_per_epoch": 1,
              "batch_size": 2, "losses": losses, "finite_gradient_parameter_counts": gradient_checks,
              "r3m_and_unet_parameters_updated": True, "r3m_bn_statistics_fixed": True,
              "ema_has_no_gradients_and_stays_eval": True, "ema_decays": decays,
              "ema_buffers_synchronized": True, "random_augmentation_active_in_eval": True,
              "sampled_action_shapes": sampling_shapes, "checkpoint_saves": saves,
              "final_checkpoint_matches_all_model_and_ema_state": True,
              "checkpoint": str(checkpoint), "validation_ran": False}
    (output / "check_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
