#!/usr/bin/env python3
"""Check real data, 16-step training, small-batch fitting and checkpoint reload."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zarr', type=Path, default=PROJECT_ROOT.parent / 'tsingtec_training_data/right_pick')
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0; no automatic fallback')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overfit-steps', type=int, default=1000, help='Fresh-noise updates on 2 sequences; 0 skips fitting')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.overfit_steps < 0:
        parser.error('--overfit-steps must be >= 0')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cpu':
        # Select CPU before importing Torch/R3M, whose DataParallel otherwise
        # discovers GPUs and remains bound to them even after model.cpu().
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import default_collate

sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.workspace.dp_workspace import DPWorkspace


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_dataset(dataset):
    buffer = dataset.replay_buffer
    assert buffer.n_episodes == 60 and buffer.n_steps == 10344
    assert dataset.train_mask.all() and len(dataset.get_validation_dataset()) == 0
    assert buffer['state'].shape == buffer['action'].shape == (10344, 28)
    starts, ends = np.r_[0, buffer.episode_ends[:-1]], buffer.episode_ends
    bs, be, ss, se = dataset.sampler.indices.T
    episode_ids = np.searchsorted(ends, bs, side='right')
    assert np.all(bs >= starts[episode_ids]) and np.all(be <= ends[episode_ids])
    assert np.all(be > bs) and np.all(be - bs == se - ss)
    assert dataset.horizon == 16 and (dataset.pad_before, dataset.pad_after) == (1, 15)
    # Two interior windows from the first episode, without boundary padding.
    interior = np.flatnonzero((episode_ids == 0) & (ss == 0) & (se == 16))
    chosen = interior[[len(interior) // 4, 3 * len(interior) // 4]]
    samples = [dataset[int(index)] for index in chosen]
    for index, sample in zip(chosen, samples):
        assert sample['obs']['image'].shape == (16, 224, 224, 3)
        assert sample['obs']['agent_pos'].shape == sample['action'].shape == (16, 28)
        for key, tensor in [('state', sample['obs']['agent_pos']), ('action', sample['action']),
                            ('img', sample['obs']['image'])]:
            assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
            np.testing.assert_array_equal(tensor.numpy(), buffer[key][bs[index]:be[index]])
    return default_collate(samples), [int(index) for index in chosen]


def train_step(workspace, ema, batch):
    workspace._set_model_modes(training=True)
    workspace.optimizer.zero_grad(set_to_none=True)
    loss = workspace.model.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in workspace.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    workspace.optimizer.step()
    ema.step(workspace.model)
    workspace.global_step += 1
    return float(loss.detach()), len(gradients)


@torch.no_grad()
def measure(policy, batch, seed):
    # Seeds are repeated only for measurements; training uses fresh noise/augmentations.
    losses = []
    for offset in range(4):
        set_seed(seed + offset)
        losses.append(float(policy.compute_loss(batch)))
    set_seed(seed)
    prediction = policy.predict_action(batch['obs'])
    assert prediction['action_pred'].shape == (2, 16, 28)
    assert prediction['action'].shape == (2, 15, 28)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    torch.testing.assert_close(prediction['action'], prediction['action_pred'][:, 1:16], rtol=0, atol=0)
    error = prediction['action_pred'] - batch['action']
    groups = ['left_arm', 'right_arm', 'left_ee', 'right_ee']
    metrics = {'mean_noise_loss': float(np.mean(losses)),
               'sampled_action_mse_rad2': float(error.square().mean()),
               'group_mae_rad': {name: float(error[..., i*7:(i+1)*7].abs().mean())
                                 for i, name in enumerate(groups)}}
    assert np.isfinite(metrics['mean_noise_loss'])
    return metrics, prediction


def check_state_dict(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name].cpu(), expected[name].cpu(), rtol=0, atol=0)


def check_reload(workspace, batch, output, seed):
    workspace._set_model_modes(training=False)
    originals = {}
    for name in ['model', 'ema_model']:
        _, originals[name] = measure(getattr(workspace, name), batch, seed)
    checkpoint = workspace.save_checkpoint(use_thread=False)
    # A new workspace starts with an empty normalizer and fresh model/optimizer.
    restored = DPWorkspace(workspace.cfg, output_dir=str(output))
    payload = restored.load_checkpoint(path=checkpoint, include_keys=['global_step', 'epoch'])
    device = batch['action'].device
    restored.model.to(device)
    restored.ema_model.to(device)
    optimizer_to(restored.optimizer, device)
    restored._set_model_modes(training=False)
    assert restored.global_step == workspace.global_step
    assert OmegaConf.to_container(payload['cfg'], resolve=True) == OmegaConf.to_container(workspace.cfg, resolve=True)
    diffs = {}
    for name in ['model', 'ema_model']:
        source, loaded = getattr(workspace, name), getattr(restored, name)
        check_state_dict(loaded.state_dict(), source.state_dict())
        check_state_dict(loaded.normalizer.state_dict(), source.normalizer.state_dict())
        _, prediction = measure(loaded, batch, seed)
        diffs[name] = {}
        for key in ['action_pred', 'action']:
            torch.testing.assert_close(prediction[key], originals[name][key], rtol=0, atol=1e-6)
            diffs[name][key] = float((prediction[key] - originals[name][key]).abs().max())
    assert restored.optimizer.state_dict()['param_groups'] == workspace.optimizer.state_dict()['param_groups']
    original_states, loaded_states = workspace.optimizer.state_dict()['state'], restored.optimizer.state_dict()['state']
    assert original_states.keys() == loaded_states.keys() and original_states
    for key in original_states:
        check_state_dict(loaded_states[key], original_states[key])
    return {'checkpoint': checkpoint, 'model_and_ema_state_exact': True,
            'normalizers_exact': True, 'optimizer_state_exact': True,
            'fixed_seed_prediction_max_abs_diff': diffs}


def main(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu for this offline check.')
    output = (args.output_dir or PROJECT_ROOT / 'data/g1_dex3_training_check' /
              datetime.now().strftime('%Y.%m.%d-%H.%M.%S.%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / 'diffusion_policy_3d/config'), version_base='1.1'):
        cfg = compose(config_name='dp_g1_dex3_rgb')
    cfg.training.device, cfg.training.seed = str(device), args.seed
    cfg.task.dataset.zarr_path = str(args.zarr.resolve())
    # The checker does not initialize WandB or call the full training loop.
    cfg.logging.mode = 'disabled'
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, str(output / 'resolved_config.yaml'))
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch, chosen = check_dataset(dataset)
    normalizer = dataset.get_normalizer()
    batch = dict_apply(batch, lambda value: value.to(device))
    workspace = DPWorkspace(cfg, output_dir=str(output))
    assert not workspace.model.mask_generator((2, 16, 28)).any()
    for model in [workspace.model, workspace.ema_model]:
        model.set_normalizer(normalizer)
        model.to(device)
    optimizer_to(workspace.optimizer, device)
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
    normalizer_state = {name: value.clone() for name, value in normalizer.state_dict().items()}
    bn_state = {name: value.clone() for name, value in workspace.model.named_buffers()}
    rgb = workspace.model.obs_encoder.key_model_map['image'].module.convnet
    tracked = {'r3m_conv': rgb.conv1.weight, 'r3m_bn_weight': rgb.bn1.weight,
               'unet_output': workspace.model.model.final_conv[-1].weight}
    initial_params = {name: value.detach().clone() for name, value in tracked.items()}
    seen_loss_shapes = []

    def record_loss_shape(module, inputs, prediction):
        assert inputs[0].shape == prediction.shape == (2, 16, 28)
        seen_loss_shapes.append(list(prediction.shape))

    hook = workspace.model.model.register_forward_hook(record_loss_shape)
    single_loss, gradient_count = train_step(workspace, ema, batch)
    hook.remove()
    for name, param in tracked.items():
        assert param.grad is not None and torch.count_nonzero(param.grad) > 0
        assert not torch.equal(initial_params[name], param)
    print(f'Dataset and full 16-step backward: PASS (loss={single_loss:.6f})', flush=True)
    workspace._set_model_modes(training=False)
    baseline, _ = measure(workspace.model, batch, 90210)
    history = []
    # Do not reset the seed inside this loop: diffusion noise/timesteps and RGB
    # augmentations change at every update, just as in regular training.
    set_seed(args.seed + 1)
    for step in range(args.overfit_steps):
        loss, _ = train_step(workspace, ema, batch)
        history.append(loss)
        if (step + 1) % 25 == 0 or step + 1 == args.overfit_steps:
            print(f'Fit {step+1}/{args.overfit_steps}: mean recent loss={np.mean(history[-25:]):.6f}', flush=True)
    workspace._set_model_modes(training=False)
    fitted, _ = measure(workspace.model, batch, 90210)
    ema_metrics, _ = measure(workspace.ema_model, batch, 90210)
    fit_passed = (fitted['mean_noise_loss'] < 0.5 * baseline['mean_noise_loss']
                  and fitted['sampled_action_mse_rad2'] < 0.5 * baseline['sampled_action_mse_rad2'])
    for name, buffer in workspace.model.named_buffers():
        torch.testing.assert_close(buffer, bn_state[name], rtol=0, atol=0)
        torch.testing.assert_close(dict(workspace.ema_model.named_buffers())[name], buffer, rtol=0, atol=0)
    for model in [workspace.model, workspace.ema_model]:
        check_state_dict(model.normalizer.state_dict(), normalizer_state)
    reload_result = check_reload(workspace, batch, output, 90210)
    report = {'passed': args.overfit_steps == 0 or fit_passed, 'device': str(device),
              'n_episodes': 60, 'n_frames': 10344, 'n_windows': len(dataset), 'validation_sequences': 0,
              'selected_training_window_indices': chosen, 'backward_shapes': seen_loss_shapes,
              'single_batch_loss': single_loss, 'finite_gradient_tensors': gradient_count,
              'r3m_and_unet_updated': True, 'r3m_bn_statistics_fixed': True,
              'overfit_steps': args.overfit_steps, 'overfit_performed': args.overfit_steps > 0,
              'overfit_optimizer': {'lr': float(cfg.optimizer.lr), 'scheduler': 'none; constant lr',
                                    'formal_training_warmup_used': False},
              'overfit_passed': fit_passed if args.overfit_steps else None,
              'overfit_pass_rule': 'Both fixed-probe noise loss and sampled action MSE fall by >50%',
              'baseline': baseline, 'fitted_model': fitted, 'fitted_ema': ema_metrics,
              'fresh_noise_training_losses': history, 'random_augmentation_preserved': True,
              'sampling_shapes': {'action_pred': [2, 16, 28], 'action': [2, 15, 28]},
              'reload': reload_result, 'formal_training_started': False}
    (output / 'check_result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'fresh_noise_training_losses'}, indent=2))
    print(f'Report: {output / "check_result.json"}', flush=True)
    assert report['passed'], 'Small-batch fitting failed; see check_result.json. More steps may be needed.'


if __name__ == '__main__':
    main(args)
