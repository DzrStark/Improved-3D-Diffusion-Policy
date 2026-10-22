"""Restore a trusted training checkpoint, without a training workspace or downloads."""
import hashlib
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_restore(policy, state):
    """The upstream dynamic normalizer bypasses strict-key checks; check it explicitly."""
    import torch
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
    prefix = "normalizer."
    fields = ("action", "agent_pos", "image")
    leaves = ("offset", "scale", "input_stats.min", "input_stats.max",
              "input_stats.mean", "input_stats.std")
    expected = {prefix + "params_dict." + f + "." + leaf for f in fields for leaf in leaves}
    actual = {k for k in state if k.startswith(prefix)}
    if actual != expected:
        raise ValueError("Missing/unexpected normalizer tensors: " + str(actual ^ expected))
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError("Invalid checkpoint tensor: " + key)
        if key in expected:
            dim = 28 if ".action." in key else 1
            if tuple(value.shape) != (dim,):
                raise ValueError("Invalid normalizer shape: " + key)
    for field in fields:
        scale = state[prefix + "params_dict." + field + ".scale"]
        offset = state[prefix + "params_dict." + field + ".offset"]
        if not (scale > 0).all():
            raise ValueError("Nonpositive normalizer scale")
        if field != "action" and (not torch.equal(scale, torch.ones_like(scale)) or
                                  not torch.equal(offset, torch.zeros_like(offset))):
            raise ValueError("State and image normalizers must be identity")
    # Materialize the dynamic normalizer before comparing the complete state dict.
    normalizer = LinearNormalizer()
    normalizer.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k in expected})
    policy.set_normalizer(normalizer)
    target = policy.state_dict()
    if target.keys() != state.keys():
        raise ValueError("Policy checkpoint key mismatch: " + str(target.keys() ^ state.keys()))
    for key in target:
        if target[key].shape != state[key].shape:
            raise ValueError("Policy checkpoint shape mismatch: " + key)
    policy.load_state_dict(state, strict=True)
    for key, value in policy.state_dict().items():
        if not torch.equal(value.cpu(), state[key].cpu()):
            raise ValueError("Checkpoint restore mismatch: " + key)


class Policy:
    def __init__(self, checkpoint, training_root, device="cpu", threads=4):
        import dill
        import torch
        import hydra
        from omegaconf import OmegaConf
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: cpu, cuda:N")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device unavailable; no silent CPU fallback")
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        training_root = Path(training_root).resolve()
        if not (training_root / "diffusion_policy_3d").is_dir():
            raise FileNotFoundError("Invalid training_root: " + str(training_root))
        sys.path.insert(0, str(training_root))
        r3m_root = training_root.parent / "third_party/r3m"
        if r3m_root.is_dir():
            sys.path.insert(0, str(r3m_root))
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        # Same serialization format as BaseWorkspace; only load locally trusted files.
        payload = torch.load(str(self.checkpoint), map_location="cpu", pickle_module=dill)
        self.cfg = payload["cfg"]
        cfg = self.cfg
        if (cfg.policy._target_ != "diffusion_policy_3d.policy.diffusion_image_policy.DiffusionImagePolicy"
                or not cfg.training.use_ema or (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps) != (16, 2, 16)
                or list(cfg.shape_meta.action.shape) != [28]
                or list(cfg.shape_meta.obs.agent_pos.shape) != [28]
                or list(cfg.shape_meta.obs.image.shape) != [3, 224, 224]
                or cfg.policy.obs_encoder.model_name != "r3m"
                or not cfg.policy.obs_encoder.pretrained or cfg.policy.obs_encoder.frozen
                or not cfg.policy.obs_as_global_cond):
            raise ValueError("Checkpoint does not match the supported G1 RGB contract")
        if "ema_model" not in payload["state_dicts"]:
            raise ValueError("Checkpoint lacks EMA; random/model fallback is forbidden")
        # Resolve only the policy subtree: logging ${now:...} and Dataset ${eval:...}
        # are not needed, and no broad eval resolver is installed.
        self.policy_config = OmegaConf.to_container(cfg.policy, resolve=True)
        import r3m
        from r3m.models.models_r3m import R3M

        class SingleDeviceR3M(torch.nn.Module):
            # Preserve DataParallel's checkpoint namespace without allowing its
            # constructor to discover/bind other GPUs on a multi-GPU host.
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        def architecture_only(modelid, pretrained=True):
            if modelid != "resnet18" or not pretrained:
                raise ValueError("Only the trained pretrained ResNet18 architecture is supported")
            # Same cleaned ~/.r3m/r3m_18/config.yaml architecture. Checkpoint supplies
            # every parameter/buffer. Preserve .module names and BatchNorm topology.
            rep = R3M(device="cpu", lr=1e-4, hidden_dim=1024, size=18,
                      l2weight=1e-5, l1weight=1e-5, langweight=0, tcnweight=1,
                      l2dist=True, bs=16)
            return SingleDeviceR3M(rep)

        # Scoped only to construction. No cached pretrained file, network, or GR1 entry.
        with patch.object(r3m, "load_r3m", architecture_only):
            self.model = hydra.utils.instantiate(OmegaConf.create(self.policy_config))
        state = payload["state_dicts"]["ema_model"]
        strict_restore(self.model, state)
        self.metadata = {"checkpoint": str(self.checkpoint), "sha256": sha256(self.checkpoint),
                         "weights": "ema_model", "tensor_count": len(state),
                         "epoch": dill.loads(payload["pickles"]["epoch"]),
                         "global_step": dill.loads(payload["pickles"]["global_step"]),
                         "device": str(self.device), "threads": threads,
                         "augmentation": self.policy_config["obs_encoder"]["transforms"],
                         "normalizer": "checkpoint action limits; identity state/image"}
        self.model.to(self.device).eval().requires_grad_(False)
        # The wrapper forwards on the explicitly selected single device.
        self.model.obs_encoder.key_model_map["image"].module.device = str(self.device)

    def tensors(self, arrays):
        import torch
        from .observations import joint28, processed_rgb
        if np.asarray(arrays["image"]).shape != (2, 224, 224, 3) or np.asarray(arrays["agent_pos"]).shape != (2, 28):
            raise ValueError("Policy requires exactly two oldest-to-newest observations")
        images = np.stack([processed_rgb(x) for x in arrays["image"]])
        states = np.stack([joint28(x) for x in arrays["agent_pos"]])
        return {"image": torch.as_tensor(images[None], device=self.device),
                "agent_pos": torch.as_tensor(states[None], device=self.device)}

    def predict(self, arrays):
        import torch
        obs = self.tensors(arrays)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model.predict_action(obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        result = {k: v.detach().cpu().numpy()[0] for k, v in result.items()}
        if (result["action"].shape != (15, 28) or result["action_pred"].shape != (16, 28)
                or not all(np.isfinite(x).all() for x in result.values())
                or not np.array_equal(result["action"], result["action_pred"][1:16])):
            raise RuntimeError("Policy returned invalid action or changed the known 15-step slice")
        return result, elapsed
