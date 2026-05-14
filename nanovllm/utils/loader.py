import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _get_param_or_buffer(model: nn.Module, name: str):
    """Return (tensor, is_buffer). Raises AttributeError if not found."""
    try:
        p = model.get_parameter(name)
        return p, False
    except AttributeError:
        pass
    # Try named buffers
    for n, b in model.named_buffers():
        if n == name:
            return b, True
    raise AttributeError(f"No parameter or buffer: {name}")


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    skip_prefixes = getattr(model, "skip_weight_prefixes", ())

    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip vision / audio encoder weights not present in text-only model
                if any(weight_name.startswith(p) for p in skip_prefixes):
                    continue

                # Handle packed projections (e.g. Qwen3 qkv merge)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param, _ = _get_param_or_buffer(model, param_name)
                        except AttributeError:
                            break
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param, is_buf = _get_param_or_buffer(model, weight_name)
                    except AttributeError:
                        continue  # weight not in this model (e.g. multimodal weights)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
