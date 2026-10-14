from contextlib import contextmanager
import gc
import time
import math
from pathlib import Path
from collections.abc import Mapping

import torch
try:
    import deepspeed.comm.comm as dist
except Exception:
    # Inference/evaluation does not require DeepSpeed.  Keep the training
    # backend when it initializes successfully, while allowing a plain
    # single-device runtime when DeepSpeed is absent or its optional CUDA
    # build environment is unavailable (for example on RTX eval nodes).
    import torch.distributed as dist
import imageio
from safetensors import safe_open
import numpy as np


DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float8': torch.float8_e4m3fn,
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
}
VIDEO_EXTENSIONS = set()
for x in imageio.config.video_extensions:
    VIDEO_EXTENSIONS.add(x.extension)
    VIDEO_EXTENSIONS.add(x.extension.upper())
AUTOCAST_DTYPE = None


def get_rank():
    if hasattr(dist, 'is_initialized') and not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@contextmanager
def zero_first():
    initialized = not hasattr(dist, 'is_initialized') or dist.is_initialized()
    if not initialized:
        yield
        return
    if not is_main_process():
        dist.barrier()
    yield
    if is_main_process():
        dist.barrier()


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def log_duration(name):
    start = time.time()
    try:
        yield
    finally:
        print(f'{name}: {time.time()-start:.3f}')


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_state_dict(path):
    path = str(path)
    if path.endswith('.safetensors'):
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, weights_only=True)
    for key in sd:
        if key.endswith('scale_input') or key.endswith('scale_weight'):
            raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
    return sd


def iterate_safetensors(path):
    path = Path(path)
    if path.is_dir():
        safetensors_files = list(path.glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise FileNotFoundError(f'Cound not find safetensors files in directory {path}')
    else:
        if path.suffix != '.safetensors':
            raise ValueError(f'Expected {path} to be a safetensors file')
        safetensors_files = [path]
    for filename in safetensors_files:
        with safe_open(str(filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith('scale_input') or key.endswith('scale_weight'):
                    raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
                yield key, f.get_tensor(key)


def round_to_nearest_multiple(x, multiple):
    return int(round(x / multiple) * multiple)


def round_down_to_multiple(x, multiple):
    return int((x // multiple) * multiple)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _config_value(config, key, default):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def get_t_distribution(model_config, n_buckets: int = 10_000):
    """Build deterministic quantile buckets for training noise fractions."""
    timestep_sample_method = _config_value(model_config, 'timestep_sample_method', 'logit_normal')

    if timestep_sample_method == 'logit_normal':
        dist = torch.distributions.normal.Normal(0, 1)
    elif timestep_sample_method == 'uniform':
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError()

    if n_buckets < 2:
        raise ValueError(f"n_buckets must be at least 2, got {n_buckets}")

    delta = 1 / n_buckets
    min_quantile = delta
    max_quantile = 1 - delta
    quantiles = torch.linspace(min_quantile, max_quantile, n_buckets)
    t = dist.icdf(quantiles)

    if timestep_sample_method == 'logit_normal':
        sigmoid_scale = float(_config_value(model_config, 'sigmoid_scale', 1.0))
        if sigmoid_scale <= 0:
            raise ValueError(f"sigmoid_scale must be positive, got {sigmoid_scale}")
        t = t * sigmoid_scale
        t = torch.sigmoid(t)

    min_t = float(_config_value(model_config, 'min_t', 0.0))
    max_t = float(_config_value(model_config, 'max_t', 1.0))
    if not 0.0 <= min_t < max_t <= 1.0:
        raise ValueError(f"Expected 0 <= min_t < max_t <= 1, got min_t={min_t}, max_t={max_t}")
    return min_t + (max_t - min_t) * t


def sample_scheduler_timestep_ids(
    scheduler,
    t_distribution: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Sample noise fractions and map them to the scheduler's nearest sigma."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if t_distribution.ndim != 1 or t_distribution.numel() == 0:
        raise ValueError("t_distribution must be a non-empty 1D tensor")

    bucket_ids = torch.randint(0, t_distribution.numel(), (batch_size,))
    target_sigmas = t_distribution.detach().cpu()[bucket_ids]
    scheduler_sigmas = scheduler.sigmas.detach().cpu().float()
    return (scheduler_sigmas[:, None] - target_sigmas[None, :]).abs().argmin(dim=0)


def slice_t_distribution(t, min_t=0.0, max_t=1.0):
    start = torch.searchsorted(t, min_t).item()
    end = torch.searchsorted(t, max_t).item()
    return t[start:end]


def sample_t(t, batch_size, quantile=None):
    if quantile is not None:
        i = (torch.full((batch_size,), quantile) * len(t)).to(torch.int32)
    else:
        i = torch.randint(0, len(t), size=(batch_size,))
    return t[i]


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """
    Get 1D positional embedding in the form of sin and cos.

    Paper:
    https://arxiv.org/abs/1706.03762

    Source:
    https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

    Args:
        embed_dim (int): output dimension for each position.
        pos (ndarray | list): a list of positions to be encoded, size (M,).
    Returns:
        out (ndarray): resulting positional embedding, size (M, D).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid_sizes):
    """
    Get ND positional embedding from grid sizes.
    All dimensions are summed up for factorization.

    Paper:
    https://arxiv.org/abs/2307.06304

    Args:
        embed_dim (int): output dimension for each position.
        grid_sizes (tuple): grids sizes in each dimension, length = K.
            If some grid size is lower than 1, we do not add any positional embedding.
    Returns:
        out (ndarray): resulting positional embedding, size (grid_sizes[0], ..., grid_sizes[K-1], D).
    """
    # We sum up all dimensions for factorization
    emb = np.zeros(grid_sizes + (embed_dim,))
    for size_idx, grid_size in enumerate(grid_sizes):
        # For grid size of 1, we do not need to add any positional embedding
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [embed_dim]
        posemb_shape[size_idx] = -1
        emb += get_1d_sincos_pos_embed_from_grid(embed_dim, pos).reshape(posemb_shape)
    return emb
