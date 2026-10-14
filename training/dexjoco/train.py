#!/usr/bin/env python3
# Training script for Motus

import os
import gc
import re
import sys
import argparse
import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
from contextlib import nullcontext

# Set CUDA memory management environment variables to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
from accelerate.utils import DeepSpeedPlugin, DistributedType, ProjectConfiguration
import yaml
from omegaconf import OmegaConf
from datetime import datetime

# Add project root to path


from runtime.dexjoco.models.motus import Motus, MotusConfig
from training.dexjoco.data.factory import _worker_init_fn, create_dataset, collate_fn
from runtime.dexjoco.utils.scheduler import create_scheduler
from runtime.dexjoco.utils.attention_contract import validate_checkpoint_joint_attention_mode
from runtime.dexjoco.utils.five_modal_initialization import (
    build_five_modal_initialization_report,
    write_initialization_report,
)
if __package__:
    from training.dexjoco.sample import evaluate_model, log_evaluation_metrics
else:
    from training.dexjoco.sample import evaluate_model, log_evaluation_metrics

logger = logging.getLogger(__name__)

PERFORMANCE_TIMING_NAMES = (
    'data_wait_time',
    'compute_wall_time',
    'step_wall_time',
    'video_decode_time',
    'qpos_load_time',
    'language_load_time',
    'sample_load_time',
    'hdf5_rgb_decode_time',
    'npy_slice_time',
    'codec_mosaic_time',
    'raw_qpos_load_time',
    'raw_language_load_time',
    'raw_sample_load_time',
    'h2d_time',
    'worker_rss_gib',
    'worker_open_file_count',
    'worker_cached_hdf5_files',
    'worker_cached_npy_files',
    'worker_resource_metrics_present',
)


def _format_duration(seconds: float) -> str:
    """Format an ETA without depending on locale or external logging packages."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"

def setup_logging(rank: int = 0, log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress specific distributed warnings that are noisy but harmless for our flow
    warnings.filterwarnings(
        "ignore",
        message=r"No device id is provided via `init_process_group` or `barrier`.*",
        category=UserWarning,
    )

def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Preserve an explicit native-clock action horizon.  Older configs omit
    # this key and retain their original video-count × ratio derivation.
    if config.common.get('action_chunk_size', None) is None:
        config.common.action_chunk_size = (
            config.common.num_video_frames * config.common.video_action_freq_ratio
        )
    config.common.action_chunk_size = int(config.common.action_chunk_size)

    # Validate dataset configuration
    dataset_config = {
        'dataset_type': config.dataset.type,
        'dataset_dir': config.dataset.dataset_dir if hasattr(config.dataset, 'dataset_dir') else None,
        'global_downsample_rate': config.common.global_downsample_rate,
        'video_action_freq_ratio': config.common.video_action_freq_ratio,
        'num_video_frames': config.common.num_video_frames
    }

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Dataset type: {config.dataset.type}")
    if hasattr(config, 'training_mode'):
        logger.info(f"Training mode: {config.training_mode}")
    logger.info(f"Action chunk size: {config.common.action_chunk_size}")
    logger.info(f"Video frames: {config.common.num_video_frames}")
    logger.info(
        "Joint attention mode: %s",
        config.model.get('joint_attention_mode', 'full_bidirectional'),
    )

    return config


def validate_global_batch_size(config: OmegaConf, world_size: int) -> int:
    """Fail when the declared global batch does not match runtime topology."""
    per_device_batch = int(config.training.batch_size)
    accumulation_steps = int(config.training.get('gradient_accumulation_steps', 1))
    actual_global_batch = per_device_batch * int(world_size) * accumulation_steps
    declared_global_batch = config.training.get('global_batch_size', None)
    expected_world_size = config.training.get('expected_world_size', None)

    if expected_world_size is not None and int(expected_world_size) != int(world_size):
        raise ValueError(
            f"Expected world_size={expected_world_size}, got {world_size}; "
            "update the topology or gradient accumulation before training"
        )
    if declared_global_batch is not None and int(declared_global_batch) != actual_global_batch:
        raise ValueError(
            f"Declared global_batch_size={declared_global_batch}, but runtime computes "
            f"{per_device_batch} * {world_size} * {accumulation_steps} = {actual_global_batch}"
        )
    return actual_global_batch


def validate_deepspeed_runtime(model, config: OmegaConf, actual_global_batch: int) -> Dict[str, Any]:
    """Validate the resolved ZeRO-1 runtime before the first training batch."""
    if not hasattr(model, "zero_optimization_stage"):
        raise RuntimeError("Expected a DeepSpeedEngine after accelerator.prepare")

    runtime = {
        "zero_stage": int(model.zero_optimization_stage()),
        "gradient_accumulation_steps": int(model.gradient_accumulation_steps()),
        "train_micro_batch_size_per_gpu": int(model.train_micro_batch_size_per_gpu()),
        "train_batch_size": int(model.train_batch_size()),
    }
    expected = {
        "zero_stage": 1,
        "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
        "train_micro_batch_size_per_gpu": int(config.training.batch_size),
        "train_batch_size": int(actual_global_batch),
    }
    if runtime != expected:
        raise RuntimeError(f"DeepSpeed runtime contract mismatch: runtime={runtime}, expected={expected}")

    zero_optimizer = model.optimizer
    if not bool(getattr(zero_optimizer, "check_grad_overflow", False)):
        raise RuntimeError("DeepSpeed BF16 gradient overflow checking must be enabled")
    master_groups = getattr(zero_optimizer, "single_partition_of_fp32_groups", None)
    if not master_groups or any(group.dtype != torch.float32 for group in master_groups):
        dtypes = [str(group.dtype) for group in (master_groups or [])]
        raise RuntimeError(f"ZeRO-1 master partitions must be FP32, got {dtypes}")
    runtime["check_grad_overflow"] = True
    runtime["master_partition_dtypes"] = sorted({str(group.dtype) for group in master_groups})
    return runtime

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


class DataLoaderShardCheckpointState:
    """Persist Accelerate's logical dataloader epoch across native resumes."""

    def __init__(self, dataloader: DataLoaderShard):
        if not isinstance(dataloader, DataLoaderShard):
            raise TypeError(
                "Prepared training dataloader must expose DataLoaderShard.iteration"
            )
        self.dataloader = dataloader

    def state_dict(self) -> Dict[str, int]:
        return {"iteration": int(self.dataloader.iteration)}

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        if "iteration" not in state_dict:
            raise KeyError("DataLoaderShard checkpoint state is missing 'iteration'")
        iteration = int(state_dict["iteration"])
        if iteration < 0:
            raise ValueError(f"DataLoaderShard iteration must be non-negative: {iteration}")
        self.dataloader.iteration = iteration


def count_dataloader_shards(dataloader: Any) -> int:
    """Count Accelerate batch-sampler shards through loader wrappers."""
    pending = [dataloader]
    visited = set()
    shard_count = 0
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if isinstance(candidate, BatchSamplerShard):
            shard_count += 1
        for attribute in ("base_dataloader", "sampler", "batch_sampler"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)
    return shard_count


def resolve_dataset_task_names(dataloader: Any) -> tuple[str, ...]:
    """Resolve the fixed RoboTwin task table through Accelerate wrappers."""
    pending = [dataloader]
    visited = set()
    while pending:
        candidate = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        task_names = getattr(candidate, "task_names", None)
        if task_names is not None:
            return tuple(str(name) for name in task_names)
        for attribute in ("base_dataloader", "dataset"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)
    return ()


def checkpoint_step_from_path(checkpoint_path: str | Path) -> int:
    """Read the step only from an exact checkpoint_step_<N> directory name."""
    checkpoint_name = Path(checkpoint_path).name
    match = re.fullmatch(r"checkpoint_step_(\d+)", checkpoint_name)
    if match is None or int(match.group(1)) <= 0:
        raise ValueError(
            "Checkpoint directory must be named checkpoint_step_<positive integer>: "
            f"{checkpoint_path}"
        )
    return int(match.group(1))


class UniDiffuserTrainer:
    """Trainer class for Motus."""

    def __init__(
        self,
        model: Motus,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: str = "./checkpoints_stage4",
        log_interval: int = 100,
        save_interval: int = 1000,
        val_interval: int = 1000,
        report_to: str = "wandb",
        tb_writer: Optional[SummaryWriter] = None,
        accelerator: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.report_to = report_to
        self.tb_writer = tb_writer
        self.accelerator = accelerator
        self.config = config

        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking variables
        self.global_step = 0
        self.epoch = 0
        system_config = getattr(config, 'system', None)
        self.task_names = resolve_dataset_task_names(train_dataloader)
        self.task_name_to_index = {
            task_name: index for index, task_name in enumerate(self.task_names)
        }
        self.per_task_action_loss_interval = int(
            getattr(system_config, 'per_task_action_loss_interval', 0)
        )
        if self.per_task_action_loss_interval < 0:
            raise ValueError("per_task_action_loss_interval must be non-negative")
        if self.per_task_action_loss_interval and not self.task_names:
            raise RuntimeError(
                "Per-task action-loss logging requires a fixed dataset task_names table"
            )
        self._per_task_action_loss_sums = torch.zeros(
            len(self.task_names), dtype=torch.float64
        )
        self._per_task_action_loss_counts = torch.zeros(
            len(self.task_names), dtype=torch.float64
        )
        self._metric_sums: Dict[str, float] = {}
        self._micro_batches_in_update = 0
        self.dexjoco_train_audit = os.environ.get("DEXJOCO_TRAIN_AUDIT", "0") == "1"
        self._dexjoco_embodiment_counts = {"single": 0, "bimanual": 0}
        self._update_start_time: Optional[float] = None
        self.optimizer_skipped_steps = 0
        self.memory_log_interval = int(
            getattr(getattr(config, 'system', None), 'memory_log_interval', 50)
        )
        self._last_memory_metrics: Dict[str, float] = {}
        checkpoint_total_limit = getattr(system_config, 'checkpoint_total_limit', 20)
        self.checkpoint_total_limit = (
            None if checkpoint_total_limit is None else int(checkpoint_total_limit)
        )
        self.save_final_checkpoint = bool(
            getattr(system_config, 'save_final_checkpoint', True)
        )
        if self.checkpoint_total_limit is not None and self.checkpoint_total_limit < 1:
            raise ValueError("checkpoint_total_limit must be positive")
        self.performance_warmup_steps = int(
            getattr(system_config, 'performance_warmup_steps', 50)
        )
        self.performance_measurement_steps = int(
            getattr(system_config, 'performance_measurement_steps', 400)
        )
        self.disable_python_gc = bool(
            getattr(system_config, 'disable_python_gc', False)
        )
        threshold_config = getattr(system_config, 'performance_thresholds', None)
        threshold_defaults = {
            'data_wait_time_p95_lt': 0.5,
            'data_wait_time_max_lt': 2.0,
            'step_wall_time_mean_lte': 1.2,
            'step_wall_time_p95_lte': 1.5,
            'data_wait_p95_to_step_p95_lte': 0.25,
            'rank_data_wait_p95_to_global_median_lte': 1.5,
        }
        self.performance_thresholds = {
            name: float(getattr(threshold_config, name, default))
            for name, default in threshold_defaults.items()
        }
        dataset_config = getattr(config, 'dataset', None)
        self.profile_data_pipeline = bool(
            getattr(dataset_config, 'profile_component_times', False)
        )
        self.require_worker_resource_metrics = bool(
            getattr(dataset_config, 'type', None)
            == 'robotwin_raw_five_modal_multiview'
            and self.profile_data_pipeline
        )
        self.worker_resource_limits = {
            'worker_rss_gib': float(getattr(dataset_config, 'max_worker_rss_gib', 16.0)),
            'worker_open_file_count': float(
                getattr(dataset_config, 'max_worker_open_files', 64)
            ),
            'worker_cached_hdf5_files': float(
                getattr(dataset_config, 'max_open_hdf5_files', 8)
            ),
            'worker_cached_npy_files': float(
                getattr(dataset_config, 'max_open_npy_files', 16)
            ),
        }
        self._timing_window: list[list[float]] = []
        self._performance_samples = {name: [] for name in PERFORMANCE_TIMING_NAMES}
        self._performance_samples_by_rank = {
            rank_index: {name: [] for name in PERFORMANCE_TIMING_NAMES}
            for rank_index in range(self.world_size)
        }
        self._run_video_modality_counts = [0] * 5
        self._performance_measurement_start_unix: Optional[float] = None
        self._performance_measurement_end_unix: Optional[float] = None
        self._performance_summary_written = False
        self.uses_deepspeed = (
            accelerator is not None
            and getattr(accelerator, "distributed_type", None) == DistributedType.DEEPSPEED
        )
        self.deepspeed_engine_owns_scheduler = bool(
            self.uses_deepspeed and getattr(self.model, 'lr_scheduler', None) is not None
        )
        self.video_training_strategy = str(
            getattr(getattr(getattr(config, 'model', None), 'wan', None), 'training_strategy', 'flow_matching')
        )
        self.video_rollout_steps = int(
            getattr(getattr(getattr(config, 'model', None), 'wan', None), 'rollout_steps', 10)
        )
        if self.video_training_strategy == 'frozen_rollout':
            accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))
            if accumulation_steps != 1:
                raise ValueError(
                    "frozen_rollout requires gradient_accumulation_steps=1; its 10 backward calls "
                    "are V substeps, not data micro-batches"
                )
            if self.video_rollout_steps <= 0:
                raise ValueError("Frozen WAN rollout_steps must be positive")
        self._audit_grad_square_sums: Dict[str, Optional[torch.Tensor]] = {
            'action': None,
            'tactile': None,
        }
        self._audit_gradient_hooks = []
        if self.dexjoco_train_audit:
            unwrapped_model = self._unwrap_training_model()
            for name, module in (
                ('action', getattr(unwrapped_model, 'action_expert', None)),
                ('tactile', getattr(unwrapped_model, 'tactile_expert', None)),
            ):
                if module is None:
                    raise RuntimeError(f"Missing {name} module for DexJoCo audit")
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        def accumulate(gradient, branch=name):
                            contribution = gradient.detach().float().square().sum()
                            current = self._audit_grad_square_sums[branch]
                            self._audit_grad_square_sums[branch] = (
                                contribution if current is None else current + contribution
                            )
                            return gradient
                        self._audit_gradient_hooks.append(
                            parameter.register_hook(accumulate)
                        )

        logger.info(f"Motus Trainer initialized on rank {rank}/{world_size}")
        logger.info(f"Logging backends: {report_to}")

    def collect_cuda_memory_metrics(self) -> Dict[str, float]:
        """Return the maximum CUDA memory values observed across all ranks."""
        if not torch.cuda.is_available():
            return {}

        values = torch.tensor(
            [
                torch.cuda.memory_allocated(self.device),
                torch.cuda.memory_reserved(self.device),
                torch.cuda.max_memory_allocated(self.device),
                torch.cuda.max_memory_reserved(self.device),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        values = values.cpu() / (1024 ** 3)
        return {
            'cuda_current_allocated_gib': values[0].item(),
            'cuda_current_reserved_gib': values[1].item(),
            'cuda_peak_allocated_gib': values[2].item(),
            'cuda_peak_reserved_gib': values[3].item(),
        }

    @staticmethod
    def _distribution_stats(values: list[float]) -> Dict[str, float]:
        if not values:
            return {}
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            'min': tensor.min().item(),
            'mean': tensor.mean().item(),
            'p50': torch.quantile(tensor, 0.50).item(),
            'p95': torch.quantile(tensor, 0.95).item(),
            'max': tensor.max().item(),
        }

    def collect_timing_window_metrics(self) -> Dict[str, float]:
        """Aggregate the complete timing window across ranks at log boundaries."""
        if not self._timing_window:
            return {}
        metric_device = self.device if torch.cuda.is_available() else torch.device('cpu')
        local = torch.tensor(self._timing_window, dtype=torch.float64, device=metric_device)
        self._timing_window.clear()
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(gathered, local)
            gathered_cpu = [rank_values.cpu() for rank_values in gathered]
            combined = torch.cat(gathered_cpu, dim=0)
        else:
            gathered_cpu = [local.cpu()]
            combined = gathered_cpu[0]

        names = PERFORMANCE_TIMING_NAMES[: combined.shape[1] - 1]
        metrics = {}
        for column, name in enumerate(names, start=1):
            values = combined[:, column].tolist()
            for statistic, value in self._distribution_stats(values).items():
                metrics[f'{name}_window_{statistic}'] = value

        if self.rank == 0:
            measurement_mask = (
                (combined[:, 0] > self.performance_warmup_steps)
                & (combined[:, 0] <= self.performance_measurement_steps)
            )
            measured = combined[measurement_mask]
            for column, name in enumerate(names, start=1):
                self._performance_samples[name].extend(measured[:, column].tolist())
            for rank_index, rank_values in enumerate(gathered_cpu):
                rank_mask = (
                    (rank_values[:, 0] > self.performance_warmup_steps)
                    & (rank_values[:, 0] <= self.performance_measurement_steps)
                )
                rank_measured = rank_values[rank_mask]
                for column, name in enumerate(names, start=1):
                    self._performance_samples_by_rank[rank_index][name].extend(
                        rank_measured[:, column].tolist()
                    )
        return metrics

    def collect_due_timing_window_metrics(self) -> Dict[str, float]:
        """Flush timings at log boundaries and at the measurement boundary."""
        if (
            self.global_step % self.log_interval != 0
            and self.global_step != self.performance_measurement_steps
        ):
            return {}
        return self.collect_timing_window_metrics()

    def write_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Write the exact rank-step timing gate summary once at the measurement boundary."""
        if self.rank != 0 or self._performance_summary_written:
            return None
        timing = {
            name: self._distribution_stats(values)
            for name, values in self._performance_samples.items()
        }
        thresholds = dict(self.performance_thresholds)
        expected_samples = max(
            self.performance_measurement_steps - self.performance_warmup_steps, 0
        ) * self.world_size
        actual_samples = len(self._performance_samples['data_wait_time'])
        per_rank_timing = {
            str(rank_index): {
                name: self._distribution_stats(values)
                for name, values in rank_samples.items()
            }
            for rank_index, rank_samples in self._performance_samples_by_rank.items()
        }
        rank_data_wait_p95 = [
            per_rank_timing[str(rank_index)]['data_wait_time'].get('p95', float('inf'))
            for rank_index in range(self.world_size)
        ]
        finite_rank_p95 = [value for value in rank_data_wait_p95 if math.isfinite(value)]
        global_rank_median = (
            torch.median(torch.tensor(finite_rank_p95, dtype=torch.float64)).item()
            if finite_rank_p95 else float('inf')
        )
        max_rank_p95 = max(rank_data_wait_p95, default=float('inf'))
        if global_rank_median > 0.0 and math.isfinite(global_rank_median):
            rank_imbalance_ratio = max_rank_p95 / global_rank_median
        elif max_rank_p95 == 0.0:
            rank_imbalance_ratio = 1.0
        else:
            rank_imbalance_ratio = float('inf')
        step_p95 = timing['step_wall_time'].get('p95', 0.0)
        data_wait_to_step_ratio = (
            timing['data_wait_time'].get('p95', float('inf')) / step_p95
            if step_p95 > 0.0 else float('inf')
        )
        ratio_gates = {
            'data_wait_p95_to_step_p95': data_wait_to_step_ratio,
            'rank_data_wait_p95_global_median': global_rank_median,
            'rank_data_wait_p95_max': max_rank_p95,
            'rank_data_wait_p95_to_global_median': rank_imbalance_ratio,
        }
        resource_checks: Dict[str, Any] = {
            'required': self.require_worker_resource_metrics,
            'limits': dict(self.worker_resource_limits),
            'metrics_present_for_every_rank_step': True,
            'observed_maxima': {},
            'passed': True,
        }
        if self.require_worker_resource_metrics:
            presence = timing['worker_resource_metrics_present']
            resource_checks['metrics_present_for_every_rank_step'] = bool(
                presence and presence.get('min', 0.0) >= 1.0
            )
            for name, limit in self.worker_resource_limits.items():
                observed = timing[name].get('max', float('inf'))
                resource_checks['observed_maxima'][name] = observed
                if not math.isfinite(observed) or observed < 0.0 or observed > limit:
                    resource_checks['passed'] = False
            resource_checks['passed'] = bool(
                resource_checks['passed']
                and resource_checks['metrics_present_for_every_rank_step']
            )
        timing_gate_passed = bool(
            actual_samples == expected_samples
            and self.optimizer_skipped_steps == 0
            and timing['data_wait_time']['p95'] < thresholds['data_wait_time_p95_lt']
            and timing['data_wait_time']['max'] < thresholds['data_wait_time_max_lt']
            and timing['step_wall_time']['mean'] <= thresholds['step_wall_time_mean_lte']
            and timing['step_wall_time']['p95'] <= thresholds['step_wall_time_p95_lte']
            and data_wait_to_step_ratio <= thresholds['data_wait_p95_to_step_p95_lte']
            and rank_imbalance_ratio <= thresholds['rank_data_wait_p95_to_global_median_lte']
            and resource_checks['passed']
        )
        summary = {
            'world_size': self.world_size,
            'warmup_steps_excluded': self.performance_warmup_steps,
            'measurement_end_step': self.performance_measurement_steps,
            'measurement_start_unix': self._performance_measurement_start_unix,
            'measurement_end_unix': self._performance_measurement_end_unix,
            'expected_rank_step_samples': expected_samples,
            'actual_rank_step_samples': actual_samples,
            'optimizer_skipped_steps': self.optimizer_skipped_steps,
            'timing': timing,
            'per_rank_timing': per_rank_timing,
            'ratio_gates': ratio_gates,
            'worker_resource_gate': resource_checks,
            'thresholds': thresholds,
            'timing_gate_passed': timing_gate_passed,
        }
        output_path = self.checkpoint_dir / 'performance_gate_summary.json'
        temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
        with temporary_path.open('w', encoding='utf-8') as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)
            output_file.write('\n')
        os.replace(temporary_path, output_path)
        self._performance_summary_written = True
        logger.info("PERFORMANCE_GATE_SUMMARY %s", json.dumps(summary, sort_keys=True))
        return summary

    def write_training_run_summary(
        self, *, requested_max_steps: int, elapsed_seconds: float
    ) -> Optional[Dict[str, Any]]:
        """Persist run-level modality and optimizer evidence for short gates."""
        if self.rank != 0:
            return None
        total_samples = sum(self._run_video_modality_counts)
        observed_fractions = [
            count / total_samples if total_samples else 0.0
            for count in self._run_video_modality_counts
        ]
        configured_weights = None
        if self.config is not None:
            model_config = getattr(self.config, 'model', None)
            wan_config = getattr(model_config, 'wan', None)
            weights = getattr(wan_config, 'task_probabilities', None)
            if weights is not None:
                configured_weights = [float(value) for value in weights]
        configured_fractions = None
        if configured_weights is not None and sum(configured_weights) > 0:
            configured_fractions = [
                value / sum(configured_weights) for value in configured_weights
            ]
        summary = {
            'global_step': self.global_step,
            'requested_max_steps': int(requested_max_steps),
            'elapsed_seconds': float(elapsed_seconds),
            'optimizer_skipped_steps': self.optimizer_skipped_steps,
            'video_modality_counts': {
                str(task_id): int(count)
                for task_id, count in enumerate(self._run_video_modality_counts)
            },
            'video_modality_total_samples': int(total_samples),
            'video_modality_observed_fractions': observed_fractions,
            'video_modality_configured_weights': configured_weights,
            'video_modality_configured_fractions': configured_fractions,
        }
        output_path = self.checkpoint_dir / 'training_run_summary.json'
        temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
        with temporary_path.open('w', encoding='utf-8') as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)
            output_file.write('\n')
        os.replace(temporary_path, output_path)
        logger.info("TRAINING_RUN_SUMMARY %s", json.dumps(summary, sort_keys=True))
        return summary

    def prune_checkpoints(self) -> list[str]:
        """Remove the oldest native checkpoint directories beyond the configured limit."""
        if self.rank != 0 or self.checkpoint_total_limit is None:
            return []
        checkpoints = []
        for path in self.checkpoint_dir.glob('checkpoint_step_*'):
            match = re.fullmatch(r'checkpoint_step_(\d+)(?:.*)?', path.name)
            if path.is_dir() and match:
                checkpoints.append((int(match.group(1)), path))
        checkpoints.sort(key=lambda item: item[0])
        removed = []
        for _, stale_path in checkpoints[:-self.checkpoint_total_limit]:
            shutil.rmtree(stale_path)
            removed.append(str(stale_path))
            logger.info(
                "Removed stale checkpoint due to total_limit=%s: %s",
                self.checkpoint_total_limit,
                stale_path,
            )
        return removed

    def save_checkpoint(self, suffix: str = ""):
        """Save complete training state using accelerator."""
        checkpoint_dir = self.checkpoint_dir / f"checkpoint_step_{self.global_step}{suffix}"

        # Use accelerator to save complete training state
        # This saves model, optimizer, scheduler, dataloader, and RNG states
        self.accelerator.save_state(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        # DeepSpeed state is partitioned across ranks; this shared metadata file is not.
        if self.rank == 0:
            try:
                from omegaconf import OmegaConf as _OmegaConf
                cfg_dict = _OmegaConf.to_container(self.config, resolve=True) if self.config is not None else {}
                # Filter only requested sections
                common = dict(cfg_dict.get("common", {}))
                common.setdefault(
                    "action_chunk_size",
                    int(common.get("num_video_frames", 0))
                    * int(common.get("video_action_freq_ratio", 0)),
                )
                model = cfg_dict.get("model", {})
                filtered = {
                    "common": common,
                    "architecture": "wam_va" if model.get("use_understanding", True) is False else "motus",
                    "joint_attention_mode": model.get(
                        "joint_attention_mode", "full_bidirectional"
                    ),
                    "model": {
                        "use_understanding": model.get("use_understanding", True),
                        "wan": model.get("wan", {}),
                        "loss_weights": model.get("loss_weights", {}),
                        "tactile": model.get("tactile", {}),
                        "tactile_expert": model.get("tactile_expert", {}),
                    },
                    "dataset": {
                        "type": cfg_dict.get("dataset", {}).get("type"),
                        "condition_only": cfg_dict.get("dataset", {}).get("condition_only", False),
                    },
                    "training": {
                        "gradient_accumulation_steps": cfg_dict.get("training", {}).get(
                            "gradient_accumulation_steps", 1
                        ),
                    },
                    "action_expert": model.get("action_expert", {}),
                    "und_expert": model.get("und_expert", {}),
                    "time_distribution": model.get("time_distribution", {}),
                    "ema": model.get("ema", {}),
                }
                import json as _json
                with open(checkpoint_dir / "config.json", "w") as f:
                    _json.dump(filtered, f, indent=2)
                logger.info(f"Wrote config.json to {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to write config.json: {e}")
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        self.prune_checkpoints()
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

    def load_checkpoint(self, checkpoint_path: str, reset_scheduler: bool = True):
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint directory
            reset_scheduler: If True, reset scheduler to new config instead of loading from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        runtime_attention_mode = "full_bidirectional"
        if self.config is not None and hasattr(self.config, "model"):
            runtime_attention_mode = self.config.model.get(
                "joint_attention_mode", "full_bidirectional"
            )
        attention_report = validate_checkpoint_joint_attention_mode(
            checkpoint_path,
            runtime_attention_mode,
        )
        logger.info("Checkpoint attention contract: %s", attention_report)

        self.global_step = checkpoint_step_from_path(checkpoint_path)
        logger.info(f"Resuming from step {self.global_step}")

        # Load using accelerator (includes model, optimizer, scheduler states)
        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
        if hasattr(self.train_dataloader, "iteration"):
            self.epoch = int(self.train_dataloader.iteration)
            logger.info(f"Restored DataLoaderShard iteration: {self.epoch}")

        # Reset scheduler with new config if requested
        if reset_scheduler and self.config is not None and self.scheduler is not None:
            logger.info("Resetting scheduler to new configuration (not using checkpoint scheduler state)...")

            # Unwrap scheduler if it's wrapped by accelerator
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module

            # Check if it's our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'warm_up_steps'):
                # Update scheduler parameters with new config
                unwrapped_scheduler.warm_up_steps = self.config.training.warmup_steps
                unwrapped_scheduler.cycle_length = self.config.training.cycle_length
                unwrapped_scheduler.f_max = self.config.training.f_max
                unwrapped_scheduler.f_min = self.config.training.f_min
                # Update base_lrs for all parameter groups
                unwrapped_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

                # Reset step_count to 0 so scheduler starts warmup from beginning
                unwrapped_scheduler.step_count = 0

                logger.info(f"Updated scheduler config: warmup={unwrapped_scheduler.warm_up_steps}, "
                          f"cycle_length={unwrapped_scheduler.cycle_length}, "
                          f"f_max={unwrapped_scheduler.f_max}, f_min={unwrapped_scheduler.f_min}")
                logger.info(f"Base learning rates: {[f'{lr:.2e}' for lr in unwrapped_scheduler.base_lrs]}")

                # Don't directly modify optimizer's lr! Let scheduler update it naturally on next step
                # Only log the target lr that scheduler will set
                initial_lrs = [base_lr * unwrapped_scheduler.f_max for base_lr in unwrapped_scheduler.base_lrs]
                logger.info(f"Reset scheduler step_count to 0 (will start warmup from next step)")
                logger.info(f"Target initial learning rates: {[f'{lr:.2e}' for lr in initial_lrs]}")
                logger.info(f"Learning rate will be updated by scheduler on first training step")

            # Log current learning rate (from checkpoint)
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (will be overridden by scheduler): {current_lr:.2e}")
        elif self.scheduler is not None:
            # If not resetting scheduler, sync scheduler progress with global_step
            unwrapped_scheduler = self.scheduler
            if hasattr(self.scheduler, 'module'):
                unwrapped_scheduler = self.scheduler.module

            # Case 1: our custom LambdaLinearScheduler
            if hasattr(unwrapped_scheduler, 'step_count'):
                old_step_count = unwrapped_scheduler.step_count
                unwrapped_scheduler.step_count = self.global_step
                logger.info(f"Synchronized scheduler step_count: {old_step_count} -> {self.global_step}")

            # Case 2: diffusers_cosine wrapper with inner scheduler
            if hasattr(unwrapped_scheduler, 'inner') and hasattr(unwrapped_scheduler.inner, 'last_epoch'):
                try:
                    old_epoch = int(getattr(unwrapped_scheduler.inner, 'last_epoch', -1))
                except Exception:
                    old_epoch = -1
                # Align inner scheduler epoch with current global_step so schedule continues
                unwrapped_scheduler.inner.last_epoch = int(self.global_step)
                logger.info(f"Aligned diffusers scheduler last_epoch: {old_epoch} -> {self.global_step}")

            # Log current optimizer LR (authoritative)
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate after checkpoint load (optimizer): {current_lr:.2e}")

    def _unwrap_training_model(self):
        if self.accelerator is not None and hasattr(self.accelerator, 'unwrap_model'):
            return self.accelerator.unwrap_model(self.model)
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _clear_gradients(self) -> None:
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()
        for parameter in self.model.parameters():
            parameter.grad = None

    @staticmethod
    def _local_losses_finite(loss_dict: Dict[str, Any]) -> torch.Tensor:
        """Return a device-side scalar without synchronizing CUDA to Python."""
        checks = [
            torch.isfinite(value.detach()).all()
            for key, value in loss_dict.items()
            if key != 'next_video_latent' and torch.is_tensor(value)
        ]
        if not checks:
            return torch.ones(
                (), device=loss_dict['total_loss'].device, dtype=torch.bool
            )
        return torch.stack(checks).all()

    @staticmethod
    def _synchronize_finite_flag(local_finite: torch.Tensor) -> bool:
        finite = local_finite.to(dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        return bool(finite.item())

    def _losses_are_finite(self, loss_dict: Dict[str, Any]) -> bool:
        return self._synchronize_finite_flag(self._local_losses_finite(loss_dict))

    def _train_frozen_wan_rollout(
        self,
        *,
        first_frame: torch.Tensor,
        state: Optional[torch.Tensor],
        actions: torch.Tensor,
        action_valid_mask: Optional[torch.Tensor],
        language_embeddings,
    ) -> Dict[str, float]:
        """Run 10 dependent V forwards, 10 immediate backwards, and one optimizer update."""
        if language_embeddings is None:
            raise ValueError("Frozen WAN rollout requires instruction T5 embeddings")
        unwrapped_model = self._unwrap_training_model()
        rollout = unwrapped_model.initialize_frozen_video_rollout(
            first_frame,
            language_embeddings,
        )
        video_timesteps = rollout['video_timesteps']
        if video_timesteps.numel() != self.video_rollout_steps + 1:
            raise RuntimeError(
                f"Expected {self.video_rollout_steps + 1} V timesteps, got {video_timesteps.numel()}"
            )

        action_losses = []
        action_sigmas = []
        rollout_losses_finite = torch.ones(
            (), device=first_frame.device, dtype=torch.bool
        )
        try:
            for substep in range(self.video_rollout_steps):
                loss_dict = self.model(
                    first_frame=first_frame,
                    video_frames=None,
                    state=state,
                    actions=actions,
                    action_valid_mask=action_valid_mask,
                    language_embeddings=None,
                    vlm_inputs=None,
                    return_dict=True,
                    video_latent=rollout['video_latent'],
                    condition_video_latent=rollout['condition_video_latent'],
                    processed_t5_context=rollout['processed_t5_context'],
                    video_timestep=video_timesteps[substep],
                    video_dt=video_timesteps[substep + 1] - video_timesteps[substep],
                    video_task_ids=rollout['video_task_ids'],
                )
                rollout_losses_finite = torch.logical_and(
                    rollout_losses_finite,
                    self._local_losses_finite(loss_dict),
                )
                next_video_latent = loss_dict['next_video_latent']
                if next_video_latent.grad_fn is not None or next_video_latent.requires_grad:
                    self._clear_gradients()
                    raise RuntimeError(f"V latent retained a cross-step graph at substep {substep}")

                scaled_loss = loss_dict['total_loss'] / self.video_rollout_steps
                if self.uses_deepspeed:
                    if not hasattr(self.model, 'set_gradient_accumulation_boundary'):
                        raise RuntimeError(
                            "DeepSpeedEngine lacks set_gradient_accumulation_boundary; "
                            "cannot defer rollout gradient synchronization"
                        )
                    self.model.set_gradient_accumulation_boundary(
                        substep == self.video_rollout_steps - 1
                    )
                    # Direct engine calls keep all backwards under one optimizer update.
                    self.model.backward(scaled_loss)
                elif self.accelerator is not None:
                    self.accelerator.backward(scaled_loss)
                else:
                    scaled_loss.backward()

                rollout['video_latent'] = next_video_latent
                action_losses.append(loss_dict['action_loss'].detach().float())
                action_sigmas.append(loss_dict['action_timestep_mean'].detach().float())
        finally:
            if self.uses_deepspeed and hasattr(
                self.model, 'set_gradient_accumulation_boundary'
            ):
                self.model.set_gradient_accumulation_boundary(True)

        if not self._synchronize_finite_flag(rollout_losses_finite):
            self._clear_gradients()
            raise FloatingPointError(
                f"Non-finite frozen-rollout loss on at least one rank after "
                f"{self.video_rollout_steps} substeps"
            )

        optimizer_skipped = False
        if self.uses_deepspeed:
            self.model.step()
            optimizer_skipped = bool(self.accelerator.optimizer_step_was_skipped)
            optimizer_step = not optimizer_skipped
            if (
                optimizer_step
                and self.scheduler is not None
                and not self.deepspeed_engine_owns_scheduler
            ):
                self.scheduler.step()
        else:
            grad_clip_norm = (
                self.config.training.grad_clip_norm
                if hasattr(self.config.training, 'grad_clip_norm')
                else 1.0
            )
            if self.accelerator is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip_norm
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=grad_clip_norm
                )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all().item()):
                self._clear_gradients()
                raise FloatingPointError(
                    f"Non-finite gradient norm before frozen-rollout optimizer step: {grad_norm}"
                )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            optimizer_step = True

        stacked_losses = torch.stack(action_losses)
        stacked_sigmas = torch.stack(action_sigmas)
        task_counts = torch.bincount(rollout['video_task_ids'], minlength=5).detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(task_counts, op=torch.distributed.ReduceOp.SUM)
        task_counts = task_counts.cpu()
        action_loss_mean = stacked_losses.mean().item()
        action_loss_weight = float(
            getattr(self._unwrap_training_model().config, 'action_loss_weight', 1.0)
        )
        metrics = {
            'total_loss': action_loss_weight * action_loss_mean,
            'video_loss': 0.0,
            'action_loss': action_loss_mean,
            'action_loss_min': stacked_losses.min().item(),
            'action_loss_max': stacked_losses.max().item(),
            'action_timestep_mean': stacked_sigmas.mean().item(),
            'video_rollout_steps': float(self.video_rollout_steps),
            'optimizer_step': optimizer_step,
            'optimizer_skipped': optimizer_skipped,
        }
        for task_id, count in enumerate(task_counts.tolist()):
            metrics[f'video_task_{task_id}_count'] = float(count)
        return metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run one micro-batch and step only at the accumulation boundary."""
        self.model.train()

        batch_task_names = batch.get('task_name')
        batch_embodiments = batch.get('source_embodiment')
        audit_embodiment = None
        audit_single_count = 0
        audit_bimanual_count = 0
        if self.dexjoco_train_audit:
            if not isinstance(batch_embodiments, list) or not batch_embodiments:
                raise RuntimeError(
                    "DEXJOCO_TRAIN_AUDIT requires source_embodiment metadata"
                )
            unknown_embodiments = {str(value) for value in batch_embodiments if str(value) not in {"single", "bimanual"}}
            if unknown_embodiments:
                raise RuntimeError(f"Unknown DexJoCo embodiment values: {unknown_embodiments}")
            audit_single_count = sum(str(value) == "single" for value in batch_embodiments)
            audit_bimanual_count = sum(str(value) == "bimanual" for value in batch_embodiments)
            audit_embodiment = "single" if audit_single_count >= audit_bimanual_count else "bimanual"
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                counts = torch.tensor([audit_single_count, audit_bimanual_count], dtype=torch.int64, device=self.device)
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
                audit_single_count, audit_bimanual_count = (int(value) for value in counts.tolist())
        batch_task_ids = batch.get('video_task_ids')
        if batch_task_ids is None:
            local_task_counts = torch.zeros(5, dtype=torch.float64)
        else:
            local_task_counts = torch.bincount(
                batch_task_ids.detach().to(device='cpu', dtype=torch.long).reshape(-1),
                minlength=5,
            ).to(dtype=torch.float64)
        h2d_wall_start = time.perf_counter()
        h2d_start_event = None
        h2d_end_event = None
        if self.profile_data_pipeline and self.device.type == 'cuda':
            h2d_start_event = torch.cuda.Event(enable_timing=True)
            h2d_end_event = torch.cuda.Event(enable_timing=True)
            h2d_start_event.record()
        first_frame = batch['first_frame'].to(
            self.device, dtype=self.dtype, non_blocking=True
        )
        video_frames = batch.get('video_frames')
        if video_frames is not None:
            video_frames = video_frames.to(
                self.device, dtype=self.dtype, non_blocking=True
            )
        language_embeddings = batch['language_embedding']
        if language_embeddings is not None:
            language_embeddings = language_embeddings.to(
                self.device, dtype=self.dtype, non_blocking=True
            )
        state = batch.get('initial_state', None)
        if state is not None:
            state = state.to(self.device, dtype=self.dtype, non_blocking=True)
        actions = batch['action_sequence'].to(
            self.device, dtype=self.dtype, non_blocking=True
        )
        action_valid_mask = batch.get('action_valid_mask')
        if action_valid_mask is not None:
            action_valid_mask = action_valid_mask.to(
                self.device, dtype=torch.bool, non_blocking=True
            )
        tactile_batch = {}
        for key in (
            'tactile_observed_source',
            'tactile_future_source',
            'tactile_grid_mask_source',
            'tactile_observed_frame_times',
            'tactile_future_query_times',
        ):
            value = batch.get(key)
            tactile_batch[key] = (
                None
                if value is None
                else value.to(
                    self.device,
                    dtype=(
                        torch.bool
                        if key == 'tactile_grid_mask_source'
                        else torch.float32
                    ),
                    non_blocking=True,
                )
            )
        video_task_ids = batch.get('video_task_ids')
        if video_task_ids is not None:
            video_task_ids = video_task_ids.to(
                self.device, dtype=torch.long, non_blocking=True
            )
        # Handle VLM inputs - it's a Dict[str, Tensor] from collate_fn
        vlm_inputs = batch['vlm_inputs']
        if vlm_inputs is not None:
            # Move all tensors in the VLM inputs dict to device
            vlm_inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in vlm_inputs.items()}
        if h2d_end_event is not None:
            h2d_end_event.record()
            h2d_end_event.synchronize()
            h2d_time = h2d_start_event.elapsed_time(h2d_end_event) / 1000.0
        else:
            h2d_time = time.perf_counter() - h2d_wall_start

        if self.video_training_strategy == 'frozen_rollout':
            metrics = self._train_frozen_wan_rollout(
                first_frame=first_frame,
                state=state,
                actions=actions,
                action_valid_mask=action_valid_mask,
                language_embeddings=language_embeddings,
            )
            metrics['h2d_time'] = h2d_time
            return metrics

        accumulation_context = (
            self.accelerator.accumulate(self.model)
            if self.accelerator is not None
            else nullcontext()
        )
        with accumulation_context:
            if self.dexjoco_train_audit:
                self._audit_grad_square_sums = {'action': None, 'tactile': None}
            # Call the prepared model so DDP/DeepSpeed forward hooks remain active.
            loss_dict = self.model(
                first_frame=first_frame,
                video_frames=video_frames,
                state=state,
                actions=actions,
                action_valid_mask=action_valid_mask,
                language_embeddings=language_embeddings,
                vlm_inputs=vlm_inputs,
                video_task_ids=video_task_ids,
                **tactile_batch,
                return_dict=True,
            )
            action_loss_per_sample = loss_dict.pop('action_loss_per_sample', None)
            total_loss = loss_dict['total_loss']

            local_losses_finite = all(
                not torch.is_tensor(value) or bool(torch.isfinite(value.detach()).all().item())
                for value in loss_dict.values()
            )
            losses_finite = torch.tensor(
                int(local_losses_finite), device=total_loss.device, dtype=torch.int32
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(losses_finite, op=torch.distributed.ReduceOp.MIN)
            if not bool(losses_finite.item()):
                for parameter in self.model.parameters():
                    parameter.grad = None
                self._metric_sums.clear()
                self._micro_batches_in_update = 0
                local_values = {
                    key: value.detach().float().mean().item() if torch.is_tensor(value) else float(value)
                    for key, value in loss_dict.items()
                }
                raise FloatingPointError(
                    f"Non-finite loss detected before backward on rank {self.rank}: {local_values}"
                )

            if self.per_task_action_loss_interval:
                if action_loss_per_sample is None:
                    raise RuntimeError("Model did not return action_loss_per_sample")
                if not isinstance(batch_task_names, list):
                    raise RuntimeError("Batch did not return the RoboTwin task_name list")
                per_sample = action_loss_per_sample.detach().float().cpu().reshape(-1)
                if not bool(torch.isfinite(per_sample).all().item()):
                    raise FloatingPointError("Per-sample action loss contains non-finite values")
                if per_sample.numel() != len(batch_task_names):
                    raise RuntimeError(
                        "Per-sample action loss and task-name batch sizes differ: "
                        f"{per_sample.numel()} != {len(batch_task_names)}"
                    )
                for task_name, value in zip(batch_task_names, per_sample.tolist()):
                    task_index = self.task_name_to_index.get(str(task_name))
                    if task_index is None:
                        raise KeyError(f"Unknown RoboTwin task in batch: {task_name}")
                    self._per_task_action_loss_sums[task_index] += float(value)
                    self._per_task_action_loss_counts[task_index] += 1

            if self.accelerator is not None:
                self.accelerator.backward(total_loss)
                accumulation_boundary = bool(self.accelerator.sync_gradients)
            else:
                total_loss.backward()
                accumulation_boundary = True

            unwrapped_model = self._unwrap_training_model()
            video_model = getattr(unwrapped_model, 'video_model', None)
            wan_model = getattr(video_model, 'wan_model', None)
            task_projection = getattr(wan_model, 'task_text_projection', None)
            if task_projection is not None and task_projection.weight.grad is not None:
                gradient = task_projection.weight.grad.detach().float()
                finite_gradient = torch.where(
                    torch.isfinite(gradient), gradient, torch.zeros_like(gradient)
                )
                loss_dict['task_projection_grad_norm'] = torch.linalg.vector_norm(
                    finite_gradient
                )
                loss_dict['task_projection_grad_nonfinite_count'] = (
                    ~torch.isfinite(gradient)
                ).sum().float()

            if self.dexjoco_train_audit:
                tactile_codec = getattr(unwrapped_model, 'tactile_codec', None)
                for branch in ('action', 'tactile'):
                    squared = self._audit_grad_square_sums[branch]
                    if squared is None:
                        raise RuntimeError(
                            f"No {branch} gradients were observed during backward"
                        )
                    if not bool(torch.isfinite(squared).all().item()):
                        raise FloatingPointError(
                            f"Non-finite {branch} gradient norm during backward"
                        )
                    loss_dict[f'audit_{branch}_grad_norm'] = squared.sqrt()
                codec_model = getattr(tactile_codec, 'model', None)
                codec_has_grad = any(
                    parameter.grad is not None for parameter in codec_model.parameters()
                ) if codec_model is not None else False
                if codec_has_grad:
                    raise RuntimeError("Frozen V3 TacAE unexpectedly received gradients")
                loss_dict['audit_tactile_codec_has_grad'] = total_loss.new_tensor(
                    float(codec_has_grad)
                )
                batch_size = max(1, audit_single_count + audit_bimanual_count)
                loss_dict['audit_token_count'] = total_loss.new_tensor(
                    (5.0 * audit_single_count + 10.0 * audit_bimanual_count) / batch_size
                )
                loss_dict['audit_single_step'] = total_loss.new_tensor(
                    float(audit_single_count) / batch_size
                )
                loss_dict['audit_bimanual_step'] = total_loss.new_tensor(
                    float(audit_bimanual_count) / batch_size
                )

            optimizer_skipped = bool(
                accumulation_boundary
                and self.accelerator is not None
                and self.uses_deepspeed
                and self.accelerator.optimizer_step_was_skipped
            )
            optimizer_step = accumulation_boundary and not optimizer_skipped

            if (
                optimizer_step
                and self.uses_deepspeed
                and self.scheduler is not None
                and not self.deepspeed_engine_owns_scheduler
            ):
                # Accelerate 1.14 does not attach arbitrary custom schedulers to
                # DeepSpeedEngine. Advance those schedulers after a finite engine step.
                self.scheduler.step()

            if accumulation_boundary and not self.uses_deepspeed:
                grad_clip_norm = (
                    self.config.training.grad_clip_norm
                    if hasattr(self.config.training, 'grad_clip_norm')
                    else 1.0
                )
                if self.accelerator is not None:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip_norm
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=grad_clip_norm
                    )

                local_grad_finite = bool(
                    torch.isfinite(torch.as_tensor(grad_norm, device=total_loss.device)).all().item()
                )
                gradients_finite = torch.tensor(
                    int(local_grad_finite), device=total_loss.device, dtype=torch.int32
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        gradients_finite, op=torch.distributed.ReduceOp.MIN
                    )
                if not bool(gradients_finite.item()):
                    for parameter in self.model.parameters():
                        parameter.grad = None
                    self._metric_sums.clear()
                    self._micro_batches_in_update = 0
                    raise FloatingPointError(
                        f"Non-finite gradient norm detected before optimizer step on rank "
                        f"{self.rank}: {float(torch.as_tensor(grad_norm).detach().float().cpu())}"
                    )

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        raw_metrics = {
            key: value.item() if torch.is_tensor(value) else float(value)
            for key, value in loss_dict.items()
        }
        raw_metrics['h2d_time'] = h2d_time
        for task_id, count in enumerate(local_task_counts.tolist()):
            raw_metrics[f'video_task_{task_id}_count'] = count
        self._micro_batches_in_update += 1
        for key, value in raw_metrics.items():
            self._metric_sums[key] = self._metric_sums.get(key, 0.0) + value

        if optimizer_skipped:
            self._metric_sums.clear()
            self._micro_batches_in_update = 0
            return {
                **raw_metrics,
                'optimizer_step': False,
                'optimizer_skipped': True,
            }

        if not optimizer_step:
            return {
                **raw_metrics,
                'optimizer_step': False,
                'optimizer_skipped': False,
            }

        metrics = {
            key: (
                value
                if key.startswith('video_task_') and key.endswith('_count')
                else value / self._micro_batches_in_update
            )
            for key, value in self._metric_sums.items()
        }
        global_task_counts = torch.tensor(
            [metrics[f'video_task_{task_id}_count'] for task_id in range(5)],
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_task_counts, op=torch.distributed.ReduceOp.SUM)
        for task_id, count in enumerate(global_task_counts.cpu().tolist()):
            metrics[f'video_task_{task_id}_count'] = count
        metrics['micro_batches'] = self._micro_batches_in_update
        metrics['optimizer_step'] = True
        metrics['optimizer_skipped'] = False
        self._metric_sums.clear()
        self._micro_batches_in_update = 0
        return metrics

    def log_per_task_action_losses(self) -> None:
        """Reduce and log a compact per-task window without changing training."""
        if not self.per_task_action_loss_interval:
            return
        loss_sums = self._per_task_action_loss_sums.to(self.device, dtype=torch.float32)
        sample_counts = self._per_task_action_loss_counts.to(
            self.device, dtype=torch.float32
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sample_counts, op=torch.distributed.ReduceOp.SUM)
        missing = [
            task_name
            for task_name, count in zip(self.task_names, sample_counts.tolist())
            if count <= 0
        ]
        if missing:
            raise RuntimeError(
                "Per-task action-loss window contains unsampled tasks: "
                f"{missing}"
            )
        means = loss_sums / sample_counts
        if not bool(torch.isfinite(means).all().item()):
            raise FloatingPointError("Per-task action-loss window contains non-finite values")
        if self.rank == 0:
            if self.tb_writer is not None:
                for task_name, value in zip(self.task_names, means.tolist()):
                    self.tb_writer.add_scalar(
                        f'train_per_task/action_loss/{task_name}',
                        value,
                        self.global_step,
                    )
                self.tb_writer.add_histogram(
                    'train_per_task/sample_counts', sample_counts.cpu(), self.global_step
                )
                self.tb_writer.add_scalar(
                    'train_per_task/sample_count_min',
                    sample_counts.min().item(),
                    self.global_step,
                )
                self.tb_writer.add_scalar(
                    'train_per_task/sample_count_max',
                    sample_counts.max().item(),
                    self.global_step,
                )
            logger.info(
                "PER_TASK_ACTION_LOSS step=%s window=%s count_min=%.0f count_max=%.0f",
                self.global_step,
                self.per_task_action_loss_interval,
                sample_counts.min().item(),
                sample_counts.max().item(),
            )
        self._per_task_action_loss_sums.zero_()
        self._per_task_action_loss_counts.zero_()

    def train(self, max_steps: int, resume_from: Optional[str] = None, val_interval: int = 500, reset_scheduler: Optional[bool] = None):
        """
        Main training loop.

        Args:
            max_steps: Maximum number of training steps
            resume_from: Path to checkpoint to resume from
            val_interval: Validation interval in steps
            reset_scheduler: If True, reset scheduler to new config. If None, use config.resume.reset_scheduler
        """
        # Load checkpoint if specified
        if resume_from:
            # Determine whether to reset scheduler
            if reset_scheduler is None:
                # Use config value if available, otherwise default to True
                if self.config is not None and hasattr(self.config, 'resume') and hasattr(self.config.resume, 'reset_scheduler'):
                    reset_scheduler = bool(self.config.resume.reset_scheduler)
                else:
                    reset_scheduler = True  # Default behavior

            self.load_checkpoint(resume_from, reset_scheduler=reset_scheduler)

        logger.info(f"Starting UniDiffuser training for {max_steps} steps")
        if self.disable_python_gc and gc.isenabled():
            gc.disable()
            logger.info("Disabled cyclic Python GC in the training process")

        start_time = time.time()
        initial_step = self.global_step

        # Step-based training loop
        epoch = int(getattr(self.train_dataloader, "iteration", self.epoch))
        dataset = getattr(self.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        data_iter = iter(self.train_dataloader)
        while self.global_step < max_steps:
            step_wall_start = time.perf_counter()
            step_wall_start_unix = time.time()
            data_wait_start = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                # End of epoch, restart dataloader
                epoch = int(getattr(self.train_dataloader, "iteration", epoch + 1))
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(epoch)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            data_wait_time = time.perf_counter() - data_wait_start

            if batch is None:  # Handle None batches
                continue
            component_times = batch.pop('_data_component_times', None)
            if component_times is None:
                component_time_maxima = [0.0, 0.0, 0.0, 0.0]
            else:
                component_time_maxima = (
                    component_times.max(dim=0).values.detach().cpu().tolist()
                )
            raw_component_times = batch.pop('_raw_data_component_times', None)
            if raw_component_times is None:
                raw_component_time_maxima = [0.0] * 6
            else:
                raw_component_time_maxima = (
                    raw_component_times.max(dim=0).values.detach().cpu().tolist()
                )
            raw_worker_resources = batch.pop('_raw_worker_resource_metrics', None)
            if raw_worker_resources is None:
                raw_worker_resource_maxima = [0.0] * 4
                worker_resource_metrics_present = 0.0
            else:
                raw_worker_resource_maxima = (
                    raw_worker_resources.max(dim=0).values.detach().cpu().tolist()
                )
                worker_resource_metrics_present = 1.0

            compute_wall_start = time.perf_counter()
            profile_gpu = torch.cuda.is_available() and (self.global_step + 1) % self.log_interval == 0
            gpu_start = torch.cuda.Event(enable_timing=True) if profile_gpu else None
            gpu_end = torch.cuda.Event(enable_timing=True) if profile_gpu else None
            if gpu_start is not None:
                gpu_start.record()
            metrics = self.train_step(batch)
            if gpu_end is not None:
                gpu_end.record()
                gpu_end.synchronize()
                gpu_compute_time = gpu_start.elapsed_time(gpu_end) / 1000.0
            else:
                gpu_compute_time = None
            compute_wall_time = time.perf_counter() - compute_wall_start
            optimizer_step = bool(metrics.pop('optimizer_step'))
            optimizer_skipped = bool(metrics.pop('optimizer_skipped', False))
            if optimizer_skipped:
                self.optimizer_skipped_steps += 1
                if self.rank == 0:
                    logger.warning(
                        "DeepSpeed skipped optimizer update due to BF16 gradient overflow: "
                        "skipped_total=%s data_wait=%.3fs compute_wall=%.3fs",
                        self.optimizer_skipped_steps,
                        data_wait_time,
                        compute_wall_time,
                    )
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        'train/optimizer_skipped_steps',
                        self.optimizer_skipped_steps,
                        self.global_step,
                    )
                continue
            if not optimizer_step:
                continue

            self.global_step += 1
            step_modality_counts = [
                int(round(metrics.get(f'video_task_{task_id}_count', 0.0)))
                for task_id in range(5)
            ]
            self._run_video_modality_counts = [
                total + current
                for total, current in zip(
                    self._run_video_modality_counts, step_modality_counts
                )
            ]
            step_wall_time = time.perf_counter() - step_wall_start
            if self.global_step == self.performance_warmup_steps + 1:
                self._performance_measurement_start_unix = step_wall_start_unix
            metrics.update({
                'data_wait_time': data_wait_time,
                'compute_wall_time': compute_wall_time,
                'step_wall_time': step_wall_time,
                'optimizer_skipped_steps': self.optimizer_skipped_steps,
            })
            if gpu_compute_time is not None:
                metrics['gpu_compute_time'] = gpu_compute_time
            if self.global_step == 1 or self.global_step % self.memory_log_interval == 0:
                self._last_memory_metrics = self.collect_cuda_memory_metrics()
            metrics.update(self._last_memory_metrics)
            self._timing_window.append([
                float(self.global_step),
                data_wait_time,
                compute_wall_time,
                step_wall_time,
                *component_time_maxima,
                *raw_component_time_maxima,
                float(metrics.get('h2d_time', 0.0)),
                *raw_worker_resource_maxima,
                worker_resource_metrics_present,
            ])
            metrics.update(self.collect_due_timing_window_metrics())
            if (
                self.per_task_action_loss_interval
                and self.global_step % self.per_task_action_loss_interval == 0
            ):
                self.log_per_task_action_losses()
            if self.global_step == self.performance_measurement_steps:
                self._performance_measurement_end_unix = time.time()
                self.write_performance_summary()

            # Logging
            if (
                self.global_step % self.log_interval == 0
                or self.dexjoco_train_audit
            ) and self.rank == 0:
                # Log optimizer LR as authoritative (may differ from inner scheduler get_last_lr)
                lr_by_name = {
                    str(group.get('name', f'group_{index}')): float(group['lr'])
                    for index, group in enumerate(self.optimizer.param_groups)
                }
                lr_main = lr_by_name.get('main', next(iter(lr_by_name.values()), 0.0))
                lr_wan = lr_by_name.get('wan', lr_main)
                lr_tactile = lr_by_name.get('tactile')
                lr_task_projection = lr_by_name.get('task_projection', 0.0)
                tactile_log = ""
                if lr_tactile is not None:
                    tactile_log = (
                        f", Tactile: {metrics['tactile_loss']:.4f}; "
                        f"Sigma(V/A/T): {metrics['video_timestep_mean']:.3f}/"
                        f"{metrics['action_timestep_mean']:.3f}/"
                        f"{metrics['tactile_timestep_mean']:.3f}"
                    )
                lr_names = "main/wan/tactile/task" if lr_tactile is not None else "main/wan/task"
                lr_values = (
                    f"{lr_main:.2e}/{lr_wan:.2e}/{lr_tactile:.2e}/{lr_task_projection:.2e}"
                    if lr_tactile is not None
                    else f"{lr_main:.2e}/{lr_wan:.2e}/{lr_task_projection:.2e}"
                )
                run_elapsed = max(time.time() - start_time, 1e-6)
                run_steps = self.global_step - initial_step
                run_speed = run_steps / run_elapsed
                eta_seconds = (max_steps - self.global_step) / run_speed if run_speed > 0 else float('inf')

                log_str = (
                    f"Step {self.global_step}/{max_steps}, "
                    f"Loss: {metrics['total_loss']:.4f} "
                    f"(Video: {metrics['video_loss']:.4f}, Action: {metrics['action_loss']:.4f}"
                    f"{tactile_log}), "
                    f"LR({lr_names}): {lr_values}, "
                    f"Data/Compute/Wall: {data_wait_time:.2f}/{compute_wall_time:.2f}/{step_wall_time:.2f}s, "
                    f"Data p95/max: {metrics.get('data_wait_time_window_p95', 0.0):.2f}/"
                    f"{metrics.get('data_wait_time_window_max', 0.0):.2f}s, "
                    f"Wall mean/p95: {metrics.get('step_wall_time_window_mean', 0.0):.2f}/"
                    f"{metrics.get('step_wall_time_window_p95', 0.0):.2f}s, "
                    f"IO video/qpos/lang max: "
                    f"{metrics.get('video_decode_time_window_max', 0.0):.2f}/"
                    f"{metrics.get('qpos_load_time_window_max', 0.0):.2f}/"
                    f"{metrics.get('language_load_time_window_max', 0.0):.2f}s, "
                    f"CUDA peak alloc/reserved: {metrics.get('cuda_peak_allocated_gib', 0.0):.2f}/"
                    f"{metrics.get('cuda_peak_reserved_gib', 0.0):.2f} GiB, "
                    f"ETA: {_format_duration(eta_seconds) if math.isfinite(eta_seconds) else 'n/a'} "
                    f"(speed={run_speed:.3f} step/s)"
                )
                logger.info(log_str)
                if self.dexjoco_train_audit:
                    single_fraction = metrics.get('audit_single_step', 0.0)
                    bimanual_fraction = metrics.get('audit_bimanual_step', 0.0)
                    self._dexjoco_embodiment_counts['single'] += single_fraction
                    self._dexjoco_embodiment_counts['bimanual'] += bimanual_fraction
                    logger.info(
                        "DEXJOCO_TRAIN_AUDIT step=%s batch_single_fraction=%.3f batch_bimanual_fraction=%.3f tokens=%.2f "
                        "action_loss=%.6f tactile_loss=%.6f "
                        "action_grad_norm=%.6f tactile_grad_norm=%.6f "
                        "tacae_has_grad=%d cumulative_single=%d cumulative_bimanual=%d",
                        self.global_step,
                        single_fraction,
                        bimanual_fraction,
                        metrics.get('audit_token_count', 0.0),
                        metrics['action_loss'],
                        metrics['tactile_loss'],
                        metrics.get('audit_action_grad_norm', 0.0),
                        metrics.get('audit_tactile_grad_norm', 0.0),
                        int(round(metrics.get('audit_tactile_codec_has_grad', 0.0))),
                        self._dexjoco_embodiment_counts['single'],
                        self._dexjoco_embodiment_counts['bimanual'],
                    )
                logger.info(
                    "VIDEO_MODALITY_COUNTS step=%s step_counts=%s cumulative_counts=%s",
                    self.global_step,
                    step_modality_counts,
                    self._run_video_modality_counts,
                )

                # Log to WandB
                if "wandb" in self.report_to:
                    wandb.log({
                        **metrics,
                        'learning_rate_main': lr_main,
                        'learning_rate_wan': lr_wan,
                        **(
                            {'learning_rate_tactile': lr_tactile}
                            if lr_tactile is not None
                            else {}
                        ),
                        'learning_rate_task_projection': lr_task_projection,
                        'step_wall_time': step_wall_time,
                        'epoch': epoch,
                        'global_step': self.global_step,
                        })

                # Log to TensorBoard
                if self.tb_writer is not None:
                    for key, value in metrics.items():
                        self.tb_writer.add_scalar(f'train/{key}', value, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_main', lr_main, self.global_step)
                    self.tb_writer.add_scalar('train/learning_rate_wan', lr_wan, self.global_step)
                    if lr_tactile is not None:
                        self.tb_writer.add_scalar(
                            'train/learning_rate_tactile',
                            lr_tactile,
                            self.global_step,
                        )
                    self.tb_writer.add_scalar(
                        'train/learning_rate_task_projection',
                        lr_task_projection,
                        self.global_step,
                    )
                    self.tb_writer.add_scalar('train/epoch', epoch, self.global_step)

            # Validation: rank0-only local eval; then synchronize all processes
            if self.global_step % val_interval == 0 and self.val_dataloader is not None:
                if self.rank == 0:
                    eval_model = (
                        self.accelerator.unwrap_model(self.model)
                        if self.accelerator is not None
                        else self.model
                    )
                    val_metrics = evaluate_model(
                        eval_model, self.val_dataloader, self.accelerator, self.config,
                        num_eval_batches=2
                    )
                    logger.info(f"Validation - Step {self.global_step}")
                    log_evaluation_metrics(val_metrics, self.tb_writer, self.accelerator, self.global_step)
                # Use explicit barrier with device_ids to avoid NCCL warnings
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier(device_ids=[torch.cuda.current_device()])
                    except TypeError:
                        # Fallback for older PyTorch versions without device_ids argument
                        dist.barrier()

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint()

        total_time = time.time() - start_time
        if self.rank == 0:
            logger.info(f"UniDiffuser training completed in {total_time:.2f}s ({self.global_step} steps)")
        self.write_training_run_summary(
            requested_max_steps=max_steps,
            elapsed_seconds=total_time,
        )
        if self.save_final_checkpoint and self.global_step % self.save_interval != 0:
            self.save_checkpoint()
        final_memory = self.collect_cuda_memory_metrics()
        if self.rank == 0 and final_memory:
            logger.info(
                "Final CUDA max across ranks: allocated=%.2f GiB reserved=%.2f GiB",
                final_memory['cuda_peak_allocated_gib'],
                final_memory['cuda_peak_reserved_gib'],
            )

def create_model_and_optimizer(config: OmegaConf) -> tuple:
    """Create UniDiffuser model and optimizer from config."""
    # Create Motus config
    tactile_config = config.model.get('tactile', {})
    tactile_enabled = bool(tactile_config.get('enabled', False))
    model_config = MotusConfig(
        wan_checkpoint_path=config.model.wan.checkpoint_path,
        wan_expected_num_tensors=config.model.wan.get('expected_num_tensors', None),
        wan_task_embedding_path=config.model.wan.get('task_embedding_path', None),
        wan_task_text_conditioning_path=config.model.wan.get('task_text_conditioning_path', None),
        wan_task_id=config.model.wan.get('task_id', None),
        wan_task_probabilities=config.model.wan.get('task_probabilities', None),
        wan_frozen=bool(config.model.wan.get('frozen', False)),
        video_training_strategy=config.model.wan.get('training_strategy', 'flow_matching'),
        video_rollout_steps=config.model.wan.get('rollout_steps', 10),
        vae_path=config.model.wan.vae_path,
        wan_config_path=config.model.wan.config_path,
        vlm_checkpoint_path=config.model.vlm.checkpoint_path,
        video_precision=config.model.wan.precision,
        action_state_dim=config.common.state_dim,
        action_dim=config.common.action_dim,
        # Action Expert configuration from config file
        action_expert_dim=config.model.action_expert.hidden_size,
        action_expert_ffn_dim_multiplier=config.model.action_expert.ffn_dim_multiplier,
        action_expert_norm_eps=config.model.action_expert.norm_eps,
        # Understanding Expert configuration from config file
        und_expert_hidden_size=config.model.und_expert.hidden_size,
        und_expert_ffn_dim_multiplier=config.model.und_expert.ffn_dim_multiplier,
        und_expert_norm_eps=config.model.und_expert.norm_eps,
        vlm_adapter_input_dim=config.model.und_expert.vlm.input_dim,
        vlm_adapter_projector_type=config.model.und_expert.vlm.projector_type,
        global_downsample_rate=config.common.global_downsample_rate,
        video_action_freq_ratio=config.common.video_action_freq_ratio,
        num_video_frames=config.common.num_video_frames,
        action_chunk_size=config.common.action_chunk_size,
        # Video dimensions from config
        video_height=config.common.video_height,
        video_width=config.common.video_width,
        batch_size=config.training.batch_size,
        video_loss_weight=config.model.loss_weights.video_loss_weight,
        action_loss_weight=config.model.loss_weights.action_loss_weight,
        per_sample_valid_mean_loss=bool(
            config.model.loss_weights.get('per_sample_valid_mean_loss', False)
        ),
        masked_zero_padding=bool(
            config.model.get('padding_contract', {}).get('masked_zero_padding', False)
        ),
        pi05_zero_padding=bool(
            config.model.get('padding_contract', {}).get('pi05_zero_padding', False)
        ),
        timestep_sample_method=config.model.time_distribution.get('timestep_sample_method', 'logit_normal'),
        sigmoid_scale=config.model.time_distribution.get('sigmoid_scale', 1.0),
        min_t=config.model.time_distribution.get('min_t', 0.0),
        max_t=config.model.time_distribution.get('max_t', 1.0),
        video_train_schedule_shift=config.model.time_distribution.get('video_train_schedule_shift', 5.0),
        action_train_schedule_shift=config.model.time_distribution.get('action_train_schedule_shift', 5.0),
        tactile_train_schedule_shift=config.model.time_distribution.get('tactile_train_schedule_shift', 5.0),
        joint_attention_mode=config.model.get(
            'joint_attention_mode', 'full_bidirectional'
        ),
        training_mode=getattr(config, 'training_mode', 'finetune'),
        load_pretrained_backbones=getattr(config.model, 'load_pretrained_backbones', None),
        use_understanding=bool(config.model.get('use_understanding', True)),
        tactile_enabled=tactile_enabled,
        tactile_loss_weight=config.model.loss_weights.get('tactile_loss_weight', 0.1),
        tactile_vae_checkpoint_path=tactile_config.get('checkpoint_path', None),
        tactile_vae_expected_sha256=tactile_config.get('expected_sha256', None),
        tactile_codec_type=tactile_config.get('codec_type', 'v21_support_vae'),
        tactile_codec_source_root=tactile_config.get('source_root', None),
        tactile_codec_source_revision=tactile_config.get('source_revision', None),
        tactile_codec_source_files_sha256=(
            OmegaConf.to_container(
                tactile_config.get('source_files_sha256'), resolve=True
            )
            if tactile_config.get('source_files_sha256', None) is not None
            else None
        ),
        tactile_codec_expected_step=tactile_config.get('expected_step', None),
        tactile_codec_normalization_group=tactile_config.get(
            'normalization_group', None
        ),
        tactile_dexjoco_channel_scale=(
            tuple(tactile_config.get('dexjoco_channel_scale'))
            if tactile_config.get('dexjoco_channel_scale', None) is not None
            else None
        ),
        tactile_latent_stats_path=tactile_config.get('latent_stats_path', None),
        tactile_latent_stats_expected_sha256=tactile_config.get('expected_latent_stats_sha256', None),
        tactile_support_calibration_path=tactile_config.get('support_calibration_path', None),
        tactile_support_calibration_expected_sha256=tactile_config.get('expected_support_calibration_sha256', None),
        tactile_latent_stats_task_name=tactile_config.get(
            'latent_stats_task_name', 'robotwin_vat_supportvae_sparse_multitask_v1'
        ),
        tactile_expert_config=(
            OmegaConf.to_container(config.model.tactile_expert, resolve=True)
            if tactile_enabled
            else None
        ),
    )

    # Create model (Accelerator will handle device placement and DDP)
    logger.info(
        "Loss/padding contract: per_sample_valid_mean=%s masked_zero=%s pi05_zero=%s "
        "action_weight=%s tactile_weight=%s",
        model_config.per_sample_valid_mean_loss,
        model_config.masked_zero_padding,
        model_config.pi05_zero_padding,
        model_config.action_loss_weight,
        model_config.tactile_loss_weight,
    )
    model = Motus(model_config)

    # Optimizer - parameter groups for separate WAN (video model) learning rate
    base_lr = float(config.training.learning_rate)
    wan_lr = float(getattr(config.training, 'wan_learning_rate', base_lr))
    task_projection_lr = float(
        getattr(config.training, 'task_projection_learning_rate', wan_lr)
    )

    # Collect WAN params explicitly (exclude VAE, we only train diffusion WAN)
    task_projection = getattr(model.video_model.wan_model, 'task_text_projection', None)
    task_projection_params = (
        [parameter for parameter in task_projection.parameters() if parameter.requires_grad]
        if task_projection is not None
        else []
    )
    task_projection_param_ids = {id(parameter) for parameter in task_projection_params}
    wan_params = [
        parameter
        for parameter in model.video_model.wan_model.parameters()
        if parameter.requires_grad and id(parameter) not in task_projection_param_ids
    ]
    tactile_params = (
        [parameter for parameter in model.tactile_expert.parameters() if parameter.requires_grad]
        if tactile_enabled
        else []
    )
    tactile_param_ids = {id(parameter) for parameter in tactile_params}
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    wan_param_ids = {id(p) for p in wan_params} | task_projection_param_ids
    other_params = [
        p for p in all_trainable
        if id(p) not in wan_param_ids and id(p) not in tactile_param_ids
    ]

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({'name': 'main', 'params': other_params, 'lr': base_lr})
    if len(wan_params) > 0:
        param_groups.append({'name': 'wan', 'params': wan_params, 'lr': wan_lr})
    if len(tactile_params) > 0:
        param_groups.append(
            {
                'name': 'tactile',
                'params': tactile_params,
                'lr': float(config.training.get('tactile_learning_rate', base_lr)),
            }
        )
    if len(task_projection_params) > 0:
        param_groups.append(
            {
                'name': 'task_projection',
                'params': task_projection_params,
                'lr': task_projection_lr,
            }
        )

    # Foreach AdamW is faster but creates a sizeable temporary tensor list at
    # the first optimizer step. Allow a gate or constrained host to disable
    # that peak without changing the optimizer algorithm or its state.
    optimizer_foreach = config.training.get('optimizer_foreach', None)
    if optimizer_foreach is not None:
        optimizer_foreach = bool(optimizer_foreach)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
        foreach=optimizer_foreach,
    )
    if bool(config.model.wan.get('frozen', False)):
        all_wan_param_ids = {id(parameter) for parameter in model.video_model.wan_model.parameters()}
        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        overlap = all_wan_param_ids & optimizer_param_ids
        if overlap:
            raise RuntimeError(
                f"Frozen WAN leaked {len(overlap)} parameters into the optimizer"
            )
        logger.info(
            "Frozen WAN optimizer audit passed: wan_parameters=%s optimizer_overlap=0",
            len(all_wan_param_ids),
        )

    # Scheduler
    scheduler = create_scheduler(optimizer, config)

    return model, optimizer, scheduler

def create_dataloaders(config: OmegaConf, rank: int, world_size: int) -> tuple:
    """Create train and validation dataloaders from config."""
    train_dataset = create_dataset(config, val=False)
    val_enabled = bool(config.dataset.get('params', {}).get('val_enabled', True))
    val_dataset = create_dataset(config, val=True) if val_enabled else None
    if not val_enabled:
        logger.info("Validation disabled: val_enabled=false; validation dataloader will not be created")
    logger.info(
        "DexJoCo train sampling: task_sampling=%s sampling_mode=%s "
        "va_uniform_window_sampling=%s windows_per_task=%s tasks=%s dataset_len=%s",
        getattr(train_dataset, "task_sampling", "n/a"),
        getattr(train_dataset, "sampling_mode", "n/a"),
        getattr(train_dataset, "va_uniform_window_sampling", False),
        getattr(train_dataset, "_va_uniform_windows_per_task", None),
        getattr(train_dataset, "task_names", "n/a"),
        len(train_dataset),
    )

    num_workers = int(config.system.num_workers)
    loader_worker_options = {}
    if num_workers > 0:
        loader_worker_options = {
            'prefetch_factor': int(config.system.get('prefetch_factor', 2)),
            'persistent_workers': bool(config.system.get('persistent_workers', False)),
            'worker_init_fn': _worker_init_fn,
        }

    # Build one complete shuffled loader. Accelerate.prepare() performs the
    # sole batch sharding, matching the RoboTwin reference implementation.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        **loader_worker_options,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.system.num_workers,
            pin_memory=config.system.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
            **loader_worker_options,
        )

    return train_dataloader, val_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train Three-Modal UniDiffuser Model")

    # Configuration file
    parser.add_argument("--config", type=str,
                       default="configs/aloha_agilex_2.yaml",
                       help="Path to configuration file")

    # System settings
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum optimizer steps")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume the complete native training state from a checkpoint directory",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")

    # Logging settings
    parser.add_argument("--report_to", type=str, default=None,
                       choices=["wandb", "tensorboard", "all", "none"],
                       help="Logging backends to use")
    parser.add_argument("--wandb_project", type=str, default=None, help="Override WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="Override run name")

    # DeepSpeed settings
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if config.model.wan.get('training_strategy', 'flow_matching') == 'frozen_rollout':
        if int(config.training.get('gradient_accumulation_steps', 1)) != 1:
            raise ValueError("frozen_rollout requires gradient_accumulation_steps=1")
        if float(config.model.loss_weights.video_loss_weight) != 0.0:
            raise ValueError("frozen_rollout requires video_loss_weight=0")
        if not bool(config.model.wan.get('frozen', False)):
            raise ValueError("frozen_rollout requires model.wan.frozen=true")
        if not bool(config.dataset.get('condition_only', False)):
            raise ValueError("frozen_rollout requires dataset.condition_only=true")
    if bool(getattr(config.system, 'disable_python_gc', False)):
        os.environ["MOTUS_DISABLE_PYTHON_GC"] = "1"
    else:
        os.environ.pop("MOTUS_DISABLE_PYTHON_GC", None)
    if args.deepspeed is not None and int(config.training.get('gradient_accumulation_steps', 1)) > 1:
        raise ValueError(
            "DeepSpeed with gradient_accumulation_steps > 1 is disabled: the pinned "
            "Accelerate 1.14.0 / DeepSpeed 0.18.3 ZeRO-1 path failed the global-batch "
            "equivalence test. Use the validated DDP launcher."
        )
    if args.checkpoint_dir is not None:
        config.system.checkpoint_dir = args.checkpoint_dir
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {args.max_steps}")
        config.training.max_steps = args.max_steps
    if args.resume_from is not None:
        resume_path = Path(args.resume_from).expanduser().resolve()
        if not resume_path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint directory does not exist: {resume_path}")
        config.resume.checkpoint_path = str(resume_path)
    if args.report_to is not None:
        config.logging.report_to = args.report_to
    if args.wandb_project is not None:
        config.logging.wandb_project = args.wandb_project
    if args.run_name is not None:
        config.logging.run_name = args.run_name
    # Decide backbone loading policy:
    # If resuming or finetuning from a pretrain checkpoint, skip loading WAN/VLM pretrained weights.
    try:
        if (getattr(config.resume, 'checkpoint_path', None) or
            (hasattr(config, 'finetune') and getattr(config.finetune, 'checkpoint_path', None))):
            config.model.load_pretrained_backbones = False
    except Exception:
        pass

    # Extract dataset name from config file path for checkpoint organization
    config_filename = os.path.basename(args.config)  # e.g., "ac_one.yaml"
    dataset_name = os.path.splitext(config_filename)[0]  # e.g., "ac_one"

    # Update checkpoint directory to include dataset name
    base_checkpoint_dir = config.system.checkpoint_dir
    config.system.checkpoint_dir = os.path.join(base_checkpoint_dir, dataset_name)

    # Create the dataset directory if it doesn't exist
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)

    # Initialize Accelerator with DeepSpeed (if provided)
    checkpoint_total_limit = config.system.get('checkpoint_total_limit', 20)
    accelerator_project_config = ProjectConfiguration(
        total_limit=None if checkpoint_total_limit is None else int(checkpoint_total_limit)
    )
    accelerator_log_with = config.logging.get('report_to', 'tensorboard')
    # ``none`` is our CLI/config sentinel.  Accelerate expects Python None,
    # not the literal tracker name "none".
    if accelerator_log_with == 'none':
        accelerator_log_with = None
    dataset_params = config.dataset.get('params', {})
    va_uniform_window_sampling = bool(
        dataset_params.get('va_uniform_window_sampling', False)
    )
    sampling_seed = int(dataset_params.get('sampling_seed', 42))
    dataloader_config = DataLoaderConfiguration(
        use_seedable_sampler=va_uniform_window_sampling,
        data_seed=sampling_seed if va_uniform_window_sampling else None,
    )
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=config.training.get('gradient_accumulation_steps', 1),
        mixed_precision="bf16",
        log_with=accelerator_log_with,
        project_dir=config.system.checkpoint_dir,
        project_config=accelerator_project_config,
        dataloader_config=dataloader_config,
    )

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank, args.log_level)
    actual_global_batch = validate_global_batch_size(config, world_size)
    logger.info(
        "Global batch: per_device=%s world_size=%s accumulation=%s total=%s",
        config.training.batch_size,
        world_size,
        config.training.get('gradient_accumulation_steps', 1),
        actual_global_batch,
    )

    # Handle report_to settings - expand "all" to individual backends
    report_to = config.logging.report_to
    if report_to == "all":
        report_to = ["wandb", "tensorboard"]
    elif report_to == "none":
        report_to = []
    elif isinstance(report_to, str):
        report_to = [report_to]

    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.logging.get('run_name', None)
    if not run_name:
        run_name = f"unidiffuser_{config.dataset.type}_bs{config.training.batch_size}_lr{config.training.learning_rate}"

    # Update checkpoint directory to include run name
    config.system.checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Checkpoints will be saved to: {config.system.checkpoint_dir}")

    # Initialize TensorBoard writer
    tb_writer = None
    if rank == 0 and "tensorboard" in report_to:
        tb_log_dir = os.path.join(config.system.checkpoint_dir, config.logging.tensorboard_log_dir)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        config_dict = OmegaConf.to_container(config, resolve=True)
        tb_writer.add_text('config', yaml.dump(config_dict))

    # Initialize WandB
    if rank == 0 and "wandb" in report_to:
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )

    try:
        # Create model and optimizer
        logger.info("Creating UniDiffuser model and optimizer...")
        model, optimizer, scheduler = create_model_and_optimizer(config)
        if (
            scheduler is not None
            and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)
            and hasattr(scheduler, 'state_dict')
            and hasattr(scheduler, 'load_state_dict')
        ):
            accelerator.register_for_checkpointing(scheduler)
            logger.info(
                "Registered custom scheduler for checkpointing: %s",
                type(scheduler).__name__,
            )

        # Optional: load Stage 2 weights under the strict RGB multiview contract.
        warmstart = config.get('warmstart', {})
        warmstart_mode = warmstart.get('mode', None)
        resume_checkpoint = config.resume.get('checkpoint_path', None)
        if resume_checkpoint:
            logger.info(
                "Native resume requested; skipping Stage 2/finetune initialization before state restore: %s",
                resume_checkpoint,
            )
        elif warmstart_mode in (
            'motus_stage2_selective',
            'motus_stage2_action_only',
            'motus_stage2_wan_only',
            'va_checkpoint_to_vat',
        ):
            if getattr(config, 'training_mode', 'finetune') != 'finetune':
                raise ValueError("Model warm-start requires training_mode=finetune")
            target_model = model.module if hasattr(model, 'module') else model
            report_dir = Path(config.system.checkpoint_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = (
                report_dir / warmstart.get('report_filename', 'stage2_warmstart_load_report.json')
                if rank == 0
                else None
            )
            if warmstart_mode == 'va_checkpoint_to_vat':
                if getattr(config.finetune, 'checkpoint_path', None):
                    raise ValueError(
                        "va_checkpoint_to_vat and finetune.checkpoint_path are mutually exclusive"
                    )
                report = target_model.load_va_checkpoint_for_vat(
                    warmstart.checkpoint_path,
                    report_path=str(report_path) if report_path is not None else None,
                    expected_sha256=(
                        warmstart.get('expected_sha256', None) if rank == 0 else None
                    ),
                    compute_sha256=(
                        bool(warmstart.get('compute_sha256', False)) if rank == 0 else False
                    ),
                    expected_wan_key_count=warmstart.get('expected_wan_key_count', None),
                    expected_action_key_count=warmstart.get('expected_action_key_count', None),
                )
                logger.info(
                    "Strict VA->VAT warm-start PASS: WAN=%s Action=%s "
                    "fresh_tactile=%s optimizer=fresh scheduler=fresh global_step=0 report=%s",
                    report["loaded_wan_key_count"],
                    report["loaded_action_key_count"],
                    report["fresh_tactile_key_count"],
                    report_path,
                )
            elif warmstart_mode == 'motus_stage2_action_only':
                report = target_model.load_stage2_action_only_weights(
                    warmstart.checkpoint_path,
                    report_path=str(report_path) if report_path is not None else None,
                    expected_sha256=warmstart.get('expected_sha256', None) if rank == 0 else None,
                    compute_sha256=(rank == 0),
                )
            elif warmstart_mode == 'motus_stage2_wan_only':
                report = target_model.load_stage2_wan_only_weights(
                    warmstart.checkpoint_path,
                    report_path=str(report_path) if report_path is not None else None,
                    expected_sha256=warmstart.get('expected_sha256', None) if rank == 0 else None,
                    compute_sha256=(rank == 0),
                )
            else:
                report = target_model.load_stage2_warmstart_weights(
                    warmstart.checkpoint_path,
                    report_path=str(report_path) if report_path is not None else None,
                    expected_sha256=warmstart.get('expected_sha256', None) if rank == 0 else None,
                    compute_sha256=(rank == 0),
                )
            if warmstart_mode != 'va_checkpoint_to_vat':
                logger.info(
                    "Stage 2 selective warm-start complete: WAN=%s Action-DiT=%s report=%s",
                    report["loaded_wan_key_count"],
                    report["loaded_action_dit_key_count"],
                    report_path,
                )
            if warmstart_mode == 'motus_stage2_action_only' and rank == 0:
                initialization_report = build_five_modal_initialization_report(
                    wan_report=target_model.video_model.wan_load_report,
                    task_conditioning_report=target_model.wan_task_text_conditioning_load_report,
                    action_report=report,
                    joint_attention_mode=target_model.config.joint_attention_mode,
                )
                initialization_report_path = (
                    report_dir / 'five_modal_initialization_report.json'
                )
                write_initialization_report(
                    initialization_report_path, initialization_report
                )
                logger.info(
                    "Five-modal initialization report validated: %s",
                    initialization_report_path,
                )
        elif warmstart_mode is not None:
            raise ValueError(f"Unsupported warmstart.mode: {warmstart_mode!r}")

        # Legacy optional finetune initialization remains available to old configs.
        finetune_ckpt = getattr(config.finetune, 'checkpoint_path', None) if hasattr(config, 'finetune') else None
        if warmstart_mode and finetune_ckpt:
            raise ValueError("warmstart and finetune.checkpoint_path are mutually exclusive")
        if (
            not resume_checkpoint
            and getattr(config, 'training_mode', 'finetune') == 'finetune'
            and finetune_ckpt
        ):
            use_understanding = bool(config.model.get('use_understanding', True))
            logger.info(f"Loading finetune weights from {finetune_ckpt} (partial, use_understanding={use_understanding})...")
            try:
                target_model = model.module if hasattr(model, 'module') else model
                if use_understanding:
                    target_model.load_pretrain_weights(finetune_ckpt)
                    logger.info("Finetune weights loaded with original Motus partial loader.")
                else:
                    report_dir = Path(config.system.checkpoint_dir)
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_path = report_dir / "wam_va_warmup_load_report.json"
                    report = target_model.load_warmup_weights(finetune_ckpt, report_path=str(report_path))
                    logger.info(
                        "WAM-VA warmup loaded: loaded=%s skipped_explicit=%s skipped_missing=%s skipped_shape=%s report=%s",
                        report.get("loaded_count"),
                        report.get("skipped_explicit_count"),
                        report.get("skipped_missing_in_target_count"),
                        report.get("skipped_shape_mismatch_count"),
                        report_path,
                    )
            except Exception as e:
                logger.error(f"Failed to load finetune weights: {e}")

        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(config, rank, world_size)
        if count_dataloader_shards(train_dataloader) != 0:
            raise RuntimeError("Training dataloader was sharded before Accelerator.prepare")

        # Prepare everything with accelerator (do not prepare val_dataloader to enable rank0-only local eval)
        logger.info("Preparing model, optimizer, and dataloaders with Accelerator...")
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        expected_shards = 1 if world_size > 1 else 0
        actual_shards = count_dataloader_shards(train_dataloader)
        if actual_shards != expected_shards:
            raise RuntimeError(
                "Training dataloader must be sharded exactly once by Accelerator: "
                f"expected={expected_shards}, actual={actual_shards}"
            )
        if hasattr(train_dataloader, "iteration"):
            accelerator.register_for_checkpointing(
                DataLoaderShardCheckpointState(train_dataloader)
            )
            logger.info(
                "Registered DataLoaderShard iteration state; dataloader shards=%s",
                actual_shards,
            )
        if args.deepspeed is not None:
            deepspeed_runtime = validate_deepspeed_runtime(
                model, config, actual_global_batch
            )
            logger.info("Validated DeepSpeed runtime: %s", deepspeed_runtime)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(accelerator.device)

        # Create trainer
        trainer = UniDiffuserTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=accelerator.device,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=config.system.checkpoint_dir,
            log_interval=config.system.log_interval,
            save_interval=config.system.save_interval,
            val_interval=config.system.val_interval,
            report_to=report_to,
            tb_writer=tb_writer,
            accelerator=accelerator,
            config=config,
        )

        # Start training
        trainer.train(
            max_steps=config.training.max_steps,
            resume_from=config.resume.checkpoint_path,
            val_interval=config.system.val_interval
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if torch.cuda.is_available():
            logger.error(
                "CUDA memory at failure on rank %s: current allocated/reserved=%.2f/%.2f GiB, "
                "peak allocated/reserved=%.2f/%.2f GiB",
                rank,
                torch.cuda.memory_allocated(accelerator.device) / (1024 ** 3),
                torch.cuda.memory_reserved(accelerator.device) / (1024 ** 3),
                torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3),
                torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 3),
            )
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        print(f"[CRITICAL ERROR] Training failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise
    finally:
        # Clean up resources
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rank == 0 and "wandb" in report_to:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()

if __name__ == "__main__":
    main()
