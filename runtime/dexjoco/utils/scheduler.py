#!/usr/bin/env python3
"""
Learning rate schedulers for joint video-action training.
"""

import torch
from typing import Optional
import math
from diffusers.optimization import get_scheduler as hf_get_scheduler


class LambdaLinearScheduler:
    """Linear warmup/decay with independently checkpointed group schedules."""

    def __init__(
        self,
        optimizer,
        warm_up_steps: int,
        cycle_length: int,
        f_max: float = 1.0,
        f_min: float = 0.1,
        f_start: float = 1e-6,
        group_schedules: Optional[dict[str, dict]] = None,
    ):
        self.optimizer = optimizer
        self.warm_up_steps = int(warm_up_steps)
        self.cycle_length = int(cycle_length)
        self.f_max = float(f_max)
        self.f_min = float(f_min)
        self.f_start = float(f_start)
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        self.base_lr = self.base_lrs[0]
        self.step_count = 0
        self.group_schedules: dict[str, dict] = {}
        self.group_step_counts: dict[str, int] = {}
        configured = group_schedules or {}
        for index, group in enumerate(optimizer.param_groups):
            name = str(group.get('name', f'group_{index}'))
            if name in self.group_schedules:
                raise ValueError(f"Duplicate optimizer parameter-group name: {name}")
            override = dict(configured.get(name, {}))
            schedule = {
                'warm_up_steps': int(override.get('warm_up_steps', self.warm_up_steps)),
                'cycle_length': int(override.get('cycle_length', self.cycle_length)),
                'f_max': float(override.get('f_max', self.f_max)),
                'f_min': float(override.get('f_min', self.f_min)),
                'f_start': float(override.get('f_start', self.f_start)),
            }
            self._validate_schedule(name, schedule)
            self.group_schedules[name] = schedule
            self.group_step_counts[name] = 0

    @staticmethod
    def _validate_schedule(name: str, schedule: dict) -> None:
        if schedule['warm_up_steps'] < 0:
            raise ValueError(f"{name}: warm_up_steps must be non-negative")
        if schedule['cycle_length'] <= schedule['warm_up_steps']:
            raise ValueError(f"{name}: cycle_length must exceed warm_up_steps")
        if schedule['f_max'] < schedule['f_min']:
            raise ValueError(f"{name}: f_max must be >= f_min")
        if schedule['f_start'] < 0:
            raise ValueError(f"{name}: f_start must be non-negative")

    def step(self):
        self.step_count += 1
        for index, group in enumerate(self.optimizer.param_groups):
            name = str(group.get('name', f'group_{index}'))
            self.group_step_counts[name] += 1
            multiplier = self.get_lr_multiplier(
                self.group_step_counts[name],
                **self.group_schedules[name],
            )
            group['lr'] = self.base_lrs[index] * multiplier

    def get_lr_multiplier(
        self,
        step: int,
        *,
        warm_up_steps: Optional[int] = None,
        cycle_length: Optional[int] = None,
        f_max: Optional[float] = None,
        f_min: Optional[float] = None,
        f_start: Optional[float] = None,
    ) -> float:
        warm_up_steps = self.warm_up_steps if warm_up_steps is None else int(warm_up_steps)
        cycle_length = self.cycle_length if cycle_length is None else int(cycle_length)
        f_max = self.f_max if f_max is None else float(f_max)
        f_min = self.f_min if f_min is None else float(f_min)
        f_start = self.f_start if f_start is None else float(f_start)
        if step <= 0:
            return f_start
        if warm_up_steps > 0 and step <= warm_up_steps:
            return f_start + (f_max - f_start) * step / warm_up_steps
        if step < cycle_length:
            remaining_steps = cycle_length - step
            decay_steps = cycle_length - warm_up_steps
            return f_min + (f_max - f_min) * remaining_steps / decay_steps
        return f_min

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'step_count': self.step_count,
            'base_lr': self.base_lr,
            'base_lrs': self.base_lrs,
            'warm_up_steps': self.warm_up_steps,
            'cycle_length': self.cycle_length,
            'f_max': self.f_max,
            'f_min': self.f_min,
            'f_start': self.f_start,
            'group_schedules': self.group_schedules,
            'group_step_counts': self.group_step_counts,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict['step_count'])
        self.base_lr = float(state_dict.get('base_lr', self.base_lr))
        self.base_lrs = [float(value) for value in state_dict.get('base_lrs', self.base_lrs)]
        self.warm_up_steps = int(state_dict['warm_up_steps'])
        self.cycle_length = int(state_dict['cycle_length'])
        self.f_max = float(state_dict['f_max'])
        self.f_min = float(state_dict['f_min'])
        self.f_start = float(state_dict['f_start'])
        if 'group_schedules' in state_dict:
            if set(state_dict['group_schedules']) != set(self.group_schedules):
                raise ValueError(
                    "Scheduler parameter-group names changed across reload: "
                    f"saved={sorted(state_dict['group_schedules'])} "
                    f"runtime={sorted(self.group_schedules)}"
                )
            self.group_schedules = state_dict['group_schedules']
            self.group_step_counts = {
                name: int(value)
                for name, value in state_dict['group_step_counts'].items()
            }
        else:
            self.group_step_counts = {
                name: self.step_count for name in self.group_step_counts
            }


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on config.

    Args:
        optimizer: PyTorch optimizer
        config: Training configuration object

    Returns:
        Learning rate scheduler
    """
    scheduler_type = getattr(config.training, 'scheduler_type', 'cosine')

    if scheduler_type == "linear":
        group_schedules = {}
        task_schedule = getattr(config.training, 'task_projection_scheduler', None)
        if task_schedule is not None:
            group_schedules['task_projection'] = {
                'warm_up_steps': int(
                    getattr(task_schedule, 'warmup_steps', config.training.warmup_steps)
                ),
                'cycle_length': int(
                    getattr(task_schedule, 'cycle_length', config.training.cycle_length)
                ),
                'f_max': float(getattr(task_schedule, 'f_max', config.training.f_max)),
                'f_min': float(getattr(task_schedule, 'f_min', config.training.f_min)),
                'f_start': float(
                    getattr(
                        task_schedule,
                        'f_start',
                        getattr(config.training, 'f_start', 1e-6),
                    )
                ),
            }
        return LambdaLinearScheduler(
            optimizer,
            warm_up_steps=config.training.warmup_steps,
            cycle_length=config.training.cycle_length,
            f_max=config.training.f_max,
            f_min=config.training.f_min,
            f_start=getattr(config.training, 'f_start', 1e-6),
            group_schedules=group_schedules,
        )
    elif scheduler_type == "cosine":
        # 保持向后兼容：原生PyTorch余弦（带min比例）
        T_max = getattr(config.training, 'max_steps', 10000)
        eta_min = float(getattr(config.training, 'min_lr', config.training.learning_rate * getattr(config.training, 'min_lr_ratio', 0.1)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    elif scheduler_type == "diffusers_cosine":
        # 使用 diffusers.optimization.get_scheduler 的余弦调度
        total_steps = int(getattr(config.training, 'lr_schedule_steps', getattr(config.training, 'max_steps', 10000)))
        warmup_steps = int(getattr(config.training, 'warmup_steps', 0))
        base = float(config.training.learning_rate)
        min_lr = float(getattr(config.training, 'min_lr', base * getattr(config.training, 'min_lr_ratio', 0.0)))

        inner = hf_get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        class _MinLRWrapper:
            def __init__(self, inner_sched, opt, min_lr_value: float):
                self.inner = inner_sched
                self.optimizer = opt
                self.min_lr = float(min_lr_value)

            def step(self):
                self.inner.step()
                if self.min_lr > 0.0:
                    for g in self.optimizer.param_groups:
                        if g.get('lr', 0.0) < self.min_lr:
                            g['lr'] = self.min_lr

            def get_last_lr(self):
                try:
                    return self.inner.get_last_lr()
                except Exception:
                    return [g['lr'] for g in self.optimizer.param_groups]

            # 兼容 accelerate.save_state / load_state
            def state_dict(self):
                sd = getattr(self.inner, 'state_dict', None)
                return sd() if callable(sd) else {}

            def load_state_dict(self, state):
                ld = getattr(self.inner, 'load_state_dict', None)
                if callable(ld):
                    ld(state)

        return _MinLRWrapper(inner, optimizer, min_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
