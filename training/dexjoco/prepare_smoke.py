"""Derive a short real-data run without changing the reference recipe."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='training/configs/dexjoco_uni.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--world-size', type=int, default=2)
    p.add_argument('--steps', type=int, default=2)
    args = p.parse_args()
    if args.world_size < 1 or args.steps < 1:
        p.error('world-size and steps must be positive')
    c = OmegaConf.load(args.config)
    c.training.batch_size = 1
    c.training.expected_world_size = args.world_size
    c.training.global_batch_size = args.world_size
    c.training.max_steps = args.steps
    c.dataset.params.sampling_global_batch_size = args.world_size
    c.system.num_workers = 0
    c.system.log_interval = 1
    c.system.memory_log_interval = 1
    c.system.save_interval = args.steps
    c.system.val_interval = args.steps + 1
    # A two-update run cannot cover all eleven tasks in a reporting window.
    c.system.per_task_action_loss_interval = 0
    c.logging.run_name = 'smoke'
    c.logging.report_to = 'none'
    c.system.checkpoint_dir = str(args.output.parent / 'checkpoints')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(c, args.output)
    print(args.output)

if __name__ == '__main__':
    main()
