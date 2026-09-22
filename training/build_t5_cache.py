"""Build the disk-backed T5 cache used by the Clean50 training loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from runtime.wan.modules.t5 import T5EncoderModel


def instruction_files(root: Path) -> list[Path]:
    task_root = root / "production" if (root / "production").is_dir() else root
    return sorted(task_root.glob("*/tactile_replay_aloha_clean50_tfa2_full/instructions/episode*.json"))


def collect_prompts(root: Path, fields: tuple[str, ...]) -> list[str]:
    prompts: set[str] = set()
    for path in instruction_files(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for field in fields:
            prompts.update(str(value) for value in payload[field])
    return sorted(prompts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--t5-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument(
        "--prefix",
        default=(
            "The whole scene is in a realistic, industrial art style with three views: "
            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
            "The aloha robot is currently performing the following task: "
        ),
    )
    args = parser.parse_args()

    prompts = collect_prompts(args.clean_root, ("seen", "unseen"))
    if not prompts:
        raise ValueError("No instruction prompts found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload_path = args.output.with_suffix(".bin.tmp")
    index: list[tuple[int, int, int]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = T5EncoderModel(
        text_len=args.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(args.t5_checkpoint),
        tokenizer_path=str(args.tokenizer),
    )
    with payload_path.open("wb") as stream:
        for start in range(0, len(prompts), args.batch_size):
            batch = [args.prefix + prompt for prompt in prompts[start : start + args.batch_size]]
            embeddings = encoder(batch, device=device)
            for embedding in embeddings:
                value = embedding.detach().to(dtype=torch.bfloat16).contiguous()
                offset = stream.tell() // np.dtype(np.uint16).itemsize
                stream.write(value.view(torch.uint16).cpu().numpy().tobytes())
                index.append((offset, int(value.shape[0]), int(value.shape[1])))
            print(f"encoded {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)

    metadata_path = args.output.with_suffix(".pt.tmp")
    torch.save(
        {
            "prompts": prompts,
            "text_len": args.text_len,
            "cache_format": "variable_length_per_prompt_disk_v2",
            "motus_prefix": args.prefix,
            "t5": {
                "checkpoint_path": str(args.t5_checkpoint),
                "tokenizer_path": str(args.tokenizer),
                "text_len": args.text_len,
                "frozen": True,
            },
            "embedding_index": index,
            "coverage": "clean50_instructions",
            "instruction_fields": ("seen", "unseen"),
        },
        metadata_path,
    )
    payload_path.replace(args.output.with_suffix(".bin"))
    metadata_path.replace(args.output)
    print(f"wrote {args.output} and {args.output.with_suffix('.bin')}")


if __name__ == "__main__":
    main()
