from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import numpy as np
import torch
from safetensors import safe_open

class DiskT5Cache:
    """Small metadata object with a lazy, copy-on-write bf16 payload mapping."""

    def __init__(self, metadata_path: str) -> None:
        self.metadata_path = Path(metadata_path)
        metadata = torch.load(self.metadata_path, map_location="cpu", weights_only=False)
        if metadata.get("cache_format") != "variable_length_per_prompt_disk_v2":
            raise ValueError(f"Unsupported T5 cache format: {metadata.get('cache_format')}")
        self.prompts = tuple(str(value) for value in metadata["prompts"])
        self.prompt_to_index = {prompt: index for index, prompt in enumerate(self.prompts)}
        self.prompt_set = frozenset(self.prompt_to_index)
        self.embedding_index = tuple(
            tuple(int(value) for value in entry) for entry in metadata["embedding_index"]
        )
        if len(self.prompts) != len(self.embedding_index):
            raise ValueError("T5 prompt table and embedding index lengths differ")
        self.payload_path = self.metadata_path.with_suffix(".bin")
        if not self.payload_path.is_file():
            raise FileNotFoundError(f"T5 cache payload not found: {self.payload_path}")
        self._payload: Optional[np.memmap] = None

    def embedding(self, prompt: str) -> torch.Tensor:
        try:
            index = self.prompt_to_index[prompt]
        except KeyError as error:
            raise KeyError(f"Instruction is absent from the T5 cache: {prompt!r}") from error
        if self._payload is None:
            self._payload = np.memmap(self.payload_path, dtype=np.uint16, mode="c")
        offset, length, hidden_size = self.embedding_index[index]
        if hidden_size != 4096 or length <= 0:
            raise ValueError(f"Invalid cached T5 shape: {(length, hidden_size)}")
        raw = np.asarray(self._payload[offset : offset + length * hidden_size])
        if raw.size != length * hidden_size:
            raise RuntimeError(f"Truncated T5 cache payload: {self.payload_path}")
        return torch.from_numpy(raw.reshape(length, hidden_size)).view(torch.bfloat16)

class TaskT5Cache:
    """Per-task raw-instruction cache shared with Motus-VA."""
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._entries: dict[str, tuple[str, Path]] = {}
        self._embeddings: dict[str, torch.Tensor] = {}
        for path in sorted(self.root.glob("*.safetensors")):
            meta = path.with_suffix(".json")
            if not meta.is_file():
                raise FileNotFoundError(meta)
            payload = json.loads(meta.read_text(encoding="utf-8"))
            instructions = payload.get("instructions", [])
            if len(instructions) != 1:
                raise ValueError(f"{meta}: expected exactly one raw instruction")
            self._entries[path.stem] = (str(instructions[0]), path)

    def embedding(self, task: str, prompt: str) -> torch.Tensor:
        try:
            cached_prompt, path = self._entries[task]
        except KeyError as error:
            raise KeyError(f"Missing per-task T5 cache for {task!r}") from error
        if cached_prompt != prompt:
            raise ValueError(f"{task}: cache instruction does not match raw task prompt")
        embedding = self._embeddings.get(task)
        if embedding is None:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if list(handle.keys()) != ["embedding_000"]:
                    raise ValueError(f"{path}: expected embedding_000 only")
                embedding = handle.get_tensor("embedding_000").contiguous()
            if embedding.ndim != 2 or embedding.shape[1] != 4096:
                raise ValueError(f"{path}: invalid embedding shape {tuple(embedding.shape)}")
            # Dataset workers only read these tensors.  Keeping one tensor per
            # task removes a safetensors open/stat sequence from every sample.
            self._embeddings[task] = embedding
        return embedding
