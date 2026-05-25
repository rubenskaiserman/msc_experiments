"""Shared utilities for the The Well VideoMAE project."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        skip_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "pos_embed" in name
            or "mask_token" in name
            or "query" in name
        )
        if skip_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_warmup_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def parse_devices(value: str) -> int | str:
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        return value


def normalize_accelerator(value: str) -> str:
    return "gpu" if value == "cuda" else value


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def to_serializable_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_serializable_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable_dict(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_serializable_dict(payload), indent=2))


def load_state_dict_flexible(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in model_state:
            cleaned[key] = value
            continue
        for prefix in ("model.", "backbone.", "videomae.", "module."):
            if key.startswith(prefix) and key[len(prefix) :] in model_state:
                cleaned[key[len(prefix) :]] = value
                break
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Failed to load checkpoint cleanly. Missing keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys after filtering: {unexpected[:8]}")

