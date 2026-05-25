"""Small reusable utilities for training and optimization."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def accuracy_tensor(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return batch accuracy as a scalar tensor."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Create AdamW groups with no decay on biases, norms, and embeddings."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_skip_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "pos_embed" in name
            or "cls_token" in name
        )

        if should_skip_decay:
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
    """Linear warmup followed by cosine decay."""
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))

    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def parse_devices(value: str) -> int | str:
    """Allow values like ``auto`` and ``1`` for Lightning's ``devices`` argument."""
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        return value


def normalize_accelerator(value: str) -> str:
    """Map a common CUDA alias to Lightning's preferred GPU accelerator name."""
    return "gpu" if value == "cuda" else value
