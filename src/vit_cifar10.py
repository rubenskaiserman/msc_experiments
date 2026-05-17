#!/usr/bin/env python3
"""
PyTorch Lightning Vision Transformer for CIFAR-10.

This is a supervised CIFAR-10 ViT baseline intended as a small, readable training
pipeline for a single consumer GPU such as an RTX 4050. It logs losses, accuracy,
learning rate, hyperparameters, and checkpoints through Lightning + TensorBoard.

Install:
    pip install torch torchvision lightning tensorboard

Train:
    python train_vit_cifar10_lightning.py --epochs 50 --batch-size 128 --amp \
        --randaugment --random-erasing 0.25

Watch TensorBoard:
    tensorboard --logdir ./runs/lightning_vit_cifar10 --port 6006

If CUDA runs out of memory:
    python train_vit_cifar10_lightning.py --epochs 50 --batch-size 64 --amp

Smoke test:
    python train_vit_cifar10_lightning.py --fast-dev-run
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
    from lightning.pytorch.loggers import TensorBoardLogger
except ModuleNotFoundError:  # fallback for older installations
    import pytorch_lightning as L  # type: ignore
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar  # type: ignore
    from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def accuracy_tensor(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Batch accuracy as a scalar tensor."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW parameter groups: no weight decay on biases, norms, and embeddings."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "pos_embed" in name
            or "cls_token" in name
        ):
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
    """Allow --devices auto or --devices 1 without making argparse awkward."""
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        return value


# -----------------------------------------------------------------------------
# ViT model
# -----------------------------------------------------------------------------


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 32, 32] -> [B, num_patches, embed_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path_prob)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path1(y)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path_prob=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls)


# -----------------------------------------------------------------------------
# Lightning DataModule
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 4
    randaugment: bool = False
    random_erasing: float = 0.0


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.test_set = None

    def prepare_data(self) -> None:
        # Download once, if needed.
        datasets.CIFAR10(root=self.cfg.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.cfg.data_dir, train=False, download=True)

    def build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        train_ops: list[Any] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if self.cfg.randaugment:
            if hasattr(transforms, "RandAugment"):
                train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
            else:
                print("Warning: torchvision.transforms.RandAugment is unavailable; ignoring --randaugment")

        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        if self.cfg.random_erasing > 0.0:
            train_ops.append(
                transforms.RandomErasing(
                    p=self.cfg.random_erasing,
                    scale=(0.02, 0.20),
                    ratio=(0.3, 3.3),
                )
            )

        train_transform = transforms.Compose(train_ops)
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        return train_transform, test_transform

    def setup(self, stage: str | None = None) -> None:
        train_transform, test_transform = self.build_transforms()
        self.train_set = datasets.CIFAR10(root=self.cfg.data_dir, train=True, download=False, transform=train_transform)
        self.test_set = datasets.CIFAR10(root=self.cfg.data_dir, train=False, download=False, transform=test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.cfg.workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


# -----------------------------------------------------------------------------
# Lightning Module
# -----------------------------------------------------------------------------


class LitViTClassifier(L.LightningModule):
    def __init__(
        self,
        *,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.05,
        lr: float = 3e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        epochs: int = 50,
        label_smoothing: float = 0.1,
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        model = VisionTransformer(
            image_size=32,
            patch_size=patch_size,
            in_chans=3,
            num_classes=10,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path_rate=drop_path,
        )

        if compile_model:
            if hasattr(torch, "compile"):
                model = torch.compile(model)  # type: ignore[assignment]
            else:
                print("Warning: torch.compile is unavailable in this PyTorch version; ignoring --compile")

        self.model = model
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.example_input_array = torch.zeros(1, 3, 32, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        acc = accuracy_tensor(logits, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        acc = accuracy_tensor(logits, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        acc = accuracy_tensor(logits, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self, self.hparams.weight_decay),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(total_steps * self.hparams.warmup_epochs / max(1, self.hparams.epochs)))
        min_lr_ratio = self.hparams.min_lr / self.hparams.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_warmup_lambda(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
            ),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small ViT on CIFAR-10 with PyTorch Lightning")

    # Data/training
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./runs/lightning_vit_cifar10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--precision", type=str, default="32-true", help="Lightning precision, e.g. 32-true, 16-mixed, bf16-mixed")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the inner ViT model if available")
    parser.add_argument("--resume", type=str, default=None, help="Path to a Lightning checkpoint to resume from")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one train/val batch for debugging")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--progress-refresh-rate", type=int, default=20)
    parser.add_argument("--log-graph", action="store_true", help="Ask TensorBoardLogger to log the model graph")

    # Hardware
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda", "mps"])
    parser.add_argument("--devices", type=str, default="auto", help="auto, 1, 2, ...")
    parser.add_argument("--matmul-precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--deterministic", action="store_true", help="More reproducible but sometimes slower")

    # Augmentation
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.0)

    # Model
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.05)

    # Optimizer/scheduler
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    precision = args.precision
    if args.amp:
        precision = "16-mixed"

    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        randaugment=args.randaugment,
        random_erasing=args.random_erasing,
    )
    datamodule = CIFAR10DataModule(data_cfg)

    model = LitViTClassifier(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path=args.drop_path,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        label_smoothing=args.label_smoothing,
        compile_model=args.compile,
    )

    print(f"Trainable parameters: {count_parameters(model):,}")

    logger = TensorBoardLogger(
        save_dir=args.out_dir,
        name="tensorboard",
        log_graph=args.log_graph,
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        filename="epoch={epoch:03d}-val_acc={val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=args.progress_refresh_rate),
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every,
        gradient_clip_val=args.clip_grad_norm,
        deterministic=args.deterministic,
        benchmark=not args.deterministic,
        fast_dev_run=args.fast_dev_run,
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    if not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print("\nTensorBoard logs:", logger.log_dir)
    print("Best checkpoint:", checkpoint.best_model_path)
    print("\nOpen TensorBoard with:")
    print(f"  tensorboard --logdir {args.out_dir} --port 6006")


if __name__ == "__main__":
    main()