#!/usr/bin/env python3
"""Train a small ViT on CIFAR-10 with PyTorch Lightning.

Example:
    python train.py --epochs 50 --batch-size 128 --amp --randaugment --random-erasing 0.25

TensorBoard:
    tensorboard --logdir ./runs/lightning_vit_cifar10 --port 6006
"""

from __future__ import annotations

import argparse
import os

import torch

from model import CIFAR10DataModule, DataConfig, LitViTClassifier
from model.lightning_compat import L, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, TensorBoardLogger
from model.utils import count_parameters, normalize_accelerator, parse_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small ViT on CIFAR-10 with PyTorch Lightning")

    # Data/training
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out-dir", type=str, default="./runs/lightning_vit_cifar10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use 16-bit mixed precision on CUDA")
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


def build_datamodule(args: argparse.Namespace) -> CIFAR10DataModule:
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        randaugment=args.randaugment,
        random_erasing=args.random_erasing,
    )
    return CIFAR10DataModule(data_cfg)


def build_model(args: argparse.Namespace) -> LitViTClassifier:
    return LitViTClassifier(
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


def build_trainer(args: argparse.Namespace, logger: TensorBoardLogger, checkpoint: ModelCheckpoint) -> L.Trainer:
    precision = "16-mixed" if args.amp else args.precision

    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=args.progress_refresh_rate),
    ]

    return L.Trainer(
        max_epochs=args.epochs,
        accelerator=normalize_accelerator(args.accelerator),
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


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    datamodule = build_datamodule(args)
    model = build_model(args)

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

    trainer = build_trainer(args, logger, checkpoint)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    if not args.fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print("\nTensorBoard logs:", logger.log_dir)
    print("Best checkpoint:", checkpoint.best_model_path)
    print("\nOpen TensorBoard with:")
    print(f"  tensorboard --logdir {args.out_dir} --port 6006")


if __name__ == "__main__":
    main()
