"""Train the small CIFAR I-JEPA sanity-check model.

Place this file next to your existing modules and run, for example:

    python train_small_ijepa.py --epochs 20 --batch-size 256 --accelerator auto --devices auto

It uses your existing CIFAR10DataModule. Labels are ignored during I-JEPA training.
"""

from __future__ import annotations

import argparse

try:
    from .data_manager import CIFAR10DataModule, DataConfig
    from .lightning_compat import L, LearningRateMonitor, ModelCheckpoint, TensorBoardLogger, TQDMProgressBar
    from .small_ijepa import LitMiniIJEPA
    from .utils import count_parameters, normalize_accelerator, parse_devices
except ImportError:
    from data_manager import CIFAR10DataModule, DataConfig  # type: ignore
    from lightning_compat import L, LearningRateMonitor, ModelCheckpoint, TensorBoardLogger, TQDMProgressBar  # type: ignore
    from small_ijepa import LitMiniIJEPA  # type: ignore
    from utils import count_parameters, normalize_accelerator, parse_devices  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small I-JEPA sanity check on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=3)
    parser.add_argument("--predictor-dim", type=int, default=128)
    parser.add_argument("--predictor-depth", type=int, default=4)
    parser.add_argument("--predictor-heads", type=int, default=4)
    parser.add_argument("--target-blocks", type=int, default=4)
    parser.add_argument("--target-block-size", type=int, default=2)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--compile", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dm = CIFAR10DataModule(
        DataConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            workers=args.workers,
            randaugment=False,
            random_erasing=0.0,
        )
    )

    model = LitMiniIJEPA(
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        num_target_blocks=args.target_blocks,
        target_block_size=args.target_block_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        compile_model=args.compile,
    )
    print(f"Trainable parameters: {count_parameters(model):,}")

    logger = TensorBoardLogger(save_dir="runs", name="mini_ijepa")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=20),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="mini-ijepa-{epoch:03d}-{val_loss:.4f}"),
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=normalize_accelerator(args.accelerator),
        devices=parse_devices(args.devices),
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=25,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
