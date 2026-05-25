"""Pretrain VideoMAE on The Well shear_flow dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thewell_videomae.config import ProjectConfig
from thewell_videomae.data import TheWellVideoDataModule
from thewell_videomae.lightning_compat import L, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, TensorBoardLogger
from thewell_videomae.modules import LitVideoMAEPretrain
from thewell_videomae.utils import count_parameters, ensure_dir, normalize_accelerator, parse_devices, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain VideoMAE on The Well")
    parser.add_argument("--data-base", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "runs" / "videomae_shear_flow"))
    parser.add_argument("--run-name", type=str, default="videomae_shear_flow")
    parser.add_argument("--use-hf-streaming", action="store_true")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--spatial-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--max-train-windows", type=int, default=8192)
    parser.add_argument("--max-val-windows", type=int, default=1024)
    parser.add_argument("--max-test-windows", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--encoder-embed-dim", type=int, default=384)
    parser.add_argument("--encoder-depth", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=6)
    parser.add_argument("--decoder-embed-dim", type=int, default=192)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=3)
    parser.add_argument("--mask-ratio", type=float, default=0.90)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    L.seed_everything(args.seed, workers=True)

    cfg = ProjectConfig()
    cfg.run_name = args.run_name
    cfg.output_dir = args.output_dir
    if args.data_base is not None:
        cfg.data.data_base = args.data_base
    cfg.data.use_hf_streaming = args.use_hf_streaming
    cfg.data.download_data = args.download_data
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.data.num_frames = args.num_frames
    cfg.data.spatial_size = tuple(args.spatial_size)
    cfg.data.max_train_windows = args.max_train_windows
    cfg.data.max_val_windows = args.max_val_windows
    cfg.data.max_test_windows = args.max_test_windows
    cfg.model.patch_size = args.patch_size
    cfg.model.tubelet_size = args.tubelet_size
    cfg.model.encoder_embed_dim = args.encoder_embed_dim
    cfg.model.encoder_depth = args.encoder_depth
    cfg.model.encoder_heads = args.encoder_heads
    cfg.model.decoder_embed_dim = args.decoder_embed_dim
    cfg.model.decoder_depth = args.decoder_depth
    cfg.model.decoder_heads = args.decoder_heads
    cfg.model.mask_ratio = args.mask_ratio
    cfg.optim.lr = args.lr
    cfg.optim.min_lr = args.min_lr
    cfg.optim.weight_decay = args.weight_decay
    cfg.trainer.max_epochs = args.epochs
    cfg.trainer.accumulate_grad_batches = args.accumulate_grad_batches
    cfg.trainer.accelerator = args.accelerator
    cfg.trainer.devices = args.devices
    cfg.trainer.precision = args.precision
    cfg.trainer.seed = args.seed

    output_dir = ensure_dir(cfg.output_dir)
    dm = TheWellVideoDataModule(cfg.data, normalization_dir=output_dir)
    dm.prepare_data()
    dm.setup("fit")
    sample = dm.train_dataset[0]
    data_shape = {
        "channels": int(sample["video"].shape[0]),
        "frames": int(sample["video"].shape[1]),
        "height": int(sample["video"].shape[2]),
        "width": int(sample["video"].shape[3]),
    }
    model = LitVideoMAEPretrain(model_cfg=cfg.model, optim_cfg=cfg.optim, data_shape=data_shape)
    print(f"Trainable parameters: {count_parameters(model):,}")

    logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=20),
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            monitor="val_masked_patch_mse",
            mode="min",
            save_top_k=1,
            filename="videomae-{epoch:03d}-{val_masked_patch_mse:.4f}",
        ),
    ]
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=normalize_accelerator(cfg.trainer.accelerator),
        devices=parse_devices(cfg.trainer.devices),
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.optim.grad_clip_norm,
    )
    write_json(output_dir / "config.json", cfg)
    write_json(output_dir / "data_summary.json", dm.metadata_summary())
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
    torch.save(model.model.state_dict(), output_dir / "videomae_backbone_state.pt")


if __name__ == "__main__":
    main()

