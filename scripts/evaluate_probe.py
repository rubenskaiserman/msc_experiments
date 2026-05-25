"""Run evaluation and write a compact report for a trained probe checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thewell_videomae.config import ProjectConfig
from thewell_videomae.data import TheWellVideoDataModule
from thewell_videomae.metrics import probe_metrics, reconstruction_metrics_dataframe
from thewell_videomae.model import VideoMAE
from thewell_videomae.modules import LitFrozenProbe
from thewell_videomae.utils import load_state_dict_flexible, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained VideoMAE probe checkpoint")
    parser.add_argument("--probe-checkpoint", type=str, required=True)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--data-base", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--use-hf-streaming", action="store_true")
    return parser


def collect_predictions(module: LitFrozenProbe, loader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(module.parameters()).device
    module.eval()
    yhats, ys, yraws = [], [], []
    with torch.no_grad():
        for batch in loader:
            yhat = module(batch["video"].to(device)).cpu()
            yhats.append(yhat)
            ys.append(batch["label"].cpu())
            yraws.append(batch["label_raw"].cpu())
    return torch.cat(yhats, dim=0), torch.cat(ys, dim=0), torch.cat(yraws, dim=0)


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def main() -> None:
    args = build_parser().parse_args()

    cfg = ProjectConfig()
    if args.data_base is not None:
        cfg.data.data_base = args.data_base
    cfg.data.use_hf_streaming = args.use_hf_streaming
    dm = TheWellVideoDataModule(cfg.data)
    dm.prepare_data()
    dm.setup("test")
    sample = dm.test_dataset[0]
    data_shape = {
        "channels": int(sample["video"].shape[0]),
        "frames": int(sample["video"].shape[1]),
        "height": int(sample["video"].shape[2]),
        "width": int(sample["video"].shape[3]),
    }
    module = LitFrozenProbe(
        model_cfg=cfg.model,
        probe_cfg=cfg.probe,
        optim_cfg=cfg.optim,
        data_shape=data_shape,
        label_names=list(dm.label_names),
        label_mean=dm.label_mean,
        label_std=dm.label_std,
        label_transform=cfg.data.label_transform,
    )
    probe_state = load_checkpoint_state(Path(args.probe_checkpoint).expanduser().resolve())
    load_state_dict_flexible(module, probe_state)
    yhat, y, yraw = collect_predictions(module, dm.test_dataloader())
    metrics = probe_metrics(
        yhat,
        y,
        yraw,
        label_names=list(dm.label_names),
        label_mean=dm.label_mean,
        label_std=dm.label_std,
        label_transform=cfg.data.label_transform,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_dir / "probe_metrics.csv", index=False)

    if args.backbone_checkpoint is not None:
        backbone = VideoMAE(
            img_size=(data_shape["height"], data_shape["width"]),
            num_frames=data_shape["frames"],
            in_chans=data_shape["channels"],
            patch_size=cfg.model.patch_size,
            tubelet_size=cfg.model.tubelet_size,
            encoder_embed_dim=cfg.model.encoder_embed_dim,
            encoder_depth=cfg.model.encoder_depth,
            encoder_heads=cfg.model.encoder_heads,
            decoder_embed_dim=cfg.model.decoder_embed_dim,
            decoder_depth=cfg.model.decoder_depth,
            decoder_heads=cfg.model.decoder_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_pix_loss=cfg.model.norm_pix_loss,
            tube_masking=cfg.model.tube_masking,
        )
        load_state_dict_flexible(backbone, load_checkpoint_state(Path(args.backbone_checkpoint).expanduser().resolve()))
        recon_df = reconstruction_metrics_dataframe(backbone, dm.val_dataloader(), mask_ratio=cfg.model.mask_ratio, max_steps=25)
        recon_df.to_csv(output_dir / "reconstruction_metrics.csv", index=False)

    write_json(output_dir / "report.json", {"metrics": metrics, "data_summary": dm.metadata_summary()})


if __name__ == "__main__":
    main()

