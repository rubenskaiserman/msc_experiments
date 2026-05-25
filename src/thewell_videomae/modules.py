"""Lightning modules for VideoMAE pretraining and frozen-probe training."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from .config import ModelConfig, OptimizationConfig, ProbeConfig
from .lightning_compat import L
from .metrics import probe_metrics
from .model import AttentiveProbe, VideoMAE
from .utils import cosine_warmup_lambda, make_param_groups


class LitVideoMAEPretrain(L.LightningModule):
    def __init__(self, *, model_cfg: ModelConfig, optim_cfg: OptimizationConfig, data_shape: dict[str, int]):
        super().__init__()
        self.save_hyperparameters(
            {
                "model_cfg": asdict(model_cfg),
                "optim_cfg": asdict(optim_cfg),
                "data_shape": data_shape,
            }
        )
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.model = VideoMAE(
            img_size=(data_shape["height"], data_shape["width"]),
            num_frames=data_shape["frames"],
            in_chans=data_shape["channels"],
            patch_size=model_cfg.patch_size,
            tubelet_size=model_cfg.tubelet_size,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_depth=model_cfg.encoder_depth,
            encoder_heads=model_cfg.encoder_heads,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_depth=model_cfg.decoder_depth,
            decoder_heads=model_cfg.decoder_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pix_loss=model_cfg.norm_pix_loss,
            tube_masking=model_cfg.tube_masking,
        )
        self.example_input_array = torch.zeros(
            1,
            data_shape["channels"],
            data_shape["frames"],
            data_shape["height"],
            data_shape["width"],
        )

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None):
        ratio = self.model_cfg.mask_ratio if mask_ratio is None else mask_ratio
        return self.model(x, mask_ratio=ratio)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _, _ = self(batch["video"])
        self.log("train_masked_patch_mse", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["video"].size(0))
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, _, _ = self(batch["video"])
        self.log("val_masked_patch_mse", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["video"].size(0))

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, _, _ = self(batch["video"])
        self.log("test_masked_patch_mse", loss, on_step=False, on_epoch=True, batch_size=batch["video"].size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self.model, self.optim_cfg.weight_decay),
            lr=self.optim_cfg.lr,
            betas=(0.9, 0.95),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(total_steps * self.optim_cfg.warmup_fraction))
        min_lr_ratio = self.optim_cfg.min_lr / self.optim_cfg.lr
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


class LitFrozenProbe(L.LightningModule):
    def __init__(
        self,
        *,
        model_cfg: ModelConfig,
        probe_cfg: ProbeConfig,
        optim_cfg: OptimizationConfig,
        data_shape: dict[str, int],
        label_names: list[str],
        label_mean: torch.Tensor,
        label_std: torch.Tensor,
        label_transform: str,
        pretrained_backbone_state: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "model_cfg": asdict(model_cfg),
                "probe_cfg": asdict(probe_cfg),
                "optim_cfg": asdict(optim_cfg),
                "data_shape": data_shape,
                "label_names": label_names,
                "label_transform": label_transform,
            }
        )
        self.model_cfg = model_cfg
        self.probe_cfg = probe_cfg
        self.optim_cfg = optim_cfg
        self.label_names = label_names
        self.label_transform = label_transform
        self.backbone = VideoMAE(
            img_size=(data_shape["height"], data_shape["width"]),
            num_frames=data_shape["frames"],
            in_chans=data_shape["channels"],
            patch_size=model_cfg.patch_size,
            tubelet_size=model_cfg.tubelet_size,
            encoder_embed_dim=model_cfg.encoder_embed_dim,
            encoder_depth=model_cfg.encoder_depth,
            encoder_heads=model_cfg.encoder_heads,
            decoder_embed_dim=model_cfg.decoder_embed_dim,
            decoder_depth=model_cfg.decoder_depth,
            decoder_heads=model_cfg.decoder_heads,
            mlp_ratio=model_cfg.mlp_ratio,
            norm_pix_loss=model_cfg.norm_pix_loss,
            tube_masking=model_cfg.tube_masking,
        )
        if pretrained_backbone_state is not None:
            self.backbone.load_state_dict(pretrained_backbone_state)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.probe = AttentiveProbe(
            dim=model_cfg.encoder_embed_dim,
            out_dim=len(label_names),
            num_queries=probe_cfg.num_queries,
            num_heads=probe_cfg.heads,
            dropout=probe_cfg.dropout,
        )
        self.register_buffer("label_mean", label_mean.clone().float(), persistent=True)
        self.register_buffer("label_std", label_std.clone().float(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.backbone.extract_features(x)
        return self.probe(tokens.detach())

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        yhat = self(batch["video"])
        loss = F.mse_loss(yhat, batch["label"])
        self.log("train_probe_mse", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["video"].size(0))
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        yhat = self(batch["video"])
        loss = F.mse_loss(yhat, batch["label"])
        self.log("val_probe_mse", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["video"].size(0))

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        yhat = self(batch["video"])
        return {
            "yhat": yhat.detach().cpu(),
            "y": batch["label"].detach().cpu(),
            "yraw": batch["label_raw"].detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self.probe, self.probe_cfg.weight_decay),
            lr=self.probe_cfg.lr,
            betas=(0.9, 0.999),
        )
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(total_steps * self.optim_cfg.warmup_fraction))
        min_lr_ratio = self.probe_cfg.min_lr / self.probe_cfg.lr
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def summarize_predictions(
        self,
        yhat_norm: torch.Tensor,
        y_norm: torch.Tensor,
        yraw: torch.Tensor,
    ) -> dict[str, float]:
        return probe_metrics(
            yhat_norm,
            y_norm,
            yraw,
            label_names=self.label_names,
            label_mean=self.label_mean.cpu(),
            label_std=self.label_std.cpu(),
            label_transform=self.label_transform,
        )
