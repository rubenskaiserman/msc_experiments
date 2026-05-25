"""LightningModule wrapping the ViT classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from .lightning_compat import L
from .vit_cifar10 import VisionTransformer
from .utils import accuracy_tensor, cosine_warmup_lambda, make_param_groups


class LitViTClassifier(L.LightningModule):
    """Supervised CIFAR-10 ViT classifier with AdamW and cosine warmup schedule."""

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
                print("Warning: torch.compile is unavailable in this PyTorch version; ignoring compile_model")

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

        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("test_acc", acc, on_step=False, on_epoch=True, batch_size=images.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            make_param_groups(self.model, self.hparams.weight_decay),
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
