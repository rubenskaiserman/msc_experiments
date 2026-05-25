"""Small I-JEPA implementation for CIFAR-sized images.

Drop this file next to your existing ``model.py``, ``utils.py`` and
``lightning_compat.py`` modules.

The implementation is intentionally small and sanity-check oriented:
- 32x32 images, patch size 4 by default -> 8x8 token grid.
- No [CLS] token during JEPA pretraining.
- Online/context encoder and EMA target encoder.
- A narrow transformer predictor conditioned on target-position mask tokens.
- Fixed-size random block masks sampled inside the LightningModule/model.

This is not an ImageNet-scale reproduction of I-JEPA; it is a compact version
for checking whether losses, gradients, EMA updates and representation variance
behave sensibly on CIFAR-10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package-style imports
    from .lightning_compat import L
    from .vit_cifar10 import PatchEmbed, TransformerBlock
    from .utils import cosine_warmup_lambda, make_param_groups
except ImportError:  # flat-folder imports
    from lightning_compat import L  # type: ignore
    from vit_cifar10 import PatchEmbed, TransformerBlock  # type: ignore
    from utils import cosine_warmup_lambda, make_param_groups  # type: ignore


@dataclass
class MiniIJEPAConfig:
    """Configuration for the small CIFAR I-JEPA sanity-check model."""

    image_size: int = 32
    patch_size: int = 4
    in_chans: int = 3

    # Online/target encoder.
    embed_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path_rate: float = 0.05

    # Predictor. Keep this smaller than the encoder for quick checks.
    predictor_dim: int = 128
    predictor_depth: int = 4
    predictor_heads: int = 4
    predictor_drop_path_rate: float = 0.0

    # Masking on the patch grid. With image_size=32, patch_size=4, grid=8.
    # Four 2x2 target blocks gives 16 target tokens and 48 context tokens.
    num_target_blocks: int = 4
    target_block_size: int = 2

    # Training knobs.
    loss: str = "mse"  # "mse" or "smooth_l1"
    ema_start: float = 0.996
    ema_end: float = 1.0

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def target_len(self) -> int:
        return self.num_target_blocks * self.target_block_size * self.target_block_size

    @property
    def context_len(self) -> int:
        return self.num_patches - self.target_len


class PatchTransformerEncoder(nn.Module):
    """ViT-style patch encoder without a [CLS] token.

    The user-provided classifier uses a [CLS] token for supervised training.
    I-JEPA pretraining is cleaner with patch tokens only: the target encoder
    returns one representation per image patch, and the loss is applied to the
    selected target patch representations.
    """

    def __init__(self, cfg: MiniIJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        if self.num_patches != cfg.num_patches:
            raise ValueError("PatchEmbed produced an unexpected number of patches.")

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        self.pos_drop = nn.Dropout(cfg.dropout)

        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.encoder_depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.encoder_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    drop_path_prob=dpr[i],
                )
                for i in range(cfg.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, images: torch.Tensor, token_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Encode all patch tokens, or only tokens at ``token_indices``.

        Args:
            images: Tensor of shape [B, C, H, W].
            token_indices: Optional LongTensor [B, K] with patch-token indices.

        Returns:
            Patch representations of shape [B, N, D] or [B, K, D].
        """
        tokens = self.patch_embed(images)  # [B, N, D]
        tokens = tokens + self.pos_embed
        if token_indices is not None:
            tokens = gather_tokens(tokens, token_indices)
        tokens = self.pos_drop(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class IjepaPredictor(nn.Module):
    """Small transformer predictor for target patch representations."""

    def __init__(self, cfg: MiniIJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.context_proj = nn.Linear(cfg.embed_dim, cfg.predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.predictor_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.predictor_dim))

        dpr = torch.linspace(0, cfg.predictor_drop_path_rate, cfg.predictor_depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.predictor_dim,
                    num_heads=cfg.predictor_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attn_dropout,
                    drop_path_prob=dpr[i],
                )
                for i in range(cfg.predictor_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.predictor_dim)
        self.out_proj = nn.Linear(cfg.predictor_dim, cfg.embed_dim)

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target-token representations.

        Args:
            context_tokens: Online encoder output [B, Kc, D].
            context_indices: Context patch indices [B, Kc].
            target_indices: Target patch indices [B, Kt].

        Returns:
            Predicted target representations [B, Kt, D].
        """
        batch_size = context_tokens.size(0)
        num_targets = target_indices.size(1)

        context_pos = gather_tokens(self.pos_embed.expand(batch_size, -1, -1), context_indices)
        target_pos = gather_tokens(self.pos_embed.expand(batch_size, -1, -1), target_indices)

        context_tokens = self.context_proj(context_tokens) + context_pos
        target_tokens = self.mask_token.expand(batch_size, num_targets, -1) + target_pos
        tokens = torch.cat([context_tokens, target_tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pred_targets = tokens[:, -num_targets:]
        return self.out_proj(pred_targets)


class MiniIJEPA(nn.Module):
    """Online encoder + EMA target encoder + predictor."""

    def __init__(self, cfg: MiniIJEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or MiniIJEPAConfig()
        self.context_encoder = PatchTransformerEncoder(self.cfg)
        self.target_encoder = PatchTransformerEncoder(self.cfg)
        self.predictor = IjepaPredictor(self.cfg)

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep the EMA encoder deterministic. Lightning will otherwise switch it
        # to train mode together with the online modules.
        self.target_encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        *,
        context_indices: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if target_indices is None or context_indices is None:
            target_indices, context_indices = sample_block_masks(
                batch_size=images.size(0),
                grid_size=self.cfg.grid_size,
                num_target_blocks=self.cfg.num_target_blocks,
                block_size=self.cfg.target_block_size,
                device=images.device,
            )

        context_tokens = self.context_encoder(images, context_indices)
        pred_tokens = self.predictor(context_tokens, context_indices, target_indices)

        with torch.no_grad():
            target_tokens_all = self.target_encoder(images)
            target_tokens = gather_tokens(target_tokens_all, target_indices)

        if self.cfg.loss == "mse":
            loss = F.mse_loss(pred_tokens, target_tokens)
        elif self.cfg.loss == "smooth_l1":
            loss = F.smooth_l1_loss(pred_tokens, target_tokens)
        else:
            raise ValueError(f"Unsupported JEPA loss: {self.cfg.loss!r}")

        return {
            "loss": loss,
            "pred_tokens": pred_tokens,
            "target_tokens": target_tokens,
            "context_indices": context_indices,
            "target_indices": target_indices,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA update: target <- m * target + (1-m) * context."""
        momentum = float(momentum)
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1].")
        for online, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, *, use_target: bool = True) -> torch.Tensor:
        """Return average-pooled patch representation for quick probing."""
        encoder = self.target_encoder if use_target else self.context_encoder
        tokens = encoder(images)
        return tokens.mean(dim=1)


class LitMiniIJEPA(L.LightningModule):
    """LightningModule for self-supervised I-JEPA sanity checking."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 3,
        predictor_dim: int = 128,
        predictor_depth: int = 4,
        predictor_heads: int = 4,
        num_target_blocks: int = 4,
        target_block_size: int = 2,
        lr: float = 3e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        epochs: int = 50,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = MiniIJEPAConfig(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            encoder_heads=encoder_heads,
            predictor_dim=predictor_dim,
            predictor_depth=predictor_depth,
            predictor_heads=predictor_heads,
            num_target_blocks=num_target_blocks,
            target_block_size=target_block_size,
            ema_start=ema_start,
            ema_end=ema_end,
        )
        model: nn.Module = MiniIJEPA(cfg)
        if compile_model:
            if hasattr(torch, "compile"):
                model = torch.compile(model)  # type: ignore[assignment]
            else:
                print("Warning: torch.compile is unavailable; ignoring compile_model")
        self.model = model
        self.example_input_array = torch.zeros(1, 3, image_size, image_size)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(images)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        out = self.model(images)
        loss = out["loss"]

        with torch.no_grad():
            pred_std = out["pred_tokens"].flatten(0, 1).std(dim=0).mean()
            target_std = out["target_tokens"].flatten(0, 1).std(dim=0).mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("train_pred_std", pred_std, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.size(0))
        self.log("train_target_std", target_std, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.size(0))
        self.log("ema_momentum", torch.tensor(self.current_ema_momentum(), device=self.device), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        out = self.model(images)
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        # Called after optimizer step in Lightning's automatic optimization loop.
        if hasattr(self.model, "update_target_encoder"):
            self.model.update_target_encoder(self.current_ema_momentum())

    def current_ema_momentum(self) -> float:
        start = float(self.hparams.ema_start)
        end = float(self.hparams.ema_end)
        if self.trainer is None:
            return start
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        t = min(1.0, float(self.global_step) / float(total_steps))
        # Cosine interpolation from start to end.
        return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * t))

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
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


@torch.no_grad()
def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather token rows from [B, N, D] using indices [B, K]."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, N, D]")
    if indices.ndim != 2:
        raise ValueError("indices must have shape [B, K]")
    return tokens.gather(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))


def sample_block_masks(
    *,
    batch_size: int,
    grid_size: int,
    num_target_blocks: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample non-overlapping square target blocks on a patch grid.

    Returns:
        target_indices: [B, Kt]
        context_indices: [B, N-Kt]

    This is deliberately stricter and simpler than the original I-JEPA sampler:
    it keeps target and context lengths fixed, which avoids custom collation.
    """
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if block_size <= 0 or block_size > grid_size:
        raise ValueError("block_size must be in [1, grid_size]")
    num_patches = grid_size * grid_size
    target_len = num_target_blocks * block_size * block_size
    if target_len >= num_patches:
        raise ValueError("target blocks cover all patches; no context tokens remain")

    target_indices = torch.empty(batch_size, target_len, dtype=torch.long, device=device)
    context_indices = torch.empty(batch_size, num_patches - target_len, dtype=torch.long, device=device)
    all_indices = torch.arange(num_patches, device=device)

    for b in range(batch_size):
        occupied = torch.zeros(num_patches, dtype=torch.bool, device=device)
        blocks_sampled = 0
        attempts = 0
        max_attempts = 100 * max(1, num_target_blocks)
        while blocks_sampled < num_target_blocks and attempts < max_attempts:
            attempts += 1
            top = int(torch.randint(0, grid_size - block_size + 1, (), device=device).item())
            left = int(torch.randint(0, grid_size - block_size + 1, (), device=device).item())
            rows = torch.arange(top, top + block_size, device=device)
            cols = torch.arange(left, left + block_size, device=device)
            block = (rows[:, None] * grid_size + cols[None, :]).flatten()
            if not occupied[block].any():
                occupied[block] = True
                blocks_sampled += 1

        # Fallback for very dense masks: fill any missing target slots randomly.
        current_count = int(occupied.sum().item())
        if current_count < target_len:
            remaining = all_indices[~occupied]
            perm = torch.randperm(remaining.numel(), device=device)
            fill = remaining[perm[: target_len - current_count]]
            occupied[fill] = True

        t_idx = all_indices[occupied]
        c_idx = all_indices[~occupied]
        # Sort gives stable spatial order, useful for debugging visualizations.
        target_indices[b] = t_idx.sort().values[:target_len]
        context_indices[b] = c_idx.sort().values

    return target_indices, context_indices


def _smoke_test() -> None:
    """Run this file directly for a CPU shape/gradient sanity check."""
    torch.manual_seed(0)
    model = MiniIJEPA(MiniIJEPAConfig(embed_dim=96, encoder_depth=2, encoder_heads=3, predictor_dim=64, predictor_depth=2))
    images = torch.randn(4, 3, 32, 32)
    out = model(images)
    out["loss"].backward()
    model.update_target_encoder(0.996)
    print("loss:", float(out["loss"].detach()))
    print("pred_tokens:", tuple(out["pred_tokens"].shape))
    print("target_tokens:", tuple(out["target_tokens"].shape))
    print("context_indices:", tuple(out["context_indices"].shape))
    print("target_indices:", tuple(out["target_indices"].shape))


if __name__ == "__main__":
    _smoke_test()
