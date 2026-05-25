"""VideoMAE backbone and attentive probe."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    positions = np.arange(n_position)[:, None]
    dims = np.arange(d_hid)[None, :]
    angle_rates = 1.0 / np.power(10000, (2 * (dims // 2)) / d_hid)
    angles = positions * angle_rates
    table = np.zeros((n_position, d_hid), dtype=np.float32)
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_frames, tubelet_size):
        super().__init__()
        height, width = img_size
        assert height % patch_size == 0 and width % patch_size == 0
        assert num_frames % tubelet_size == 0
        self.img_size = (height, width)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (num_frames // tubelet_size, height // patch_size, width // patch_size)
        self.num_patches = int(np.prod(self.grid_size))
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, height, width = x.shape
        assert (height, width) == self.img_size, f"Expected {self.img_size}, got {(height, width)}"
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VideoMAE(nn.Module):
    def __init__(
        self,
        *,
        img_size: tuple[int, int],
        num_frames: int,
        in_chans: int,
        patch_size: int = 16,
        tubelet_size: int = 2,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 2,
        decoder_heads: int = 3,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        tube_masking: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size, patch_size, in_chans, encoder_embed_dim, num_frames, tubelet_size)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.norm_pix_loss = norm_pix_loss
        self.tube_masking = tube_masking
        self.encoder_embed_dim = encoder_embed_dim

        self.register_buffer("pos_embed", sinusoid_encoding(self.num_patches, encoder_embed_dim), persistent=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=encoder_embed_dim,
            nhead=encoder_heads,
            dim_feedforward=int(encoder_embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.register_buffer(
            "decoder_pos_embed",
            sinusoid_encoding(self.num_patches, decoder_embed_dim),
            persistent=False,
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_dim = in_chans * tubelet_size * patch_size * patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def make_mask(self, batch_size: int, mask_ratio: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tp, hp, wp = self.grid_size
        num_tokens = tp * hp * wp
        if self.tube_masking:
            spatial_tokens = hp * wp
            keep_spatial = max(1, int(spatial_tokens * (1.0 - mask_ratio)))
            noise = torch.rand(batch_size, spatial_tokens, device=device)
            ids_spatial = torch.argsort(noise, dim=1)[:, :keep_spatial]
            ids_keep = torch.cat([ids_spatial + t * spatial_tokens for t in range(tp)], dim=1)
        else:
            keep_tokens = max(1, int(num_tokens * (1.0 - mask_ratio)))
            noise = torch.rand(batch_size, num_tokens, device=device)
            ids_keep = torch.argsort(noise, dim=1)[:, :keep_tokens]
        mask = torch.ones(batch_size, num_tokens, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        return ids_keep.long(), mask

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        tube = self.tubelet_size
        batch, channels, frames, height, width = imgs.shape
        tp, hp, wp = self.grid_size
        assert frames == tp * tube and height == hp * patch and width == wp * patch
        x = imgs.reshape(batch, channels, tp, tube, hp, patch, wp, patch)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(batch, tp * hp * wp, channels * tube * patch * patch)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        patch = self.patch_size
        tube = self.tubelet_size
        tp, hp, wp = self.grid_size
        batch, num_tokens, _ = patches.shape
        channels = self.in_chans
        assert num_tokens == tp * hp * wp
        x = patches.reshape(batch, tp, hp, wp, channels, tube, patch, patch)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(batch, channels, tp * tube, hp * patch, wp * patch)

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        _, _, dim = x.shape
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        ids_keep, mask = self.make_mask(x.shape[0], mask_ratio, x.device)
        x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        x_vis = self.encoder(x_vis)
        x_vis = self.encoder_norm(x_vis)
        return x_vis, mask, ids_keep

    def forward_decoder(self, x_vis: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        x_vis = self.decoder_embed(x_vis)
        batch, _, dim = x_vis.shape
        full = self.mask_token.to(dtype=x_vis.dtype, device=x_vis.device).expand(batch, self.num_patches, -1).clone()
        full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, dim), x_vis)
        full = full + self.decoder_pos_embed.to(device=full.device, dtype=full.dtype)
        full = self.decoder(full)
        full = self.decoder_norm(full)
        return self.decoder_pred(full)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.90) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_vis, mask, ids_keep = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x_vis, ids_keep)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor, mask_ratio: float = 0.90) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()
        x_vis, mask, ids_keep = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x_vis, ids_keep)
        target = self.patchify(imgs)
        mixed = target.clone()
        mask_bool = mask.bool().unsqueeze(-1).expand_as(mixed)
        mixed[mask_bool] = pred[mask_bool]
        recon = self.unpatchify(mixed)
        masked = self.unpatchify(target * (1.0 - mask.unsqueeze(-1)))
        if was_training:
            self.train()
        return recon, masked, pred, mask

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x


class AttentiveProbe(nn.Module):
    def __init__(self, dim: int, out_dim: int = 2, num_queries: int = 4, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(dim * num_queries),
            nn.Linear(dim * num_queries, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        pooled = self.norm(pooled)
        return self.head(pooled)

