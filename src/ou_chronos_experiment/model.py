from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TinyChronos2Inspired(nn.Module):
    def __init__(
        self,
        context_length: int,
        forecast_chunk_length: int,
        patch_len: int,
        quantiles: tuple[float, ...],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.forecast_chunk_length = forecast_chunk_length
        self.patch_len = patch_len
        self.quantiles = tuple(quantiles)
        self.n_quantiles = len(quantiles)
        self.n_patches = context_length // patch_len

        self.patch_embed = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, forecast_chunk_length * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        h = self.patch_embed(patches)
        h = h + self.pos_embed
        h = self.encoder(h)
        last = h[:, -1, :]
        out = self.head(last).view(batch_size, self.forecast_chunk_length, self.n_quantiles)
        out, _ = torch.sort(out, dim=-1)
        return out


def robust_scale_batch(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loc = x.median(dim=1, keepdim=True).values
    mad = (x - loc).abs().median(dim=1, keepdim=True).values
    scale = (1.4826 * mad).clamp_min(eps)
    x_s = torch.asinh((x - loc) / scale)

    if y is None:
        return x_s, loc, scale

    y_s = torch.asinh((y - loc) / scale)
    return x_s, y_s, loc, scale


def inverse_robust_scale(z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return loc + scale * torch.sinh(z)


def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    q = quantiles.view(1, 1, -1)
    errors = target.unsqueeze(-1) - pred_q
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    return loss.mean()


@torch.no_grad()
def predict_quantiles_batch(
    model: TinyChronos2Inspired,
    x_batch: np.ndarray,
    horizon: int,
    forecast_chunk_length: int,
    quantiles: tuple[float, ...],
    device: str,
) -> np.ndarray:
    model.eval()
    base_context = torch.tensor(x_batch, dtype=torch.float32, device=device)
    median_context = base_context
    lower_context = base_context
    upper_context = base_context
    chunks = []
    q05_idx = quantiles.index(0.05)
    median_idx = quantiles.index(0.5)
    q95_idx = quantiles.index(0.95)
    n_chunks = horizon // forecast_chunk_length

    for _ in range(n_chunks):
        contexts = torch.cat([lower_context, median_context, upper_context], dim=0)
        x_s, loc, scale = robust_scale_batch(contexts)
        pred_q_s = model(x_s)
        pred_q = inverse_robust_scale(pred_q_s, loc.unsqueeze(-1), scale.unsqueeze(-1))
        lower_pred, median_pred, upper_pred = pred_q.chunk(3, dim=0)

        median_chunk = median_pred[:, :, median_idx]
        lower_width = median_chunk - lower_pred[:, :, q05_idx]
        upper_width = upper_pred[:, :, q95_idx] - median_chunk

        if chunks:
            previous = chunks[-1]
            previous_median = previous[:, -forecast_chunk_length:, median_idx]
            previous_lower_width = previous_median - previous[:, -forecast_chunk_length:, q05_idx]
            previous_upper_width = previous[:, -forecast_chunk_length:, q95_idx] - previous_median
            lower_width = lower_width + previous_lower_width
            upper_width = upper_width + previous_upper_width

        adjusted_chunk = median_pred.clone()
        adjusted_chunk[:, :, q05_idx] = median_chunk - lower_width.clamp_min(0.0)
        adjusted_chunk[:, :, q95_idx] = median_chunk + upper_width.clamp_min(0.0)

        if 0.1 in quantiles:
            q10_idx = quantiles.index(0.1)
            adjusted_chunk[:, :, q10_idx] = torch.minimum(
                adjusted_chunk[:, :, q10_idx],
                adjusted_chunk[:, :, q05_idx] + 0.5 * lower_width.clamp_min(0.0),
            )
        if 0.9 in quantiles:
            q90_idx = quantiles.index(0.9)
            adjusted_chunk[:, :, q90_idx] = torch.maximum(
                adjusted_chunk[:, :, q90_idx],
                adjusted_chunk[:, :, q95_idx] - 0.5 * upper_width.clamp_min(0.0),
            )

        adjusted_chunk, _ = torch.sort(adjusted_chunk, dim=-1)
        chunks.append(adjusted_chunk)
        lower_context = adjusted_chunk[:, :, q05_idx]
        median_context = median_chunk
        upper_context = adjusted_chunk[:, :, q95_idx]

    return torch.cat(chunks, dim=1).cpu().numpy()


def predict_quantiles(
    model: TinyChronos2Inspired,
    x_np: np.ndarray,
    horizon: int,
    forecast_chunk_length: int,
    quantiles: tuple[float, ...],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    preds = []
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        preds.append(
            predict_quantiles_batch(
                model,
                x_np[start:end],
                horizon,
                forecast_chunk_length,
                quantiles,
                device,
            )
        )
    return np.concatenate(preds, axis=0)

