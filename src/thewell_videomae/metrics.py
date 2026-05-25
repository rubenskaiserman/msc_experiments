"""Metrics for reconstruction and probe evaluation."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import r2_score

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def nmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.mean((a - b) ** 2) / (torch.mean(b**2) + eps)


def vrmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2) / (torch.mean((b - b.mean()) ** 2) + eps))


def linf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.abs(a - b))


def inverse_label_transform(y_norm: torch.Tensor, label_mean: torch.Tensor, label_std: torch.Tensor, label_transform: str) -> torch.Tensor:
    if "zscore" in label_transform:
        return y_norm * label_std + label_mean
    return y_norm


def probe_metrics(
    yhat_norm: torch.Tensor,
    y_norm: torch.Tensor,
    yraw: torch.Tensor,
    *,
    label_names: list[str],
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    label_transform: str,
) -> dict[str, float]:
    yhat_pre = inverse_label_transform(yhat_norm, label_mean, label_std, label_transform)
    y_pre = inverse_label_transform(y_norm, label_mean, label_std, label_transform)
    out: dict[str, float] = {}
    out["mse_normalized_avg"] = float(F.mse_loss(yhat_norm, y_norm).item())
    out["rmse_normalized_avg"] = float(torch.sqrt(F.mse_loss(yhat_norm, y_norm)).item())
    for idx, name in enumerate(label_names):
        out[f"mse_normalized_{name}"] = float(torch.mean((yhat_norm[:, idx] - y_norm[:, idx]) ** 2))
        out[f"mae_normalized_{name}"] = float(torch.mean(torch.abs(yhat_norm[:, idx] - y_norm[:, idx])))
        out[f"rmse_pre_zscore_{name}"] = float(torch.sqrt(torch.mean((yhat_pre[:, idx] - y_pre[:, idx]) ** 2)))
        out[f"mae_pre_zscore_{name}"] = float(torch.mean(torch.abs(yhat_pre[:, idx] - y_pre[:, idx])))
        if label_transform.startswith("log10"):
            raw_pred = torch.pow(10.0, yhat_pre[:, idx])
            raw_true = yraw[:, idx]
            out[f"relative_mae_raw_{name}"] = float(torch.mean(torch.abs(raw_pred - raw_true) / raw_true.clamp_min(1e-12)))
    if SKLEARN_AVAILABLE:
        out["r2_normalized_avg"] = float(r2_score(y_norm.numpy(), yhat_norm.numpy(), multioutput="uniform_average"))
    return out


def reconstruction_metrics_dataframe(model, loader, mask_ratio: float, max_steps: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= total_steps:
                break
            x = batch["video"].to(device)
            recon, _, _, _ = model.reconstruct(x, mask_ratio=mask_ratio)
            rows.append(
                {
                    "mse_blended_reconstruction": float(mse(recon, x).cpu()),
                    "rmse_blended_reconstruction": float(torch.sqrt(mse(recon, x)).cpu()),
                    "nmse_blended_reconstruction": float(nmse(recon, x).cpu()),
                    "vrmse_blended_reconstruction": float(vrmse(recon, x).cpu()),
                    "linf_blended_reconstruction": float(linf(recon, x).cpu()),
                }
            )
    df = pd.DataFrame(rows)
    return pd.DataFrame([df.mean().to_dict()]) if not df.empty else pd.DataFrame()

