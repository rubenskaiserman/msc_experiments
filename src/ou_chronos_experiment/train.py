from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ou_chronos_experiment.data import DataConfig, build_datasets
from ou_chronos_experiment.model import TinyChronos2Inspired, pinball_loss, robust_scale_batch


@dataclass
class ModelConfig:
    context_length: int = 100
    horizon: int = 100
    forecast_chunk_length: int = 100
    patch_len: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.10
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.5, 0.8, 0.9, 0.95)


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    min_delta: float = 1e-4
    checkpoint_dir: str = ""
    checkpoint_name: str = "best"
    log_dir: str = "lightning_logs"
    experiment_name: str = "ou_chronos"
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _require_lightning() -> Any:
    try:
        return importlib.import_module("lightning.pytorch")
    except Exception as exc:
        raise RuntimeError(
            "PyTorch Lightning is required for training. Install/fix it with something like "
            "`pip install lightning tensorboard`. If Lightning is already installed, check that "
            "`torch`, `torchvision`, and `torchaudio` versions match; the current environment "
            f"failed to import Lightning with: {type(exc).__name__}: {exc}"
        ) from exc


def _require_lightning_component(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import Lightning component {module}.{name}. "
            "Check the Lightning/TensorBoard installation."
        ) from exc


def full_horizon_teacher_forced_loss(
    model: TinyChronos2Inspired,
    x: torch.Tensor,
    y: torch.Tensor,
    model_cfg: ModelConfig,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    chunk = model_cfg.forecast_chunk_length
    n_chunks = model_cfg.horizon // chunk
    context = x
    losses = []
    for k in range(n_chunks):
        target = y[:, k * chunk:(k + 1) * chunk]
        x_s, target_s, _, _ = robust_scale_batch(context, target)
        pred_q = model(x_s)
        losses.append(pinball_loss(pred_q, target_s, quantiles))
        context = target.detach()
    return torch.stack(losses).mean()


def make_lightning_module(pl: Any, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig) -> Any:
    class OUForecastModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.data_cfg = data_cfg
            self.model_cfg = model_cfg
            self.train_cfg = train_cfg
            self.model = TinyChronos2Inspired(
                context_length=model_cfg.context_length,
                forecast_chunk_length=model_cfg.forecast_chunk_length,
                patch_len=model_cfg.patch_len,
                quantiles=model_cfg.quantiles,
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                n_layers=model_cfg.n_layers,
                dropout=model_cfg.dropout,
            )
            self.register_buffer(
                "quantiles_tensor",
                torch.tensor(model_cfg.quantiles, dtype=torch.float32),
                persistent=False,
            )
            self.save_hyperparameters(
                {
                    "data_cfg": asdict(data_cfg),
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(train_cfg),
                }
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
            x, y = batch
            loss = full_horizon_teacher_forced_loss(
                self.model,
                x,
                y,
                self.model_cfg,
                self.quantiles_tensor,
            )
            self.log(
                f"{stage}_loss",
                loss,
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=True,
                batch_size=x.shape[0],
            )
            return loss

        def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            return self._step(batch, "train")

        def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
            del batch_idx
            return self._step(batch, "val")

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                self.parameters(),
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
            )

        def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            checkpoint["data_cfg"] = asdict(self.data_cfg)
            checkpoint["model_cfg"] = asdict(self.model_cfg)
            checkpoint["train_cfg"] = asdict(self.train_cfg)

    return OUForecastModule()


def train_model(
    data_cfg: DataConfig | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
) -> tuple[TinyChronos2Inspired, pd.DataFrame, dict[str, object], Path]:
    pl = _require_lightning()
    EarlyStopping = _require_lightning_component("lightning.pytorch.callbacks", "EarlyStopping")
    ModelCheckpoint = _require_lightning_component("lightning.pytorch.callbacks", "ModelCheckpoint")
    TensorBoardLogger = _require_lightning_component("lightning.pytorch.loggers", "TensorBoardLogger")
    CSVLogger = _require_lightning_component("lightning.pytorch.loggers", "CSVLogger")

    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    pl.seed_everything(train_cfg.seed, workers=True)
    datasets = build_datasets(data_cfg)

    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        persistent_workers=train_cfg.num_workers > 0,
    )

    module = make_lightning_module(pl, data_cfg, model_cfg, train_cfg)
    checkpoint_dir = Path(train_cfg.checkpoint_dir) if train_cfg.checkpoint_dir else None
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=train_cfg.checkpoint_name,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=train_cfg.early_stopping_patience,
        min_delta=train_cfg.min_delta,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=train_cfg.log_dir,
        name=train_cfg.experiment_name,
    )
    csv_logger = CSVLogger(
        save_dir=train_cfg.log_dir,
        name=f"{train_cfg.experiment_name}_csv",
    )

    accelerator = "gpu" if train_cfg.device == "cuda" else "cpu"
    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[tensorboard_logger, csv_logger],
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    print(f"DEVICE: {train_cfg.device}")
    print(f"parameters: {count_parameters(module.model):,}")
    print(f"train windows: {len(datasets['train_dataset']):,}")
    print(f"val windows: {len(datasets['val_dataset']):,}")
    print(f"TensorBoard logs: {Path(train_cfg.log_dir) / train_cfg.experiment_name}")

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    fallback_checkpoint = Path(train_cfg.checkpoint_dir) / f"{train_cfg.checkpoint_name}.ckpt"
    checkpoint_path = Path(checkpoint_callback.best_model_path or fallback_checkpoint)
    history_path = Path(csv_logger.log_dir) / "metrics.csv"
    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    best_model, _ = load_model_from_checkpoint(checkpoint_path, device=train_cfg.device)
    return best_model, history_df, datasets, checkpoint_path


def _strip_lightning_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("model.") for key in state_dict):
        return {key.removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def load_model_from_checkpoint(path: str | Path, device: str | None = None) -> tuple[TinyChronos2Inspired, dict]:
    checkpoint = torch.load(path, map_location=device or "cpu")
    model_cfg_raw = checkpoint.get("model_cfg")
    if model_cfg_raw is None:
        model_cfg_raw = checkpoint.get("hyper_parameters", {}).get("model_cfg")
    if model_cfg_raw is None:
        raise KeyError(f"Checkpoint {path} does not contain model_cfg.")

    model_cfg = ModelConfig(**model_cfg_raw)
    model = TinyChronos2Inspired(
        context_length=model_cfg.context_length,
        forecast_chunk_length=model_cfg.forecast_chunk_length,
        patch_len=model_cfg.patch_len,
        quantiles=model_cfg.quantiles,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        dropout=model_cfg.dropout,
    )
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = _strip_lightning_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.to(device or "cpu")
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OU TinyChronos model with PyTorch Lightning.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--checkpoint-dir", type=str, default=TrainConfig.checkpoint_dir)
    parser.add_argument("--checkpoint-name", type=str, default=TrainConfig.checkpoint_name)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--device", type=str, default=TrainConfig.device, choices=("cpu", "cuda"))
    parser.add_argument("--log-dir", type=str, default=TrainConfig.log_dir)
    parser.add_argument("--experiment-name", type=str, default=TrainConfig.experiment_name)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        batch_size=args.batch_size,
        device=args.device,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        num_workers=args.num_workers,
    )
    try:
        _, history_df, _, checkpoint_path = train_model(data_cfg, model_cfg, train_cfg)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    history_path = checkpoint_path.with_name("history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"saved best Lightning checkpoint: {checkpoint_path}")
    print(f"saved history copy: {history_path}")
    print(f"view logs with: tensorboard --logdir {train_cfg.log_dir}")


if __name__ == "__main__":
    main()
