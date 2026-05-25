"""Configuration dataclasses for The Well VideoMAE experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def notebooks_dir() -> Path:
    return repo_root() / "notebooks"


def detect_data_base() -> str:
    candidates = [
        os.environ.get("THE_WELL_BASE_PATH"),
        repo_root() / "the_well_data",
        notebooks_dir() / "the_well_data",
        Path.home() / "the_well_data",
        Path.home() / "data" / "the_well_data",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())
    return str((repo_root() / "the_well_data").resolve())


@dataclass
class DataConfig:
    data_base: str = field(default_factory=detect_data_base)
    dataset_name: str = "shear_flow"
    use_hf_streaming: bool = False
    download_data: bool = False
    num_frames: int = 16
    crop_to_square: bool = True
    crop_mode: str = "left"
    spatial_size: tuple[int, int] = (128, 128)
    max_train_windows: int | None = 8192
    max_val_windows: int | None = 1024
    max_test_windows: int | None = 1024
    restriction_seed: int = 42
    num_workers: int = 2
    batch_size: int = 1
    target_names: tuple[str, ...] = ("reynolds", "schmidt")
    label_transform: str = "log10_zscore"
    stats_max_batches: int = 256
    label_stats_max_batches: int = 128


@dataclass
class ModelConfig:
    patch_size: int = 16
    tubelet_size: int = 2
    encoder_embed_dim: int = 384
    encoder_depth: int = 6
    encoder_heads: int = 6
    decoder_embed_dim: int = 192
    decoder_depth: int = 2
    decoder_heads: int = 3
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.90
    tube_masking: bool = True
    norm_pix_loss: bool = False


@dataclass
class OptimizationConfig:
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.05
    grad_clip_norm: float | None = 1.0


@dataclass
class ProbeConfig:
    num_queries: int = 4
    heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-2


@dataclass
class TrainerConfig:
    max_epochs: int = 6
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 8
    log_every_n_steps: int = 10
    max_train_steps_per_epoch: int | None = 300
    max_eval_steps: int | None = 100
    seed: int = 42


@dataclass
class ProbeTrainerConfig(TrainerConfig):
    max_epochs: int = 100
    accumulate_grad_batches: int = 1
    max_train_steps_per_epoch: int | None = 300


@dataclass
class ProjectConfig:
    run_name: str = "videomae_shear_flow"
    output_dir: str = str(repo_root() / "runs" / "videomae_shear_flow")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

