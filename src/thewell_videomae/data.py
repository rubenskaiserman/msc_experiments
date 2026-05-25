"""Data download, dataset wrappers, and LightningDataModule for The Well."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .config import DataConfig
from .lightning_compat import L
from .utils import write_json

try:
    from the_well.data import WellDataset
    from the_well.utils.download import well_download

    THE_WELL_AVAILABLE = True
except Exception as exc:
    WellDataset = None
    well_download = None
    THE_WELL_AVAILABLE = False
    THE_WELL_IMPORT_ERROR = exc


def ensure_the_well_available() -> None:
    if not THE_WELL_AVAILABLE:
        raise RuntimeError(
            "the_well is not installed. Install it with `pip install the_well` before running this project."
        ) from THE_WELL_IMPORT_ERROR


def flatten_field_names(metadata) -> list[str]:
    return [name for order in sorted(metadata.field_names) for name in metadata.field_names[order]]


def norm_name(name: str) -> str:
    return str(name).lower().replace(" ", "_").replace("-", "_")


def infer_label_indices(metadata, target_names: tuple[str, ...]) -> tuple[list[int], list[str]]:
    names = list(metadata.constant_scalar_names)
    normalized = [norm_name(name) for name in names]
    aliases = {
        "reynolds": ["reynolds", "reynolds_number", "re"],
        "schmidt": ["schmidt", "schmidt_number", "sc"],
    }
    indices: list[int] = []
    for target in target_names:
        candidates = aliases.get(target, [target])
        found = None
        for candidate in candidates:
            if candidate in normalized:
                found = normalized.index(candidate)
                break
        if found is None:
            for idx, normalized_name in enumerate(normalized):
                if target in normalized_name:
                    found = idx
                    break
        if found is None:
            raise ValueError(
                f"Could not find target {target!r} in constant scalars {names}. "
                "Inspect dataset metadata and set target_names explicitly."
            )
        indices.append(found)
    return indices, [names[idx] for idx in indices]


def has_hdf5_files(path: Path) -> bool:
    return path.exists() and (any(path.glob("*.h5")) or any(path.glob("*.hdf5")))


def resolve_well_dataset_root(cfg: DataConfig) -> str:
    ensure_the_well_available()
    data_base = Path(cfg.data_base).expanduser().resolve()
    well_base_path = data_base / "datasets"
    if cfg.download_data:
        for split in ("train", "valid", "test"):
            well_download(base_path=str(data_base), dataset=cfg.dataset_name, split=split)
    if cfg.use_hf_streaming:
        return "hf://datasets/polymathic-ai/"
    train_dir = well_base_path / cfg.dataset_name / "data" / "train"
    if not has_hdf5_files(train_dir):
        raise FileNotFoundError(
            f"No local The Well files found for {cfg.dataset_name!r}. Expected HDF5 files under {train_dir}. "
            f"Set data_base to the directory containing ./datasets, enable download_data, or set use_hf_streaming=True."
        )
    return str(well_base_path)


def download_dataset(base_path: str | Path, dataset_name: str, splits: tuple[str, ...] = ("train", "valid", "test")) -> None:
    ensure_the_well_available()
    for split in splits:
        well_download(base_path=str(Path(base_path).expanduser().resolve()), dataset=dataset_name, split=split)


def make_well_dataset(cfg: DataConfig, split: str, restrict_num_samples: int | float | None, dataset_root: str):
    ensure_the_well_available()
    return WellDataset(
        well_base_path=dataset_root,
        well_dataset_name=cfg.dataset_name,
        well_split_name=split,
        n_steps_input=cfg.num_frames,
        n_steps_output=1,
        use_normalization=False,
        restrict_num_samples=restrict_num_samples,
        restriction_seed=cfg.restriction_seed,
        return_grid=False,
        boundary_return_type=None,
    )


class ShearFlowVideoDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        label_indices: list[int],
        *,
        spatial_size: tuple[int, int] | None = (128, 128),
        crop_to_square: bool = True,
        crop_mode: str = "left",
        field_mean: torch.Tensor | None = None,
        field_std: torch.Tensor | None = None,
        label_mean: torch.Tensor | None = None,
        label_std: torch.Tensor | None = None,
        label_transform: str = "log10_zscore",
    ):
        self.base = base_dataset
        self.label_indices = list(label_indices)
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None
        self.crop_to_square = bool(crop_to_square)
        self.crop_mode = crop_mode
        self.field_mean = field_mean
        self.field_std = field_std
        self.label_mean = label_mean
        self.label_std = label_std
        self.label_transform = label_transform
        self.field_names = flatten_field_names(self.base.metadata)
        self.label_names = [self.base.metadata.constant_scalar_names[idx] for idx in self.label_indices]

    def __len__(self) -> int:
        return len(self.base)

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.crop_to_square:
            return x
        height, width = int(x.shape[1]), int(x.shape[2])
        side = min(height, width)
        if self.crop_mode == "left":
            h0, w0 = 0, 0
        elif self.crop_mode == "center":
            h0 = (height - side) // 2
            w0 = (width - side) // 2
        else:
            raise ValueError("crop_mode must be 'left' or 'center'")
        return x[:, h0 : h0 + side, w0 : w0 + side, :]

    def _video_from_item(self, item) -> torch.Tensor:
        x = torch.as_tensor(item["input_fields"]).float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self._crop(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.spatial_size is not None and tuple(x.shape[-2:]) != tuple(self.spatial_size):
            x = F.interpolate(x, size=self.spatial_size, mode="bilinear", align_corners=False)
        x = x.permute(1, 0, 2, 3).contiguous()
        if self.field_mean is not None and self.field_std is not None:
            mean = self.field_mean[:, None, None, None]
            std = self.field_std[:, None, None, None]
            x = (x - mean) / std.clamp_min(1e-6)
        return x

    def _constant_scalar_vector(self, item) -> torch.Tensor:
        scalars = item["constant_scalars"]
        if isinstance(scalars, dict):
            values = []
            for name in self.base.metadata.constant_scalar_names:
                value = torch.as_tensor(scalars[name]).float().reshape(-1)
                values.append(value[0])
            return torch.stack(values)
        return torch.as_tensor(scalars).float().reshape(-1)

    def _label_from_item(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        y_raw = self._constant_scalar_vector(item)[self.label_indices]
        if self.label_transform.startswith("log10"):
            y = torch.log10(y_raw.clamp_min(1e-12))
        else:
            y = y_raw.clone()
        if "zscore" in self.label_transform and self.label_mean is not None and self.label_std is not None:
            y = (y - self.label_mean) / self.label_std.clamp_min(1e-6)
        return y, y_raw

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        video = self._video_from_item(item)
        label, label_raw = self._label_from_item(item)
        return {"video": video, "label": label, "label_raw": label_raw, "index": torch.tensor(idx)}


def estimate_field_stats(dataset: Dataset, max_batches: int = 256, num_workers: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=False)
    running_sum = None
    running_sq_sum = None
    n = 0
    for idx, batch in enumerate(loader):
        x = batch["video"].float()
        if running_sum is None:
            channels = x.shape[1]
            running_sum = torch.zeros(channels)
            running_sq_sum = torch.zeros(channels)
        dims = (0, 2, 3, 4)
        running_sum += x.sum(dim=dims).cpu()
        running_sq_sum += (x * x).sum(dim=dims).cpu()
        n += x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]
        if idx + 1 >= max_batches:
            break
    mean = running_sum / max(n, 1)
    var = (running_sq_sum / max(n, 1) - mean**2).clamp_min(1e-12)
    return mean, torch.sqrt(var).clamp_min(1e-6)


def estimate_label_stats(dataset: Dataset, max_batches: int = 128, num_workers: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=False)
    ys: list[torch.Tensor] = []
    for idx, batch in enumerate(loader):
        ys.append(batch["label"].float())
        if idx + 1 >= max_batches:
            break
    y = torch.cat(ys, dim=0)
    return y.mean(dim=0), y.std(dim=0).clamp_min(1e-6)


class TheWellVideoDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig, *, normalization_dir: str | Path | None = None):
        super().__init__()
        self.cfg = cfg
        self.normalization_dir = Path(normalization_dir).expanduser().resolve() if normalization_dir else None
        self.dataset_root: str | None = None
        self.train_base = None
        self.valid_base = None
        self.test_base = None
        self.train_dataset: ShearFlowVideoDataset | None = None
        self.valid_dataset: ShearFlowVideoDataset | None = None
        self.test_dataset: ShearFlowVideoDataset | None = None
        self.label_indices: list[int] | None = None
        self.label_names: list[str] | None = None
        self.field_mean: torch.Tensor | None = None
        self.field_std: torch.Tensor | None = None
        self.label_mean: torch.Tensor | None = None
        self.label_std: torch.Tensor | None = None
        self.field_names: list[str] | None = None

    def prepare_data(self) -> None:
        self.dataset_root = resolve_well_dataset_root(self.cfg)

    def setup(self, stage: str | None = None) -> None:
        if self.dataset_root is None:
            self.dataset_root = resolve_well_dataset_root(self.cfg)
        random.seed(self.cfg.restriction_seed)
        np.random.seed(self.cfg.restriction_seed)
        torch.manual_seed(self.cfg.restriction_seed)
        self.train_base = make_well_dataset(self.cfg, "train", self.cfg.max_train_windows, self.dataset_root)
        self.valid_base = make_well_dataset(self.cfg, "valid", self.cfg.max_val_windows, self.dataset_root)
        self.test_base = make_well_dataset(self.cfg, "test", self.cfg.max_test_windows, self.dataset_root)
        self.label_indices, self.label_names = infer_label_indices(self.train_base.metadata, self.cfg.target_names)
        temp_train = ShearFlowVideoDataset(
            self.train_base,
            self.label_indices,
            spatial_size=self.cfg.spatial_size,
            crop_to_square=self.cfg.crop_to_square,
            crop_mode=self.cfg.crop_mode,
            label_transform=self.cfg.label_transform,
        )
        self.field_mean, self.field_std = estimate_field_stats(
            temp_train,
            max_batches=min(self.cfg.stats_max_batches, len(temp_train)),
            num_workers=0,
        )
        self.label_mean, self.label_std = estimate_label_stats(
            temp_train,
            max_batches=self.cfg.label_stats_max_batches,
            num_workers=0,
        )
        self.train_dataset = ShearFlowVideoDataset(
            self.train_base,
            self.label_indices,
            spatial_size=self.cfg.spatial_size,
            crop_to_square=self.cfg.crop_to_square,
            crop_mode=self.cfg.crop_mode,
            field_mean=self.field_mean,
            field_std=self.field_std,
            label_mean=self.label_mean,
            label_std=self.label_std,
            label_transform=self.cfg.label_transform,
        )
        self.valid_dataset = ShearFlowVideoDataset(
            self.valid_base,
            self.label_indices,
            spatial_size=self.cfg.spatial_size,
            crop_to_square=self.cfg.crop_to_square,
            crop_mode=self.cfg.crop_mode,
            field_mean=self.field_mean,
            field_std=self.field_std,
            label_mean=self.label_mean,
            label_std=self.label_std,
            label_transform=self.cfg.label_transform,
        )
        self.test_dataset = ShearFlowVideoDataset(
            self.test_base,
            self.label_indices,
            spatial_size=self.cfg.spatial_size,
            crop_to_square=self.cfg.crop_to_square,
            crop_mode=self.cfg.crop_mode,
            field_mean=self.field_mean,
            field_std=self.field_std,
            label_mean=self.label_mean,
            label_std=self.label_std,
            label_transform=self.cfg.label_transform,
        )
        self.field_names = self.train_dataset.field_names
        if self.normalization_dir is not None:
            payload = {
                "config": asdict(self.cfg),
                "field_names": self.field_names,
                "label_names": self.label_names,
                "field_mean": self.field_mean.tolist(),
                "field_std": self.field_std.tolist(),
                "label_mean": self.label_mean.tolist(),
                "label_std": self.label_std.tolist(),
                "label_transform": self.cfg.label_transform,
            }
            write_json(self.normalization_dir / "normalization.json", payload)
            torch.save(
                {
                    "field_mean": self.field_mean,
                    "field_std": self.field_std,
                    "label_mean": self.label_mean,
                    "label_std": self.label_std,
                    "label_names": self.label_names,
                    "field_names": self.field_names,
                    "config": asdict(self.cfg),
                },
                self.normalization_dir / "normalization.pt",
            )

    @property
    def in_chans(self) -> int:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before reading in_chans.")
        return int(self.train_dataset[0]["video"].shape[0])

    def metadata_summary(self) -> dict[str, object]:
        if self.train_base is None or self.label_names is None:
            raise RuntimeError("DataModule.setup() must be called before metadata_summary().")
        return {
            "dataset_name": self.cfg.dataset_name,
            "train_windows": len(self.train_base),
            "valid_windows": len(self.valid_base),
            "test_windows": len(self.test_base),
            "field_names": self.field_names,
            "label_names": self.label_names,
            "spatial_resolution": getattr(self.train_base.metadata, "spatial_resolution", None),
            "constant_scalar_names": list(self.train_base.metadata.constant_scalar_names),
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before val_dataloader().")
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=False,
        )


def make_fraction_subset(dataset: Dataset, fraction: float, seed: int = 42) -> Subset:
    n = len(dataset)
    k = max(1, int(round(n * fraction)))
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=generator)[:k].tolist()
    return Subset(dataset, idx)

