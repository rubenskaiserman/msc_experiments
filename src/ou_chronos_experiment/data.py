from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    train_n_paths: int = 800
    val_n_paths: int = 200
    n_train_windows: int = 8000
    context_length: int = 100
    horizon: int = 100
    dt: float = 0.1
    train_t_start: float = 0.0
    train_t_end: float = 800.0
    val_t_start: float = 800.0
    val_t_end: float = 1000.0
    sigma: float = 0.10
    ou_theta: float = 0.80
    ou_mu: float = 0.0
    ou_sigma: float = 0.35
    seed: int = 123
    cache_path: str = "data/ou_chronos_paths.npz"
    use_cache: bool = True
    force_regenerate: bool = False


def deterministic_signal(t: np.ndarray) -> np.ndarray:
    return (np.sin(t) * np.cos(0.01 * t)).astype(np.float32)


def simulate_ou_paths(
    n_paths: int,
    t_start: float,
    t_end: float,
    dt: float,
    sigma: float,
    ou_theta: float,
    ou_mu: float,
    ou_sigma: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    t = np.arange(t_start, t_end + 0.5 * dt, dt, dtype=np.float32)
    g = deterministic_signal(t)

    ou = np.empty((n_paths, len(t)), dtype=np.float32)
    ou_deterministic = np.empty_like(ou)
    ou_stochastic = np.empty_like(ou)
    stationary_std = ou_sigma / np.sqrt(2.0 * ou_theta)
    ou[:, 0] = rng.normal(ou_mu, stationary_std, size=n_paths).astype(np.float32)
    ou_deterministic[:, 0] = ou[:, 0]
    ou_stochastic[:, 0] = 0.0

    for k in range(1, len(t)):
        innovation = rng.normal(0.0, 1.0, size=n_paths).astype(np.float32)
        stochastic_increment = ou_sigma * np.sqrt(dt) * innovation
        ou[:, k] = (
            ou[:, k - 1]
            + ou_theta * (ou_mu - ou[:, k - 1]) * dt
            + stochastic_increment
        )
        ou_deterministic[:, k] = (
            ou_deterministic[:, k - 1]
            + ou_theta * (ou_mu - ou_deterministic[:, k - 1]) * dt
        )
        ou_stochastic[:, k] = (
            ou_stochastic[:, k - 1]
            - ou_theta * ou_stochastic[:, k - 1] * dt
            + stochastic_increment
        )

    signal = g[None, :] + ou
    noise = rng.normal(0.0, sigma, size=signal.shape).astype(np.float32)
    series = signal + noise
    return (
        series.astype(np.float32),
        signal.astype(np.float32),
        ou,
        ou_deterministic,
        ou_stochastic,
        noise,
        g,
        t,
    )


def _simulation_config(cfg: DataConfig) -> dict[str, object]:
    keys = (
        "train_n_paths",
        "val_n_paths",
        "dt",
        "train_t_start",
        "train_t_end",
        "val_t_start",
        "val_t_end",
        "sigma",
        "ou_theta",
        "ou_mu",
        "ou_sigma",
        "seed",
        "component_cache_version",
    )
    config = {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}
    config["component_cache_version"] = 2
    return config


def _load_cached_paths(cfg: DataConfig) -> dict[str, np.ndarray] | None:
    path = Path(cfg.cache_path)
    if not cfg.use_cache or cfg.force_regenerate or not path.exists():
        return None

    with np.load(path, allow_pickle=False) as cached:
        cached_cfg = json.loads(cached["simulation_config"].item())
        if cached_cfg != _simulation_config(cfg):
            return None
        return {key: cached[key] for key in cached.files if key != "simulation_config"}


def generate_paths(cfg: DataConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    (
        train_series,
        train_signal,
        train_ou,
        train_ou_deterministic,
        train_ou_stochastic,
        train_noise,
        train_g,
        train_t,
    ) = simulate_ou_paths(
        cfg.train_n_paths,
        cfg.train_t_start,
        cfg.train_t_end,
        cfg.dt,
        cfg.sigma,
        cfg.ou_theta,
        cfg.ou_mu,
        cfg.ou_sigma,
        rng,
    )
    (
        val_series,
        val_signal,
        val_ou,
        val_ou_deterministic,
        val_ou_stochastic,
        val_noise,
        val_g,
        val_t,
    ) = simulate_ou_paths(
        cfg.val_n_paths,
        cfg.val_t_start,
        cfg.val_t_end,
        cfg.dt,
        cfg.sigma,
        cfg.ou_theta,
        cfg.ou_mu,
        cfg.ou_sigma,
        rng,
    )
    return {
        "train_series": train_series,
        "val_series": val_series,
        "train_signal": train_signal,
        "val_signal": val_signal,
        "train_ou": train_ou,
        "val_ou": val_ou,
        "train_ou_deterministic": train_ou_deterministic,
        "val_ou_deterministic": val_ou_deterministic,
        "train_ou_stochastic": train_ou_stochastic,
        "val_ou_stochastic": val_ou_stochastic,
        "train_noise": train_noise,
        "val_noise": val_noise,
        "train_g": train_g,
        "val_g": val_g,
        "train_t": train_t,
        "val_t": val_t,
    }


def save_paths(paths: dict[str, np.ndarray], cfg: DataConfig) -> Path:
    path = Path(cfg.cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        simulation_config=json.dumps(_simulation_config(cfg), sort_keys=True),
        **paths,
    )
    return path


def load_or_generate_paths(cfg: DataConfig) -> dict[str, np.ndarray]:
    cached = _load_cached_paths(cfg)
    if cached is not None:
        print(f"loaded cached paths: {Path(cfg.cache_path)}")
        return cached

    paths = generate_paths(cfg)
    if cfg.use_cache:
        path = save_paths(paths, cfg)
        print(f"saved generated paths: {path}")
    return paths


class RandomWindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        t: np.ndarray,
        n_windows: int,
        context_length: int,
        horizon: int,
        rng: np.random.Generator,
    ):
        self.series = series
        self.t = t
        self.n_windows = n_windows
        self.context_length = context_length
        self.horizon = horizon
        self.window_len = context_length + horizon

        max_start = len(t) - self.window_len
        self.path_idx = rng.integers(0, series.shape[0], size=n_windows)
        self.start_idx = rng.integers(0, max_start + 1, size=n_windows)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.path_idx[idx]
        start = self.start_idx[idx]
        window = self.series[path, start:start + self.window_len]
        x = window[:self.context_length]
        y = window[self.context_length:]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def window_times(self) -> np.ndarray:
        return np.stack([self.t[s:s + self.window_len] for s in self.start_idx], axis=0)


class ExhaustiveWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, t: np.ndarray, context_length: int, horizon: int):
        self.series = series
        self.t = t
        self.context_length = context_length
        self.horizon = horizon
        self.window_len = context_length + horizon
        self.start_idx = np.arange(0, len(t) - self.window_len + 1, dtype=np.int64)
        self.n_per_path = len(self.start_idx)

    def __len__(self) -> int:
        return self.series.shape[0] * self.n_per_path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = idx // self.n_per_path
        start = self.start_idx[idx % self.n_per_path]
        window = self.series[path, start:start + self.window_len]
        x = window[:self.context_length]
        y = window[self.context_length:]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def time_window(self, idx: int) -> np.ndarray:
        start = self.start_idx[idx % self.n_per_path]
        return self.t[start:start + self.window_len]


def build_datasets(cfg: DataConfig) -> dict[str, object]:
    arrays = load_or_generate_paths(cfg)
    rng = np.random.default_rng(cfg.seed + 1)

    train_dataset = RandomWindowDataset(
        arrays["train_series"],
        arrays["train_t"],
        cfg.n_train_windows,
        cfg.context_length,
        cfg.horizon,
        rng,
    )
    val_dataset = ExhaustiveWindowDataset(
        arrays["val_series"],
        arrays["val_t"],
        cfg.context_length,
        cfg.horizon,
    )

    return arrays | {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and cache OU experiment paths.")
    parser.add_argument("--cache-path", type=str, default=DataConfig.cache_path)
    parser.add_argument("--force", action="store_true", help="Regenerate even if a matching cache exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DataConfig(cache_path=args.cache_path, force_regenerate=args.force)
    paths = load_or_generate_paths(cfg)
    print("cache path:", Path(cfg.cache_path).resolve())
    print("train series:", paths["train_series"].shape)
    print("validation series:", paths["val_series"].shape)
    print("config:", asdict(cfg))


if __name__ == "__main__":
    main()
