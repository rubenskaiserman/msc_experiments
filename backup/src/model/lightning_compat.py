"""Compatibility imports for Lightning."""

from __future__ import annotations

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
    from lightning.pytorch.loggers import TensorBoardLogger
except ModuleNotFoundError:  # fallback for older installations
    import pytorch_lightning as L  # type: ignore
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar  # type: ignore
    from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore

__all__ = [
    "L",
    "LearningRateMonitor",
    "ModelCheckpoint",
    "TQDMProgressBar",
    "TensorBoardLogger",
]
