"""CIFAR-10 DataModule and augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .lightning_compat import L

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    workers: int = 4
    randaugment: bool = False
    random_erasing: float = 0.0


class CIFAR10DataModule(L.LightningDataModule):
    """Lightning DataModule for supervised CIFAR-10 training."""

    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set: datasets.CIFAR10 | None = None
        self.test_set: datasets.CIFAR10 | None = None

    def prepare_data(self) -> None:
        # Download once, if needed.
        datasets.CIFAR10(root=self.cfg.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.cfg.data_dir, train=False, download=True)

    def build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        train_ops: list[Any] = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if self.cfg.randaugment:
            if hasattr(transforms, "RandAugment"):
                train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
            else:
                print("Warning: torchvision.transforms.RandAugment is unavailable; ignoring randaugment")

        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        if self.cfg.random_erasing > 0.0:
            train_ops.append(
                transforms.RandomErasing(
                    p=self.cfg.random_erasing,
                    scale=(0.02, 0.20),
                    ratio=(0.3, 3.3),
                )
            )

        train_transform = transforms.Compose(train_ops)
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        return train_transform, test_transform

    def setup(self, stage: str | None = None) -> None:
        train_transform, test_transform = self.build_transforms()
        self.train_set = datasets.CIFAR10(
            root=self.cfg.data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )
        self.test_set = datasets.CIFAR10(
            root=self.cfg.data_dir,
            train=False,
            download=False,
            transform=test_transform,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            raise RuntimeError("CIFAR10DataModule.setup() must be called before train_dataloader().")

        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.cfg.workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.test_set is None:
            raise RuntimeError("CIFAR10DataModule.setup() must be called before val_dataloader().")

        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.cfg.workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
