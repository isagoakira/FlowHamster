"""
Data Pipeline Module

Provides efficient data loading and preprocessing utilities.
"""

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision.transforms as T
    import torchvision.io
    TORCHVISION_AVAILABLE = True
except ImportError:
    T = None
    TORCHVISION_AVAILABLE = False
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import pandas as pd
import json


class ImageFolderDataset(Dataset):
    """Load images from a folder with glob pattern."""

    def __init__(
        self,
        path: str,
        pattern: str = "*.jpg",
        transform: Optional[Callable] = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
    ):
        self.path = Path(path)
        self.pattern = pattern
        self.transform = transform
        self.extensions = extensions

        # Collect all image paths
        self.paths = []
        for ext in self.extensions:
            self.paths.extend(sorted(self.path.glob(f"*{ext}")))
            self.paths.extend(sorted(self.path.glob(f"*{ext.upper()}")))

        if len(self.paths) == 0:
            raise ValueError(f"No images found in {path} with pattern {pattern}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.paths[idx]
        if TORCHVISION_AVAILABLE:
            image = torchvision.io.read_image(str(img_path)).float() / 255.0
        else:
            raise ImportError("torchvision is required for ImageFolderDataset. Please install it with: pip install torchvision")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "path": str(img_path),
            "filename": img_path.name,
        }


class CSVDataset(Dataset):
    """Load data from CSV file."""

    def __init__(
        self,
        path: str,
        delimiter: str = ",",
        transform: Optional[Callable] = None,
    ):
        self.df = pd.read_csv(path, delimiter=delimiter)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        result = row.to_dict()

        if self.transform:
            result = self.transform(result)

        return result


class JSONLDataset(Dataset):
    """Load data from JSONL (JSON Lines) file."""

    def __init__(
        self,
        path: str,
        transform: Optional[Callable] = None,
    ):
        self.path = Path(path)
        self.transform = transform

        # Load all records
        with open(self.path, 'r') as f:
            self.records = [json.loads(line) for line in f]

        if len(self.records) == 0:
            raise ValueError(f"No records found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        if self.transform:
            record = self.transform(record)

        return record


class ParquetDataset(Dataset):
    """Load data from Parquet file."""

    def __init__(
        self,
        path: str,
        transform: Optional[Callable] = None,
    ):
        self.df = pd.read_parquet(path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        result = row.to_dict()

        if self.transform:
            result = self.transform(result)

        return result


class ComposeTransforms:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.flip(image, dims=[2])
        return image


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.flip(image, dims=[1])
        return image


class Resize:
    """Resize image to target size."""

    def __init__(self, size: tuple = (224, 224)):
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # image: (C, H, W)
        return torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)


class Normalize:
    """Normalize tensor with mean and std."""

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


class ToTensor:
    """Convert PIL Image or numpy array to tensor."""

    def __call__(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.tensor(x).float()


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader from a Dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
