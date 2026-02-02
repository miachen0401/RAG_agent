"""PyTorch Dataset producing sliding windows over prepared event sequences."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventWindowDataset(Dataset):
    """
    Sliding-window dataset over a sequence of instrument events.

    Each sample is a window of `window_size` consecutive events.

    Returns:
        features: Tensor of shape (window_size, input_dim)
        gaps: LongTensor of shape (window_size,) -- discretized gap bin indices
    """

    def __init__(
        self,
        features: np.ndarray,
        run_gaps: np.ndarray,
        window_size: int,
        stride: int,
        num_bins: int,
        max_gap: int,
    ):
        self.features = features.astype(np.float32)
        self.gap_bins = self.discretize_gaps(run_gaps, num_bins, max_gap)
        self.window_size = window_size
        self.stride = stride

        n = len(features)
        self.num_windows = max(0, (n - window_size) // stride + 1)
        logger.debug("Created dataset with %d windows (size=%d, stride=%d)", self.num_windows, window_size, stride)

    @staticmethod
    def discretize_gaps(gaps: np.ndarray, num_bins: int, max_gap: int) -> np.ndarray:
        """
        Discretize continuous gap values into bin indices [0, num_bins - 1].

        Uses uniform binning: bin = floor(clamp(gap, 0, max_gap) / max_gap * (num_bins - 1)).
        """
        clamped = np.clip(gaps, 0, max_gap).astype(np.float64)
        bins = np.floor(clamped / max_gap * (num_bins - 1)).astype(np.int64)
        return bins

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.LongTensor]:
        start = idx * self.stride
        end = start + self.window_size
        feat = torch.from_numpy(self.features[start:end])
        gaps = torch.from_numpy(self.gap_bins[start:end])
        return feat, gaps


def build_dataloaders(
    features: np.ndarray,
    run_gaps: np.ndarray,
    data_cfg: dict,
    model_cfg: dict,
    val_split: float,
) -> tuple[DataLoader, DataLoader]:
    """
    Split data into train/val, create EventWindowDataset for each,
    return (train_loader, val_loader).

    Split is done chronologically (not random) to respect temporal ordering.
    """
    n = len(features)
    split_idx = int(n * (1 - val_split))

    gap_cfg = model_cfg["gap_embedding"]

    train_ds = EventWindowDataset(
        features=features[:split_idx],
        run_gaps=run_gaps[:split_idx],
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        num_bins=gap_cfg["num_bins"],
        max_gap=gap_cfg["max_gap"],
    )
    val_ds = EventWindowDataset(
        features=features[split_idx:],
        run_gaps=run_gaps[split_idx:],
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        num_bins=gap_cfg["num_bins"],
        max_gap=gap_cfg["max_gap"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
    )

    logger.info("Built dataloaders: train=%d windows, val=%d windows", len(train_ds), len(val_ds))
    return train_loader, val_loader
