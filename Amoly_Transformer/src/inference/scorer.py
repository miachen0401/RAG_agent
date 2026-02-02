"""Inference: compute per-sample anomaly scores and apply thresholds."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyScorer:
    """Loads a trained model, runs inference, produces per-sample anomaly scores."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        logger.info(
            "Loaded checkpoint from %s (epoch %d, val_loss=%.6f)",
            checkpoint_path, checkpoint["epoch"], checkpoint["val_loss"],
        )

    @torch.no_grad()
    def score_dataset(
        self,
        dataloader: DataLoader,
        total_samples: int,
        window_size: int,
        stride: int,
    ) -> np.ndarray:
        """
        Run reconstruction on every window. Aggregate per-sample scores
        by averaging reconstruction errors across all overlapping windows.

        Args:
            dataloader: DataLoader over EventWindowDataset (no shuffle).
            total_samples: Total number of original samples (N).
            window_size: Window size used in the dataset.
            stride: Stride used in the dataset.

        Returns:
            scores: np.ndarray of shape (total_samples,) with per-sample anomaly scores.
        """
        score_sums = np.zeros(total_samples, dtype=np.float64)
        score_counts = np.zeros(total_samples, dtype=np.float64)

        for batch_idx, (features, gap_bins) in enumerate(dataloader):
            features = features.to(self.device)
            gap_bins = gap_bins.to(self.device)

            reconstructed = self.model(features, gap_bins)
            # Per-sample MSE: (batch, seq_len)
            errors = ((reconstructed - features) ** 2).mean(dim=-1).cpu().numpy()

            batch_size = features.size(0)
            for i in range(batch_size):
                global_window_idx = batch_idx * dataloader.batch_size + i
                start = global_window_idx * stride
                for j in range(window_size):
                    sample_idx = start + j
                    if sample_idx < total_samples:
                        score_sums[sample_idx] += errors[i, j]
                        score_counts[sample_idx] += 1

        # Avoid division by zero for samples not covered by any window
        score_counts = np.maximum(score_counts, 1)
        scores = score_sums / score_counts

        logger.info(
            "Scored %d samples: mean=%.6f, std=%.6f, max=%.6f",
            total_samples, scores.mean(), scores.std(), scores.max(),
        )
        return scores

    @staticmethod
    def apply_threshold(
        scores: np.ndarray,
        method: str,
        percentile: float | None = None,
        manual_threshold: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Apply threshold to anomaly scores.

        Args:
            method: "percentile" or "manual"
            percentile: Percentile cutoff (used if method == "percentile")
            manual_threshold: Fixed threshold (used if method == "manual")

        Returns:
            (binary_flags, threshold_value)
        """
        if method == "percentile":
            threshold = float(np.percentile(scores, percentile))
        elif method == "manual":
            threshold = float(manual_threshold)
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        flags = (scores >= threshold).astype(np.int32)
        num_flagged = flags.sum()
        logger.info(
            "Threshold=%.6f (%s), flagged %d / %d samples",
            threshold, method, num_flagged, len(scores),
        )
        return flags, threshold

    @staticmethod
    def save_results(
        df: pd.DataFrame,
        scores: np.ndarray,
        flags: np.ndarray,
        output_path: str,
        flag_column: str,
    ) -> None:
        """Append anomaly_score and flag column to DataFrame, save to CSV."""
        df = df.copy()
        df["anomaly_score"] = scores
        df[flag_column] = flags

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved anomaly results to %s", output_path)
