"""Training loop with validation, early stopping, and checkpointing."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Manages the full training lifecycle for AnomalyTransformer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_cfg: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = training_cfg

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg["learning_rate"],
            weight_decay=training_cfg["weight_decay"],
        )
        self.scheduler = self._build_scheduler()
        self.criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Build LR scheduler from config."""
        name = self.cfg["lr_scheduler"]
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg["cosine_T_max"]
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg["step_size"],
                gamma=self.cfg["step_gamma"],
            )
        return None

    def train(self) -> None:
        """
        Full training loop with early stopping and checkpointing.

        Each epoch: train -> validate -> checkpoint if improved -> early stop check.
        """
        logger.info("Starting training for up to %d epochs", self.cfg["epochs"])

        for epoch in range(1, self.cfg["epochs"] + 1):
            train_loss = self._train_one_epoch()
            val_loss = self._validate()

            logger.info(
                "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch, self.cfg["epochs"], train_loss, val_loss,
            )

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.cfg["early_stopping_patience"]:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Training complete. Best val_loss=%.6f", self.best_val_loss)

    def _train_one_epoch(self) -> float:
        """Single training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, gap_bins in self.train_loader:
            features = features.to(self.device)
            gap_bins = gap_bins.to(self.device)

            self.optimizer.zero_grad()
            reconstructed = self.model(features, gap_bins)
            loss = self.criterion(reconstructed, features)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["gradient_clip_norm"]
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Single validation pass. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for features, gap_bins in self.val_loader:
            features = features.to(self.device)
            gap_bins = gap_bins.to(self.device)

            reconstructed = self.model(features, gap_bins)
            loss = self.criterion(reconstructed, features)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        ckpt_dir = Path(self.cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "best_model.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            path,
        )
        logger.info("Saved checkpoint to %s (epoch %d, val_loss=%.6f)", path, epoch, val_loss)
