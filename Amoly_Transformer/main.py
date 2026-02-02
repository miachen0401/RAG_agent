"""
Entry point for Amoly Transformer anomaly detection system.

Usage:
    uv run main.py generate-data
    uv run main.py train
    uv run main.py infer
"""

import argparse

import numpy as np
import torch

from src.config.loader import load_config
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amoly Transformer Anomaly Detection")
    parser.add_argument("mode", choices=["train", "infer", "generate-data"])
    return parser.parse_args()


def run_generate_data(cfg: dict) -> None:
    """Generate fake instrument run data and save to CSV."""
    from src.data.generate_fake import generate_fake_data, save_fake_data

    data_cfg = cfg["data"]
    df = generate_fake_data()
    save_fake_data(df, data_cfg["csv_path"])


def run_train(cfg: dict) -> None:
    """Load data, build model, and train."""
    from src.data.loader import load_and_prepare, get_feature_columns, normalize_features
    from src.data.dataset import build_dataloaders
    from src.models.transformer import build_model
    from src.training.trainer import Trainer

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    # Seed
    torch.manual_seed(training_cfg["seed"])
    np.random.seed(training_cfg["seed"])

    # Load and prepare data
    df = load_and_prepare(
        csv_path=data_cfg["csv_path"],
        timestamp_col=data_cfg["timestamp_col"],
        instrument_id_col=data_cfg["instrument_id_col"],
        feature_prefix=data_cfg["feature_prefix"],
    )

    feature_cols = get_feature_columns(df, data_cfg["feature_prefix"])
    if data_cfg["normalize"]:
        df, stats = normalize_features(df, feature_cols)
        logger.info("Feature normalization applied")

    features = df[feature_cols].values
    run_gaps = df["run_gap"].values

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        features=features,
        run_gaps=run_gaps,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        val_split=training_cfg["val_split"],
    )

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg)
    logger.info("Model built with %d parameters", sum(p.numel() for p in model.parameters()))

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        device=device,
    )
    trainer.train()


def run_infer(cfg: dict) -> None:
    """Load trained model, score data, apply threshold, save results."""
    from src.data.loader import load_and_prepare, get_feature_columns, normalize_features
    from src.data.dataset import EventWindowDataset
    from src.models.transformer import build_model
    from src.inference.scorer import AnomalyScorer

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]

    # Use inference csv_path if specified, otherwise fall back to data csv_path
    csv_path = infer_cfg.get("csv_path", data_cfg["csv_path"])

    df = load_and_prepare(
        csv_path=csv_path,
        timestamp_col=data_cfg["timestamp_col"],
        instrument_id_col=data_cfg["instrument_id_col"],
        feature_prefix=data_cfg["feature_prefix"],
    )

    feature_cols = get_feature_columns(df, data_cfg["feature_prefix"])
    if data_cfg["normalize"]:
        df, stats = normalize_features(df, feature_cols)

    features = df[feature_cols].values.astype(np.float32)
    run_gaps = df["run_gap"].values

    gap_cfg = model_cfg["gap_embedding"]

    # Build full dataset (no train/val split)
    dataset = EventWindowDataset(
        features=features,
        run_gaps=run_gaps,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        num_bins=gap_cfg["num_bins"],
        max_gap=gap_cfg["max_gap"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=0,
    )

    # Build model and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg)

    scorer = AnomalyScorer(
        model=model,
        checkpoint_path=infer_cfg["checkpoint_path"],
        device=device,
    )

    scores = scorer.score_dataset(
        dataloader=dataloader,
        total_samples=len(features),
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
    )

    flags, threshold = AnomalyScorer.apply_threshold(
        scores=scores,
        method=infer_cfg["threshold_method"],
        percentile=infer_cfg.get("percentile"),
        manual_threshold=infer_cfg.get("manual_threshold"),
    )

    AnomalyScorer.save_results(
        df=df,
        scores=scores,
        flags=flags,
        output_path=infer_cfg["output_path"],
        flag_column=infer_cfg["flag_column"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    setup_logging(level="INFO")

    logger.info("Starting Amoly Transformer in '%s' mode", args.mode)

    if args.mode == "generate-data":
        run_generate_data(cfg)
    elif args.mode == "train":
        run_train(cfg)
    elif args.mode == "infer":
        run_infer(cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
