"""Generate synthetic instrument run data for testing."""

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_NUM_ROWS = 50
DEFAULT_NUM_FEATURES = 160
DEFAULT_NUM_INSTRUMENTS = 3
DEFAULT_ANOMALY_FRACTION = 0.08


def generate_fake_data(
    num_rows: int = DEFAULT_NUM_ROWS,
    num_features: int = DEFAULT_NUM_FEATURES,
    num_instruments: int = DEFAULT_NUM_INSTRUMENTS,
    anomaly_fraction: float = DEFAULT_ANOMALY_FRACTION,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a DataFrame with irregular timestamps, instrument IDs, and 160 features.

    Normal rows have features ~ N(0, 1). Anomalous rows have a subset of features
    shifted by +/- 4 sigma with 3x variance.
    """
    rng = np.random.default_rng(seed)

    # Irregular timestamps: random intervals of 1-48 hours
    base = pd.Timestamp("2024-01-01")
    hour_gaps = rng.uniform(1, 48, size=num_rows)
    cumulative_hours = np.cumsum(hour_gaps)
    timestamps = [base + pd.Timedelta(hours=float(h)) for h in cumulative_hours]

    # Instrument IDs assigned randomly
    instruments = [f"INST_{i + 1:03d}" for i in range(num_instruments)]
    instrument_ids = rng.choice(instruments, size=num_rows).tolist()

    # Normal features
    features = rng.standard_normal((num_rows, num_features))

    # Inject anomalies
    num_anomalies = max(1, int(num_rows * anomaly_fraction))
    anomaly_indices = rng.choice(num_rows, size=num_anomalies, replace=False)

    for idx in anomaly_indices:
        num_affected = rng.integers(20, 41)
        affected_cols = rng.choice(num_features, size=num_affected, replace=False)
        shift_direction = rng.choice([-1, 1], size=num_affected)
        features[idx, affected_cols] += shift_direction * 4.0
        features[idx, affected_cols] *= 3.0

    # Build DataFrame
    feature_cols = [f"feat_{i:03d}" for i in range(num_features)]
    df = pd.DataFrame(features, columns=feature_cols)
    df.insert(0, "instrument_id", instrument_ids)
    df.insert(0, "timestamp", timestamps)

    logger.info(
        "Generated %d fake rows (%d anomalous) with %d features",
        num_rows, num_anomalies, num_features,
    )
    return df


def save_fake_data(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to CSV."""
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved fake data to %s", output_path)
