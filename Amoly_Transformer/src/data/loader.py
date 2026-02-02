"""Load raw CSV data and compute run-load gaps for each instrument."""

import pandas as pd
import numpy as np
from typing import Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_and_prepare(
    csv_path: str,
    timestamp_col: str,
    instrument_id_col: str,
    feature_prefix: str,
) -> pd.DataFrame:
    """
    Load CSV, sort by timestamp globally, compute run_gap column.

    The run-load gap for a given row is the number of intervening samples
    (from other instruments) between this row and the previous row of the
    same instrument in global timestamp order.

    Returns:
        DataFrame sorted by timestamp with added 'run_gap' column.
    """
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df["run_gap"] = _compute_run_gaps(df, instrument_id_col)

    logger.info(
        "Loaded %d rows from %s with %d instruments",
        len(df), csv_path, df[instrument_id_col].nunique(),
    )
    return df


def _compute_run_gaps(df: pd.DataFrame, instrument_id_col: str) -> pd.Series:
    """
    For each instrument, count intervening rows from other instruments
    between consecutive runs of the same instrument.

    First occurrence of each instrument gets gap = 0.
    """
    gaps = np.zeros(len(df), dtype=np.int64)
    last_seen: dict[str, int] = {}

    for i, inst_id in enumerate(df[instrument_id_col]):
        if inst_id in last_seen:
            gaps[i] = i - last_seen[inst_id] - 1
        else:
            gaps[i] = 0
        last_seen[inst_id] = i

    return pd.Series(gaps, index=df.index)


def get_feature_columns(df: pd.DataFrame, feature_prefix: str) -> list[str]:
    """Return sorted list of feature column names matching the prefix."""
    return sorted([c for c in df.columns if c.startswith(feature_prefix)])


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    stats: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Z-score normalize feature columns.

    If stats dict is provided (keys: 'mean', 'std'), use those values.
    Otherwise compute from df.

    Returns:
        (DataFrame with normalized features, stats dict for reuse at inference)
    """
    df = df.copy()
    if stats is None:
        mean = df[feature_cols].mean()
        std = df[feature_cols].std().replace(0, 1.0)
        stats = {"mean": mean, "std": std}
    else:
        mean = stats["mean"]
        std = stats["std"]

    df[feature_cols] = (df[feature_cols] - mean) / std
    logger.debug("Normalized %d feature columns", len(feature_cols))
    return df, stats
