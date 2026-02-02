# Amoly Transformer

Unsupervised, event-based anomaly detection system for high-dimensional experimental instrument data using Transformer reconstruction.

## Overview

Each sample corresponds to a single instrument run with a fixed 160-dimensional feature vector. Temporal dependency is modeled via the **run-load gap** — the number of intervening samples between consecutive runs of the same instrument — rather than calendar time. A multi-layer Transformer encoder processes sliding windows of events, trained with an MSE reconstruction objective on historical data assumed to be predominantly normal. At inference, per-sample anomaly scores are derived from reconstruction errors.

## Project Structure

```
├── main.py                 # Entry point (train / infer / generate-data)
├── pyproject.toml          # uv package config
├── data/                   # CSV data files
├── src/
│   ├── config/             # YAML configs + loader
│   ├── data/               # Data loading, preprocessing, fake data generation
│   ├── models/             # AnomalyTransformer model
│   ├── training/           # Training loop with early stopping
│   ├── inference/          # Scoring and threshold calibration
│   └── utils/              # Logger factory
├── outputs/                # Checkpoints and inference results
└── docs/                   # Documentation
```

## Usage

```bash
# Install dependencies
uv sync

# Generate fake test data
uv run main.py generate-data

# Train model
uv run main.py train

# Run inference
uv run main.py infer
```

## Configuration

All adjustable parameters are in YAML config files under `src/config/`:

- `model.yaml` — Model architecture (d_model, nhead, num_layers, gap embedding)
- `training.yaml` — Training hyperparameters (lr, epochs, batch_size, early stopping)
- `data.yaml` — Data pipeline (csv path, window size, normalization)
- `inference.yaml` — Inference settings (checkpoint path, threshold method)
