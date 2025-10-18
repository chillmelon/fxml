# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning forex trading strategy project implementing deep learning models (Transformers, LSTMs) for predicting price movements in the USD/JPY currency pair. The project follows a complete ML pipeline from data preprocessing to backtesting.

## Instructions
- DO NOT RUN ANY COMMAND
## Environment Setup

```bash
# Create virtual environment using uv
uv venv .venv
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
uv sync

# Or using Make
make create_environment
make requirements
```

## Development Commands

### Training Models

Each model type has its own training script in `fxml/models/<model_type>/train.py`:

```bash
# Train baseline classifier (simple feedforward baseline)
python -m fxml.models.baseline_classifier.train

# Train transformer classifier (direction prediction)
python -m fxml.models.transformer_classifier.train

# Train transformer regressor (return prediction)
python -m fxml.models.transformer_regressor.train

# Train LSTM classifier
python -m fxml.models.lstm_classifier.train

# Train LSTM regressor
python -m fxml.models.lstm_regressor.train
```

### Model Inference & Backtesting

```bash
# Generate predictions from trained model
python predict.py

# Run backtest on trading strategy
python backtest.py

# MetaTrader 5 specific backtesting
python mt5_backtest.py
```

### Monitoring & Exploration

```bash
# View training metrics in TensorBoard
tensorboard --logdir lightning_logs

# Launch Jupyter for exploration
jupyter notebook
```

### Utilities

```bash
# Clean compiled Python files
make clean
```

## Architecture Overview

### Package Structure

The project is organized as a Python package under `fxml/`:

- `fxml/data/` - Data loading, preprocessing, and datasets
  - `datasets/` - PyTorch Dataset implementations
  - `datamodules/` - PyTorch Lightning DataModules
  - `preprocessing/` - Feature engineering and event filters
- `fxml/models/` - Model architectures organized by type
  - `baseline_classifier/` - Simple feedforward baseline for direction prediction
  - `transformer_classifier/` - Transformer for direction prediction
  - `transformer_regressor/` - Transformer for return prediction
  - `lstm_classifier/` - LSTM for direction prediction
  - `lstm_regressor/` - LSTM for return prediction
- `fxml/trading/` - Trading strategies and backtesting
  - `strategies/` - Strategy implementations
  - `mqpy/` - MetaTrader integration utilities
- `fxml/visualization/` - Plotting and evaluation tools

### Data Pipeline Flow

1. **Raw Data** (`data/raw/`) - Original market data (tick/bar data from Dukascopy/Oanda)
2. **Intermediate** (`data/interm/`) - Resampled data (time/event-based bars)
3. **Processed** (`data/processed/`) - Feature-engineered data with technical indicators
4. **Direction Labels** (`data/direction_labels/`) - Event-based directional labels using CUSUM/Z-score filters
5. **Predictions** (`data/predictions/`) - Model inference outputs

### Event-Based Labeling System

The project uses event-driven labeling rather than fixed-time intervals:

- **CUSUM Filter** (`fxml/data/preprocessing/filters.py:cusum_filter`) - Detects events when cumulative price movements exceed a threshold
- **Z-Score Filter** (`fxml/data/preprocessing/filters.py:z_score_filter`) - Triggers events when price deviates significantly from moving average

This produces sparse labels at significant market events, which are then used for training.

### Model Training Architecture

All models use PyTorch Lightning for standardized training:

1. **Configuration** - YAML files in `configs/` define:
   - Data paths and feature columns
   - Model hyperparameters (architecture, layers, dropout)
   - Training parameters (batch size, learning rate, validation split)

2. **DataModule Pattern** - `EventBasedDataModule` handles:
   - Loading processed data and event labels
   - Creating sequence datasets with lookback windows
   - Train/validation temporal splits (no data leakage)
   - DataLoader configuration with batching

3. **Dataset Pattern** - `DirectionDataset` and `ReturnDataset`:
   - Takes event timestamps and creates sequences
   - Extracts `sequence_length` bars before each event
   - Returns (features, target, index) tuples

4. **Training Loop** - Handled by Lightning Trainer with:
   - Automatic device selection (CUDA/MPS/CPU)
   - TensorBoard logging
   - Model checkpointing (best and last)
   - Early stopping on validation loss
   - Learning rate scheduling

### Model Architectures

**Baseline Classifier** (`fxml/models/baseline_classifier/model.py`):
- Simple non-sequential baseline model
- Mean pooling across time dimension (ignores temporal order)
- Two-layer feedforward network with ReLU activation
- Provides performance baseline to compare against sequential models
- Useful for determining if temporal structure adds value

**Transformer Models** (`fxml/models/transformer_*/model.py`):
- Input projection layer (features → d_model)
- Positional encoding for temporal awareness
- Multi-head self-attention encoder layers
- Pooling strategy: "mean" (average all timesteps) or "last" (final timestep)
- Output layer for classification or regression

**LSTM Models** (`fxml/models/lstm_*/model.py`):
- Standard LSTM architecture
- Configurable hidden size and number of layers
- Dropout for regularization
- Final hidden state used for prediction

### Feature Engineering

Located in `fxml/data/preprocessing/features.py`:

- **Return Features** - Delta, percent return, log return with rolling means
- **Technical Indicators** - Using `pandas_ta`:
  - Volatility: RV (Realized Volatility), ATR, Bollinger Bands
  - Momentum: RSI, MACD, Stochastic
  - Trend: EMA, ADX, Donchian Channels
- **Time Features** - Cyclical encoding (sin/cos pairs) for hour, day of week, day of month, month

Configuration driven by `configs/features.yaml`.

### Trading Strategy Framework

Strategies inherit from `backtesting.Strategy`:

- **DirectionModelStrategy** - Uses classifier predictions for long/short signals
- **DuoModelStrategy** - Combines direction and confidence models
- **DirectionConfidenceStrategy** - Trades only when confidence is high
- **LabelTestStrategy** - Validates event labels by trading directly on them

Backtesting uses the `backtesting` library with OHLCV data joined with model predictions.

## Configuration System

All training configs in `configs/`:
- `baseline_classifier.yaml` - Baseline feedforward direction model
- `transformer_classifier.yaml` - Transformer direction model
- `transformer_regressor.yaml` - Transformer return model
- `lstm_classifier.yaml` - LSTM direction model
- `lstm_regressor.yaml` - LSTM return model
- `features.yaml` - Feature engineering settings

Config structure:
```yaml
data:
  dataset_path: path/to/processed/data.pkl
  label_path: path/to/labels.pkl
  features: [list of feature column names]
  target: target_column_name
  sequence_length: 24  # lookback window

model:
  name: "model_name"
  # architecture hyperparameters

training:
  batch_size: 1024
  val_split: 0.2
  num_workers: 0  # CPU workers for DataLoader
  shuffle: True
```

## Important Implementation Details

### Temporal Ordering

The system maintains strict temporal order to prevent data leakage:
- Labels are sorted by timestamp before train/val split
- Validation set is always the most recent data (last `val_split` proportion)
- Sequences are created by looking back in time from event timestamps
- No shuffling across train/val boundary

### Model Checkpoints

Saved in `lightning_logs/<model_name>/version_X/checkpoints/`:
- `best_checkpoint.ckpt` - Best validation loss
- `last.ckpt` - Most recent epoch

Load with: `Model.load_from_checkpoint("path/to/checkpoint.ckpt")`

### Prediction Workflow

1. Load checkpoint with trained model
2. Load processed data and create dataset
3. Run inference to generate predictions
4. Save predictions to `data/predictions/` as pickle
5. Join predictions with OHLCV data for backtesting
