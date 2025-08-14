# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning forex trading strategy project that implements deep learning models for predicting price movements in the USD/JPY currency pair. The project follows a complete ML pipeline from data preprocessing to backtesting trading strategies.

## Development Commands

### Environment Setup
```bash
# Create virtual environment (using uv package manager)
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r pyproject.toml
```

### Main Training Scripts
- `python train_side.py` - Train direction classification models (side prediction)
- `python train_ret.py` - Train return regression models  
- `python main.py` - Simple hello world entry point
- `python preprocessing.py` - Data preprocessing pipeline with configurable parameters
- `python predict.py` - Model inference/prediction
- `python backtest.py` - Backtesting framework
- `python mt5_backtest.py` - MetaTrader 5 specific backtesting

### Development Workflow
```bash
# Launch Jupyter for exploration
jupyter notebook

# Data preprocessing with custom parameters
python preprocessing.py --input <path> --output <path> --threshold 3e-5 --time_gap_tolerance 60

# View TensorBoard logs
tensorboard --logdir lightning_logs
```

## Architecture Overview

### Data Pipeline
- **Raw Data**: Located in `data/raw/` (tick and bar data from Dukascopy and Oanda)
- **Resampling**: Convert tick data to time/event-based bars using `libs/resampling.py`
- **Feature Engineering**: Technical indicators, log returns, cyclic time features in `preprocess/`
- **Normalization**: MinMax and Standard scalers stored in `data/scalers/`
- **Labeling**: Event-based directional labels using CUSUM filters in `data/direction_labels/`

### Model Architecture
The project implements several deep learning models in `models/`:

#### Classification Models (`models/classification/`)
- **SimpleTransformerModel**: Transformer with positional encoding for sequence classification
- **T2VTransformerModel**: Time2Vec + Transformer for temporal patterns
- **GRUModule**: GRU-based recurrent network
- **LSTMModule**: LSTM-based recurrent network

#### Model Training Framework
- Built on PyTorch Lightning for scalable training
- Automatic device detection (CUDA/MPS/CPU)  
- TensorBoard logging and model checkpointing
- Early stopping and learning rate scheduling

### Data Loading
- **EventBasedDataModule**: Lightning data module for event-driven sampling
- **DirectionDataset**: Custom dataset for sequence-based directional prediction
- **ConfidenceDataset**: Dataset with confidence/meta-labeling support

### Trading Strategy Framework
Located in `strategies/`:
- **DirectionModelStrategy**: Uses ML model predictions for trade signals
- **SimpleStrategy**: Basic rule-based trading
- Integration with `backtesting` library for performance evaluation

### Key Configuration
- `config/config.yaml`: Central configuration for data paths, model hyperparameters, and training settings
- Configurable sequence lengths, feature sets, and target horizons
- Support for multiple timeframes (1m, 5m, 58m-dollar bars, etc.)

## Data Structure

The project uses a hierarchical data organization:
- `data/raw/` - Original market data files
- `data/resampled/` - Time/event-based resampled data  
- `data/processed/` - Feature-engineered data
- `data/normalized/` - Scaled data ready for training
- `data/direction_labels/` - Event-based directional labels
- `data/meta_labels/` - Meta-labeling for confidence estimation
- `data/predictions/` - Model output predictions

## Important Implementation Details

### Event-Based Sampling
The system uses event-based sampling (e.g., dollar bars, volume bars) rather than just time-based bars for more informative market microstructure analysis.

### Feature Engineering
- Cyclic encoding for temporal features (hour_cos, dow_cos, etc.)
- Technical indicators via `ta` library
- Log returns and volatility-adjusted features
- Multi-timeframe feature aggregation

### Model Training Pipeline
1. Load normalized data and event labels
2. Create train/validation splits maintaining temporal order
3. Use class weighting for imbalanced datasets
4. Lightning trainer with callbacks for checkpointing and early stopping
5. TensorBoard logging for monitoring

### Trading Strategy Implementation
Strategies integrate with the `backtesting` library and can use:
- Model predictions as signals
- Dynamic stop-loss and take-profit based on volatility (ATR)
- Position sizing and risk management

### Notebooks for Exploration
The `notebook/` directory contains Jupyter notebooks for:
- Data exploration and visualization
- Feature engineering experiments  
- Model evaluation and analysis
- Trading strategy development and testing