#!/usr/bin/env python3
"""
Train volatility forecasting models (GARCH, realized vol).
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v1.data.loaders import load_ohlcv, load_config, merge_data_sources
from v1.features.pipeline import build_features
from v1.models.volatility.garch import GARCHForecaster, garch_forecast_pipeline
from v1.models.volatility.realized_vol import (
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train volatility models")
    parser.add_argument("--data", type=str, required=True, help="Path to OHLCV data")
    parser.add_argument("--config", type=str, default="configs/volatility_model.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="models/volatility", help="Output directory")
    parser.add_argument("--model", type=str, default="garch", choices=["garch", "realized"], help="Model type")
    parser.add_argument("--start", type=str, help="Start date")
    parser.add_argument("--end", type=str, help="End date")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    ohlcv = load_ohlcv(args.data, start=args.start, end=args.end)
    logger.info(f"Loaded {len(ohlcv)} rows from {ohlcv.index.min()} to {ohlcv.index.max()}")
    
    # Build features
    logger.info("Building features...")
    features_df = build_features(ohlcv, config, is_training=False)
    
    if args.model == "garch":
        train_garch(features_df, config, args.output)
    elif args.model == "realized":
        train_realized_vol(features_df, config, args.output)

def train_garch(df: pd.DataFrame, config: Dict, output_dir: str):
    """Train GARCH model."""
    logger = logging.getLogger(__name__)
    
    garch_config = config.get("garch", {})
    
    # Prepare returns
    returns = df["close"].pct_change().dropna()
    returns = returns * 100  # Scale for numerical stability
    
    # Split data
    train_frac = config.get("training", {}).get("train_frac", 0.7)
    val_frac = config.get("training", {}).get("val_frac", 0.15)
    
    n = len(returns)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_returns = returns.iloc[:train_end]
    val_returns = returns.iloc[train_end:val_end]
    test_returns = returns.iloc[val_end:]
    
    # Train GARCH
    logger.info("Training GARCH model...")
    forecaster = GARCHForecaster(
        p=garch_config.get("p", 1),
        q=garch_config.get("q", 1),
        dist=garch_config.get("dist", "normal"),
        mean_model=garch_config.get("mean_model", "constant"),
        forecast_horizon=garch_config.get("forecast_horizon", 24),
        min_obs=garch_config.get("min_obs", 500),
        refit_frequency=garch_config.get("refit_frequency", 24)
    )
    
    forecaster.fit(train_returns)
    
    # Evaluate on validation
    logger.info("Evaluating on validation set...")
    val_forecasts = forecaster.forecast_rolling(
        df["close"].iloc[:val_end],
        horizon=garch_config.get("forecast_horizon", 24),
        step=garch_config.get("refit_frequency", 24)
    )
    
    # Calculate realized vol for comparison
    val_realized = realized_volatility(
        df["close"].iloc[train_end:val_end],
        window=24,
        annualization_factor=8760
    )
    
    # Align and compute metrics
    common_idx = val_forecasts.index.intersection(val_realized.index)
    if len(common_idx) > 0:
        forecast_1h = val_forecasts.loc[common_idx, "garch_forecast_1h"]
        realized_1h = val_realized.loc[common_idx]
        
        mse = np.mean((forecast_1h - realized_1h) ** 2)
        mae = np.mean(np.abs(forecast_1h - realized_1h))
        corr = forecast_1h.corr(realized_1h)
        
        logger.info(f"Validation MSE: {mse:.6f}, MAE: {mae:.6f}, Correlation: {corr:.4f}")
    
    # Save model
    output_path = Path(output_dir) / "garch_model"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save forecaster state
    import joblib
    joblib.dump(forecaster, f"{output_path}.pkl")
    
    # Save config
    with open(f"{output_path}_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    logger.info(f"Model saved to {output_path}")

def train_realized_vol(df: pd.DataFrame, config: Dict, output_dir: str):
    """Train/validate realized volatility models."""
    logger = logging.getLogger(__name__)
    
    rv_config = config.get("realized_vol", {})
    windows = rv_config.get("windows", [24, 168, 720])
    method = rv_config.get("method", "parkinson")
    
    # Calculate realized volatility for all windows
    results = {}
    for window in windows:
        if method == "parkinson":
            rv = parkinson_volatility(df["high"], df["low"], window=window)
        elif method == "garman_klass":
            rv = garman_klass_volatility(df["open"], df["high"], df["low"], df["close"], window=window)
        elif method == "yang_zhang":
            rv = yang_zhang_volatility(df["open"], df["high"], df["low"], df["close"], window=window)
        else:
            rv = realized_volatility(df["close"], window=window)
        
        results[f"rv_{window}"] = rv
    
    rv_df = pd.DataFrame(results, index=df.index)
    
    # Save
    output_path = Path(output_dir) / "realized_volatility.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rv_df.to_parquet(output_path)
    
    logger.info(f"Realized volatility saved to {output_path}")

if __name__ == "__main__":
    main()
