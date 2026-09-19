"""Single entry point that chains all feature builders into one dataframe."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from src.v1.features.lag_returns import add_lagged_returns
from src.v1.features.ema_trend import add_ema_trend_features
from src.v1.features.interactions import add_interaction_features
from src.v1.features.volatility_features import add_volatility_features

def build_features(
    df: pd.DataFrame,
    config: Dict,
    target_horizon: int = 4,
    threshold_pct: float = 0.015,
    labeling_method: str = "terciles",
    is_training: bool = True
) -> pd.DataFrame:
    """
    Build complete feature set for modeling.
    
    Args:
        df: DataFrame with OHLCV data (and optionally OI, options)
        config: Feature configuration dictionary
        target_horizon: Hours ahead for target
        threshold_pct: Threshold for direction labeling
        labeling_method: Method for creating target labels
        is_training: Whether building for training (creates target) or inference
        
    Returns:
        DataFrame with features and optionally target
    """
    df = df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Sort by time
    df = df.sort_index()
    
    # 1. Basic lagged returns
    if config.get("features", {}).get("return_lags"):
        df = add_lagged_returns(
            df,
            lags=config["features"]["return_lags"],
            price_col="close"
        )
    
    # 2. Price lags
    if config.get("features", {}).get("price_lags"):
        df = add_price_lags(
            df,
            lags=config["features"]["price_lags"],
            price_col="close"
        )
    
    # 3. Volume lags
    if config.get("features", {}).get("volume_lags"):
        df = add_volume_lags(
            df,
            lags=config["features"]["volume_lags"],
            volume_col="volume"
        )
    
    # 4. Volatility features (realized, ATR, etc.)
    if config.get("features", {}):
        vol_config = config["features"]
        df = add_volatility_features(
            df,
            atr_window=vol_config.get("atr_window", 24),
            rv_windows=vol_config.get("rv_windows", [24, 168, 720]),
            volume_windows=vol_config.get("volume_windows", [24, 168]),
            include_garch=vol_config.get("include_garch_features", False),
            include_implied=vol_config.get("include_implied_features", False)
        )
    
    # 5. EMA trend features
    if config.get("features", {}).get("ema_windows"):
        df = add_ema_trend_features(
            df,
            windows=config["features"]["ema_windows"],
            price_col="close"
        )
    
    # 6. Interaction features
    if config.get("features", {}).get("interaction_terms", False):
        df = add_interaction_features(df)
    
    # 7. Volatility regime features
    if config.get("features", {}).get("volatility_regime", False):
        df = add_volatility_regime_features(df)
    
    # 8. Time-based features
    df = add_time_features(df)
    
    # 9. Create target (only for training)
    if is_training:
        df = create_direction_target(
            df,
            horizon=target_horizon,
            threshold_pct=threshold_pct,
            method=labeling_method
        )
    
    # Drop rows with NaN (from lagging/rolling)
    df = df.dropna()
    
    return df

def add_price_lags(df: pd.DataFrame, lags: List[int], price_col: str = "close") -> pd.DataFrame:
    """Add lagged price features."""
    for lag in lags:
        df[f"price_lag_{lag}"] = df[price_col].shift(lag)
    return df

def add_volume_lags(df: pd.DataFrame, lags: List[int], volume_col: str = "volume") -> pd.DataFrame:
    """Add lagged volume features."""
    for lag in lags:
        df[f"volume_lag_{lag}"] = df[volume_col].shift(lag)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    df = df.copy()
    # Hour of day (0-23)
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Day of week (0-6)
    df["dayofweek"] = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    # Month (1-12)
    df["month"] = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df

def add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility regime features using rolling quantiles."""
    df = df.copy()
    
    # Use realized volatility if available, else use close returns
    if "realized_vol_24" in df.columns:
        vol_series = df["realized_vol_24"]
    else:
        vol_series = df["close"].pct_change().rolling(24).std() * np.sqrt(8760)
    
    # Rolling quantiles (1-year window)
    window = min(8760, len(df) // 2)
    if window < 100:
        window = 100
    
    df["vol_regime_low"] = (vol_series <= vol_series.rolling(window).quantile(0.33)).astype(int)
    df["vol_regime_med"] = ((vol_series > vol_series.rolling(window).quantile(0.33)) & 
                            (vol_series <= vol_series.rolling(window).quantile(0.67))).astype(int)
    df["vol_regime_high"] = (vol_series > vol_series.rolling(window).quantile(0.67)).astype(int)
    
    # Volatility trend
    df["vol_trend_24"] = vol_series.rolling(24).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 24 else np.nan
    )
    
    return df

def create_direction_target(
    df: pd.DataFrame,
    horizon: int = 4,
    threshold_pct: float = 0.015,
    method: str = "terciles"
) -> pd.DataFrame:
    """
    Create direction target variable.
    
    Args:
        df: DataFrame with price data
        horizon: Hours ahead to predict
        threshold_pct: Minimum move for long/short (for fixed_threshold)
        method: 'terciles', 'fixed_threshold', or 'quantile'
        
    Returns:
        DataFrame with 'target' column (0=short, 1=neutral, 2=long)
    """
    df = df.copy()
    
    # Future return
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    
    if method == "terciles":
        # Use expanding window to avoid look-ahead bias
        # For training, we can use the full series quantiles
        # But for proper walk-forward, we should use expanding quantiles
        # Here we use global quantiles for simplicity
        low_thresh = future_return.quantile(1/3)
        high_thresh = future_return.quantile(2/3)
        
        df["target"] = 1  # neutral
        df.loc[future_return <= low_thresh, "target"] = 0  # short
        df.loc[future_return >= high_thresh, "target"] = 2  # long
        
    elif method == "fixed_threshold":
        df["target"] = 1  # neutral
        df.loc[future_return <= -threshold_pct, "target"] = 0  # short
        df.loc[future_return >= threshold_pct, "target"] = 2  # long
        
    elif method == "quantile":
        # Use rolling quantiles (expanding window)
        low_thresh = future_return.expanding().quantile(1/3)
        high_thresh = future_return.expanding().quantile(2/3)
        
        df["target"] = 1
        df.loc[future_return <= low_thresh, "target"] = 0
        df.loc[future_return >= high_thresh, "target"] = 2
        
    else:
        raise ValueError(f"Unknown labeling method: {method}")
    
    # Store future return for analysis
    df["future_return"] = future_return
    
    return df

def get_feature_columns(df: pd.DataFrame, exclude_patterns: List[str] = None) -> List[str]:
    """Get list of feature columns (exclude target, metadata)."""
    if exclude_patterns is None:
        exclude_patterns = [
            "target", "future_return", "timestamp", "hour", "dayofweek", "month",
            "open", "high", "low", "close", "volume", "open_interest"
        ]
    
    feature_cols = []
    for col in df.columns:
        if not any(pattern in col for pattern in exclude_patterns):
            feature_cols.append(col)
    
    return feature_cols

def prepare_train_val_test(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_gap: int = 24,
    embargo_pct: float = 0.01
) -> tuple:
    """
    Split data into train/val/test with purging and embargo.
    
    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    # Purge gap: remove samples near boundaries
    train_df = df.iloc[:train_end - purge_gap]
    val_df = df.iloc[train_end + purge_gap:val_end - purge_gap]
    test_df = df.iloc[val_end + purge_gap:]
    
    # Embargo: remove additional samples after test start
    embargo_size = int(len(test_df) * embargo_pct)
    if embargo_size > 0:
        test_df = test_df.iloc[embargo_size:]
    
    return train_df, val_df, test_df
