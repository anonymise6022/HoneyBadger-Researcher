"""ATR%, rolling realized volatility, relative-volume features."""
import pandas as pd
import numpy as np
from typing import List, Optional
from src.v1.models.volatility.realized_vol import (
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility
)
from src.v1.models.volatility.garch import GARCHForecaster

def add_volatility_features(
    df: pd.DataFrame,
    atr_window: int = 24,
    rv_windows: List[int] = None,
    volume_windows: List[int] = None,
    include_garch: bool = False,
    include_implied: bool = False,
    garch_model: Optional[GARCHForecaster] = None
) -> pd.DataFrame:
    """
    Add comprehensive volatility features.
    
    Args:
        df: DataFrame with OHLCV data
        atr_window: Window for ATR calculation
        rv_windows: Windows for realized volatility
        volume_windows: Windows for relative volume
        include_garch: Whether to add GARCH forecasts
        include_implied: Whether to add implied vol features
        garch_model: Pre-fitted GARCH model (for inference)
        
    Returns:
        DataFrame with added volatility features
    """
    if rv_windows is None:
        rv_windows = [24, 168, 720]  # 1d, 1w, 1m
    if volume_windows is None:
        volume_windows = [24, 168]
    
    df = df.copy()
    
    # 1. ATR (Average True Range) and ATR%
    df = _add_atr_features(df, atr_window)
    
    # 2. Realized volatility (multiple methods)
    df = _add_realized_vol_features(df, rv_windows)
    
    # 3. Relative volume features
    df = _add_relative_volume_features(df, volume_windows)
    
    # 4. Volatility of volatility
    df = _add_vol_of_vol_features(df, rv_windows)
    
    # 5. GARCH features (if requested and model provided)
    if include_garch and garch_model is not None:
        df = _add_garch_features(df, garch_model)
    
    # 6. Implied volatility features (if available in data)
    if include_implied and "atm_iv" in df.columns:
        df = _add_implied_vol_features(df)
    
    return df

def _add_atr_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add ATR and ATR% features."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    
    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR
    atr = true_range.rolling(window, min_periods=window//2).mean()
    df[f"atr_{window}"] = atr
    
    # ATR% (normalized by price)
    df[f"atr_pct_{window}"] = atr / close * 100
    
    # Normalized ATR (by rolling median)
    atr_median = atr.rolling(window*5, min_periods=window).median()
    df[f"atr_norm_{window}"] = atr / atr_median
    
    return df

def _add_realized_vol_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add realized volatility features using multiple estimators."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    
    for window in windows:
        # Standard close-to-close
        rv_cc = realized_volatility(close, window=window)
        df[f"realized_vol_{window}"] = rv_cc
        
        # Parkinson (high-low)
        rv_park = parkinson_volatility(high, low, window=window)
        df[f"realized_vol_park_{window}"] = rv_park
        
        # Garman-Klass (OHLC)
        rv_gk = garman_klass_volatility(open_, high, low, close, window=window)
        df[f"realized_vol_gk_{window}"] = rv_gk
        
        # Rogers-Satchell (OHLC, drift-independent)
        rv_rs = rogers_satchell_volatility(open_, high, low, close, window=window)
        df[f"realized_vol_rs_{window}"] = rv_rs
        
        # Volatility ratios
        df[f"rv_ratio_park_cc_{window}"] = rv_park / rv_cc
        df[f"rv_ratio_gk_cc_{window}"] = rv_gk / rv_cc
        df[f"rv_ratio_rs_cc_{window}"] = rv_rs / rv_cc
        
        # Log volatility (for stationarity)
        df[f"log_rv_{window}"] = np.log(rv_cc + 1e-10)
    
    return df

def _add_relative_volume_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Add relative volume features."""
    volume = df["volume"]
    
    for window in windows:
        # Rolling median volume
        vol_median = volume.rolling(window, min_periods=window//2).median()
        df[f"rel_volume_{window}"] = volume / (vol_median + 1e-10)
        
        # Volume z-score
        vol_mean = volume.rolling(window, min_periods=window//2).mean()
        vol_std = volume.rolling(window, min_periods=window//2).std()
        df[f"volume_zscore_{window}"] = (volume - vol_mean) / (vol_std + 1e-10)
        
        # Volume trend
        df[f"volume_trend_{window}"] = volume.rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
    
    return df

def _add_vol_of_vol_features(df: pd.DataFrame, rv_windows: List[int]) -> pd.DataFrame:
    """Add volatility of volatility features."""
    for window in rv_windows:
        rv_col = f"realized_vol_{window}"
        if rv_col in df.columns:
            # Rolling std of realized vol
            df[f"vol_of_vol_{window}"] = df[rv_col].rolling(window).std()
            
            # Realized vol z-score
            rv_mean = df[rv_col].rolling(window*5, min_periods=window).mean()
            rv_std = df[rv_col].rolling(window*5, min_periods=window).std()
            df[f"rv_zscore_{window}"] = (df[rv_col] - rv_mean) / (rv_std + 1e-10)
    
    return df

def _add_garch_features(df: pd.DataFrame, garch_model: GARCHForecaster) -> pd.DataFrame:
    """Add GARCH forecast features."""
    # Get conditional volatility forecasts
    forecasts = garch_model.forecast_conditional_vol(df["close"])
    
    # Align forecasts with dataframe
    forecast_df = pd.DataFrame(forecasts, index=df.index)
    forecast_df.columns = [f"garch_forecast_{h}h" for h in forecast_df.columns]
    
    # Merge
    df = df.join(forecast_df)
    
    # GARCH forecast ratios
    if "garch_forecast_1h" in df.columns and "realized_vol_24" in df.columns:
        df["garch_rv_ratio_1h"] = df["garch_forecast_1h"] / df["realized_vol_24"]
    
    return df

def _add_implied_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add implied volatility features."""
    if "atm_iv" not in df.columns:
        return df
    
    df = df.copy()
    iv = df["atm_iv"]
    
    # IV term structure (if multiple expiries available)
    # For now, just use ATM IV
    
    # IV percentile (1-year lookback)
    window = min(8760, len(df) // 2)
    if window < 100:
        window = 100
    
    df["iv_percentile"] = iv.rolling(window).rank(pct=True)
    
    # IV vs realized vol spread
    if "realized_vol_24" in df.columns:
        df["iv_rv_spread"] = iv - df["realized_vol_24"]
        df["iv_rv_ratio"] = iv / (df["realized_vol_24"] + 1e-10)
    
    # IV skew (if we have call/put IV)
    if "atm_iv_call" in df.columns and "atm_iv_put" in df.columns:
        df["iv_skew"] = df["atm_iv_put"] - df["atm_iv_call"]
    
    return df
