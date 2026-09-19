"""Shared CSV/parquet loading + schema validation used by every script."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union
import yaml
import logging

logger = logging.getLogger(__name__)

# Expected schemas for different data types
OHLCV_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

OI_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    "open_interest": "float64",
}

OPTIONS_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",
    "symbol": "object",
    "strike": "float64",
    "expiry": "datetime64[ns, UTC]",
    "option_type": "object",  # call/put
    "bid": "float64",
    "ask": "float64",
    "mid": "float64",
    "iv": "float64",
    "volume": "float64",
    "open_interest": "float64",
}

def load_ohlcv(
    path: Union[str, Path],
    freq: str = "1H",
    start: Optional[str] = None,
    end: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Load and validate OHLCV data.
    
    Args:
        path: Path to CSV/Parquet file
        freq: Expected frequency (for validation)
        start: Optional start date filter
        end: Optional end date filter
        validate: Whether to validate schema and continuity
        
    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Ensure timestamp column exists and is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif df.index.name != "timestamp":
        raise ValueError("Data must have 'timestamp' column or datetime index")
    
    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # Sort and remove duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    # Filter date range
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    
    # Validate schema
    if validate:
        _validate_schema(df, OHLCV_SCHEMA, "OHLCV")
        _validate_continuity(df, freq)
        _validate_ohlcv_logic(df)
    
    # Ensure required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df[required_cols].astype("float64")

def load_open_interest(
    path: Union[str, Path],
    freq: str = "1H",
    start: Optional[str] = None,
    end: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """Load and validate open interest data."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    
    if validate:
        _validate_schema(df, OI_SCHEMA, "Open Interest")
        _validate_continuity(df, freq)
    
    if "open_interest" not in df.columns:
        raise ValueError("Missing 'open_interest' column")
    
    return df[["open_interest"]].astype("float64")

def load_options_chain(
    path: Union[str, Path],
    start: Optional[str] = None,
    end: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """Load and validate options chain data."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Handle timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise ValueError("Options data must have 'timestamp' column")
    
    # Handle expiry
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    
    # Sort
    df = df.sort_values(["timestamp", "expiry", "strike", "option_type"])
    df = df.reset_index(drop=True)
    
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    
    if validate:
        _validate_schema(df, OPTIONS_SCHEMA, "Options Chain")
    
    return df

def load_config(config_path: Union[str, Path]) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config: Dict, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def _validate_schema(df: pd.DataFrame, schema: Dict, name: str) -> None:
    """Validate DataFrame schema."""
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            logger.warning(f"{name}: Missing column '{col}'")
            continue
        
        actual_dtype = str(df[col].dtype)
        if expected_dtype not in actual_dtype and not (
            expected_dtype == "float64" and actual_dtype in ["float32", "int64", "int32"]
        ):
            logger.warning(
                f"{name}: Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
            )

def _validate_continuity(df: pd.DataFrame, freq: str) -> None:
    """Check for gaps in time series."""
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    missing = expected.difference(df.index)
    if len(missing) > 0:
        pct_missing = len(missing) / len(expected) * 100
        logger.warning(f"Time series has {len(missing)} missing periods ({pct_missing:.2f}%)")
        if pct_missing > 5:
            logger.error(f"High missing data percentage: {pct_missing:.2f}%")

def _validate_ohlcv_logic(df: pd.DataFrame) -> None:
    """Validate OHLCV logical consistency."""
    # High >= Low
    invalid_hl = df["high"] < df["low"]
    if invalid_hl.any():
        logger.warning(f"Found {invalid_hl.sum()} rows where high < low")
    
    # High >= Open, Close
    invalid_ho = df["high"] < df["open"]
    invalid_hc = df["high"] < df["close"]
    if invalid_ho.any() or invalid_hc.any():
        logger.warning("Found rows where high < open or high < close")
    
    # Low <= Open, Close
    invalid_lo = df["low"] > df["open"]
    invalid_lc = df["low"] > df["close"]
    if invalid_lo.any() or invalid_lc.any():
        logger.warning("Found rows where low > open or low > close")
    
    # Non-negative volume
    neg_vol = df["volume"] < 0
    if neg_vol.any():
        logger.warning(f"Found {neg_vol.sum()} rows with negative volume")
    
    # Positive prices
    for col in ["open", "high", "low", "close"]:
        non_pos = df[col] <= 0
        if non_pos.any():
            logger.warning(f"Found {non_pos.sum()} rows with non-positive {col}")

def merge_data_sources(
    ohlcv: pd.DataFrame,
    oi: Optional[pd.DataFrame] = None,
    options: Optional[pd.DataFrame] = None,
    how: str = "left"
) -> pd.DataFrame:
    """Merge multiple data sources on timestamp index."""
    df = ohlcv.copy()
    
    if oi is not None:
        df = df.join(oi, how=how, rsuffix="_oi")
    
    if options is not None:
        # Aggregate options data by timestamp (e.g., ATM IV, total volume)
        opt_agg = _aggregate_options_by_time(options)
        df = df.join(opt_agg, how=how, rsuffix="_opt")
    
    return df

def _aggregate_options_by_time(options: pd.DataFrame) -> pd.DataFrame:
    """Aggregate options chain data to hourly frequency."""
    # Calculate time to expiry in years
    options = options.copy()
    options["dte"] = (options["expiry"] - options["timestamp"]).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Filter reasonable DTE
    options = options[(options["dte"] > 0) & (options["dte"] < 1)]
    
    # Find ATM options for each timestamp
    # Group by timestamp and find strike closest to spot (approximate with median strike)
    spot_approx = options.groupby("timestamp")["strike"].median().rename("spot_approx")
    options = options.join(spot_approx, on="timestamp")
    options["moneyness"] = options["strike"] / options["spot_approx"]
    
    # Select near-ATM options (0.95 < moneyness < 1.05)
    atm_options = options[(options["moneyness"] > 0.95) & (options["moneyness"] < 1.05)]
    
    # Aggregate by timestamp
    agg = atm_options.groupby("timestamp").agg(
        atm_iv_call=("iv", lambda x: x[atm_options.loc[x.index, "option_type"] == "call"].mean()),
        atm_iv_put=("iv", lambda x: x[atm_options.loc[x.index, "option_type"] == "put"].mean()),
        total_opt_volume=("volume", "sum"),
        total_opt_oi=("open_interest", "sum"),
    ).rename_axis("timestamp")
    
    # Average call/put IV
    agg["atm_iv"] = agg[["atm_iv_call", "atm_iv_put"]].mean(axis=1)
    agg = agg.drop(["atm_iv_call", "atm_iv_put"], axis=1)
    
    return agg
