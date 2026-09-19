"""Historical/realized volatility calculation from price data."""
import numpy as np
import pandas as pd
from typing import Union, Optional

def realized_volatility(
    prices: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760,
    method: str = "std"
) -> pd.Series:
    """
    Calculate realized volatility from price series.
    
    Args:
        prices: Price series
        window: Rolling window (in periods)
        annualization_factor: Periods per year (8760 for hourly)
        method: 'std', 'parkinson', 'garman_klass', 'rogers_satchell'
        
    Returns:
        Annualized realized volatility series
    """
    if method == "std":
        returns = np.log(prices / prices.shift(1))
        rv = returns.rolling(window, min_periods=window//2).std() * np.sqrt(annualization_factor)
    elif method == "parkinson":
        rv = parkinson_volatility(prices, window=window, annualization_factor=annualization_factor)
    elif method == "garman_klass":
        # Need OHLC data
        raise ValueError("Garman-Klass requires OHLC data, use garman_klass_volatility directly")
    elif method == "rogers_satchell":
        raise ValueError("Rogers-Satchell requires OHLC data, use rogers_satchell_volatility directly")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return rv

def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Parkinson (1980) high-low volatility estimator.
    
    More efficient than close-to-close for same data frequency.
    Assumes zero drift.
    """
    # Parkinson estimator: sqrt(1/(4*ln(2)*N) * sum(ln(H/L)^2))
    hl_ratio = np.log(high / low)
    hl_ratio_sq = hl_ratio ** 2
    
    # Rolling sum
    rolling_sum = hl_ratio_sq.rolling(window, min_periods=window//2).sum()
    
    # Parkinson factor
    factor = 1 / (4 * np.log(2) * window)
    
    rv = np.sqrt(factor * rolling_sum * annualization_factor)
    return rv

def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC volatility estimator.
    
    More efficient than Parkinson, assumes zero drift.
    """
    # Garman-Klass estimator
    # sigma^2 = 1/N * sum[ 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2 ]
    hl = np.log(high / low)
    co = np.log(close / open_)
    
    term1 = 0.5 * hl ** 2
    term2 = (2 * np.log(2) - 1) * co ** 2
    
    estimator = term1 - term2
    rolling_sum = estimator.rolling(window, min_periods=window//2).sum()
    
    rv = np.sqrt(rolling_sum / window * annualization_factor)
    return rv

def rogers_satchell_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Rogers-Satchell (1991) OHLC volatility estimator.
    
    Drift-independent, more efficient than Garman-Klass when drift != 0.
    """
    # Rogers-Satchell estimator
    # sigma^2 = 1/N * sum[ ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) ]
    hc = np.log(high / close)
    ho = np.log(high / open_)
    lc = np.log(low / close)
    lo = np.log(low / open_)
    
    estimator = hc * ho + lc * lo
    rolling_sum = estimator.rolling(window, min_periods=window//2).sum()
    
    rv = np.sqrt(rolling_sum / window * annualization_factor)
    return rv

def yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator.
    
    Combines overnight and intraday volatility, handles drift and jumps.
    Most efficient for OHLC data.
    """
    # Overnight volatility (close-to-open)
    co = np.log(open_ / close.shift(1))
    overnight_var = co.rolling(window, min_periods=window//2).var()
    
    # Open-to-close volatility
    oc = np.log(close / open_)
    oc_var = oc.rolling(window, min_periods=window//2).var()
    
    # Rogers-Satchell (intraday)
    rs = rogers_satchell_volatility(open_, high, low, close, window=1, annualization_factor=1)
    rs_var = (rs ** 2).rolling(window, min_periods=window//2).sum() / window
    
    # Weighting factor
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    # Combined estimator
    yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
    rv = np.sqrt(yz_var * annualization_factor)
    
    return rv

def realized_quarticity(
    prices: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Realized quarticity (volatility of volatility).
    
    Used for confidence intervals on realized volatility.
    """
    returns = np.log(prices / prices.shift(1))
    ret_pow4 = returns ** 4
    
    # Realized quarticity: (N/3) * sum(r_i^4)
    rq = (window / 3) * ret_pow4.rolling(window, min_periods=window//2).sum()
    
    # Annualize
    rq_ann = rq * (annualization_factor / window) ** 2
    
    return rq_ann

def bipower_variation(
    prices: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Bipower variation (robust to jumps).
    
    BPV = (pi/2) * sum(|r_i| * |r_{i-1}|)
    """
    returns = np.log(prices / prices.shift(1))
    abs_ret = returns.abs()
    abs_ret_lag = abs_ret.shift(1)
    
    bpv = (np.pi / 2) * (abs_ret * abs_ret_lag).rolling(window, min_periods=window//2).sum()
    bpv_ann = bpv * annualization_factor / window
    
    return bpv_ann

def jump_variation(
    prices: pd.Series,
    window: int = 24,
    annualization_factor: float = 8760
) -> pd.Series:
    """
    Jump variation = Realized variance - Bipower variation.
    
    Positive values indicate jump activity.
    """
    rv = realized_volatility(prices, window, annualization_factor) ** 2
    bpv = bipower_variation(prices, window, annualization_factor)
    
    jump_var = rv - bpv
    jump_var = jump_var.clip(lower=0)  # No negative jump variation
    
    return jump_var
