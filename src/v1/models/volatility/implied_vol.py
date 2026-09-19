"""Back out implied volatility from market option prices via Black-Scholes inversion."""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Union, Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Black-Scholes option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    if T <= 0 or sigma <= 0:
        # Intrinsic value at expiry
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> dict:
    """Calculate Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common terms
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    sqrt_T = np.sqrt(T)
    
    if option_type == "call":
        delta = cdf_d1
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) 
                 - r * K * np.exp(-r * T) * cdf_d2)
        rho = K * T * np.exp(-r * T) * cdf_d2
    else:
        delta = cdf_d1 - 1
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # Per 1% vol change
    
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta / 365,  # Per day
        "rho": rho / 100  # Per 1% rate change
    }

def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    max_iter: int = 100,
    tol: float = 1e-6
) -> Optional[float]:
    """
    Calculate implied volatility using Brent's method.
    
    Args:
        price: Market option price
        S: Spot price
        K: Strike price
        T: Time to expiry (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        max_iter: Maximum iterations
        tol: Tolerance
        
    Returns:
        Implied volatility or None if failed
    """
    # Input validation
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    
    # Intrinsic value bounds
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0)
        max_price = S
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
        max_price = K * np.exp(-r * T)
    
    if price < intrinsic - 1e-8 or price > max_price + 1e-8:
        return None  # Arbitrage violation
    
    # Define objective function
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price
    
    # Search bounds
    try:
        # Lower bound: near zero
        # Upper bound: start with 500% and expand if needed
        iv = brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=max_iter)
        return iv
    except ValueError:
        # Try expanding upper bound
        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
            return iv
        except ValueError:
            return None

def implied_volatility_surface(
    options_df: pd.DataFrame,
    S: float,
    r: float = 0.05
) -> pd.DataFrame:
    """
    Calculate implied volatility for a DataFrame of options.
    
    Args:
        options_df: DataFrame with columns ['strike', 'expiry', 'option_type', 'bid', 'ask', 'mid']
        S: Current spot price
        r: Risk-free rate
        
    Returns:
        DataFrame with added 'iv' column
    """
    df = options_df.copy()
    
    # Calculate mid price if not present
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    
    # Time to expiry in years
    df["T"] = (df["expiry"] - df["timestamp"]).dt.total_seconds() / (365.25 * 24 * 3600)
    
    # Calculate IV for each option
    ivs = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            price=row["mid"],
            S=S,
            K=row["strike"],
            T=row["T"],
            r=r,
            option_type=row["option_type"]
        )
        ivs.append(iv)
    
    df["iv"] = ivs
    
    # Filter out failed calculations
    df = df[df["iv"].notna()]
    
    return df

def atm_implied_vol(
    options_df: pd.DataFrame,
    S: float,
    r: float = 0.05,
    moneyness_range: Tuple[float, float] = (0.95, 1.05)
) -> Optional[float]:
    """
    Calculate ATM implied volatility from options chain.
    
    Args:
        options_df: Options chain for a single timestamp
        S: Spot price
        r: Risk-free rate
        moneyness_range: Range around ATM to consider
        
    Returns:
        ATM IV or None
    """
    if len(options_df) == 0:
        return None
    
    # Calculate IVs
    df = implied_volatility_surface(options_df, S, r)
    
    if len(df) == 0:
        return None
    
    # Filter near ATM
    df["moneyness"] = df["strike"] / S
    atm_options = df[
        (df["moneyness"] >= moneyness_range[0]) & 
        (df["moneyness"] <= moneyness_range[1])
    ]
    
    if len(atm_options) == 0:
        return None
    
    # Weight by open interest or volume
    if "open_interest" in atm_options.columns and atm_options["open_interest"].sum() > 0:
        weights = atm_options["open_interest"]
    elif "volume" in atm_options.columns and atm_options["volume"].sum() > 0:
        weights = atm_options["volume"]
    else:
        weights = np.ones(len(atm_options))
    
    atm_iv = np.average(atm_options["iv"], weights=weights)
    return atm_iv

def iv_surface_interpolation(
    options_df: pd.DataFrame,
    S: float,
    r: float = 0.05,
    target_dtes: list = [7, 14, 30, 60, 90],
    target_moneyness: list = [0.8, 0.9, 1.0, 1.1, 1.2]
) -> pd.DataFrame:
    """
    Interpolate IV surface to standard grid.
    
    Returns:
        DataFrame with interpolated IVs
    """
    from scipy.interpolate import griddata
    
    # Calculate IVs
    df = implied_volatility_surface(options_df, S, r)
    
    if len(df) < 4:
        return pd.DataFrame()
    
    # Prepare points for interpolation
    df["moneyness"] = df["strike"] / S
    df["dte"] = df["T"] * 365.25
    
    points = df[["moneyness", "dte"]].values
    values = df["iv"].values
    
    # Create target grid
    grid_m, grid_d = np.meshgrid(target_moneyness, target_dtes)
    grid_points = np.column_stack([grid_m.ravel(), grid_d.ravel()])
    
    # Interpolate
    try:
        interp_iv = griddata(points, values, grid_points, method="cubic", fill_value=np.nan)
        interp_iv = interp_iv.reshape(len(target_dtes), len(target_moneyness))
    except:
        return pd.DataFrame()
    
    # Create result DataFrame
    result = pd.DataFrame(
        interp_iv,
        index=target_dtes,
        columns=target_moneyness
    )
    result.index.name = "dte"
    result.columns.name = "moneyness"
    
    return result

def vix_style_iv(
    options_df: pd.DataFrame,
    S: float,
    r: float = 0.05,
    target_dte: int = 30
) -> Optional[float]:
    """
    Calculate VIX-style 30-day implied volatility.
    
    Uses the CBOE VIX methodology (variance swap replication).
    """
    # This is a simplified version
    # Full VIX calculation requires specific option selection and weighting
    
    df = implied_volatility_surface(options_df, S, r)
    if len(df) == 0:
        return None
    
    df["dte"] = df["T"] * 365.25
    
    # Find options bracketing target DTE
    near = df[df["dte"] <= target_dte].nlargest(1, "dte")
    far = df[df["dte"] >= target_dte].nsmallest(1, "dte")
    
    if len(near) == 0 or len(far) == 0:
        return None
    
    # Simple interpolation
    near_iv = near["iv"].values[0]
    far_iv = far["iv"].values[0]
    near_dte = near["dte"].values[0]
    far_dte = far["dte"].values[0]
    
    # Variance interpolation
    near_var = near_iv ** 2 * near_dte
    far_var = far_iv ** 2 * far_dte
    
    interp_var = near_var + (far_var - near_var) * (target_dte - near_dte) / (far_dte - near_dte)
    interp_iv = np.sqrt(interp_var / target_dte)
    
    return interp_iv
