"""GARCH-based near-term volatility forecast."""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from arch import arch_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class GARCHForecaster:
    """
    GARCH(1,1) volatility forecaster with automatic refitting.
    
    Uses the `arch` package for robust estimation.
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean_model: str = "constant",
        vol_target: str = "realized",
        forecast_horizon: int = 24,
        min_obs: int = 500,
        refit_frequency: int = 24,
        random_seed: int = 42
    ):
        self.p = p
        self.q = q
        self.dist = dist
        self.mean_model = mean_model
        self.vol_target = vol_target
        self.forecast_horizon = forecast_horizon
        self.min_obs = min_obs
        self.refit_frequency = refit_frequency
        self.random_seed = random_seed
        
        self.model = None
        self.fitted_model = None
        self.last_fit_idx = None
        self.conditional_vol = None
        self.fitted_params = None
        
    def fit(self, returns: pd.Series) -> "GARCHForecaster":
        """
        Fit GARCH model to returns.
        
        Args:
            returns: Series of returns (not prices)
            
        Returns:
            Self
        """
        # Clean returns
        returns = returns.dropna()
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < self.min_obs:
            raise ValueError(f"Need at least {self.min_obs} observations, got {len(returns)}")
        
        # Scale returns for numerical stability (arch expects ~percentage returns)
        scale = 100 if returns.std() < 0.01 else 1
        scaled_returns = returns * scale
        
        # Specify model
        self.model = arch_model(
            scaled_returns,
            mean=self.mean_model,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False
        )
        
        # Fit
        self.fitted_model = self.model.fit(
            update_freq=0,
            show_warning=False,
            options={"ftol": 1e-8, "maxiter": 1000}
        )
        
        # Store conditional volatility (rescaled back)
        self.conditional_vol = self.fitted_model.conditional_volatility / scale
        self.fitted_params = self.fitted_model.params.to_dict()
        self.last_fit_idx = returns.index[-1]
        
        return self
    
    def refit_if_needed(self, returns: pd.Series, current_idx: pd.Timestamp) -> bool:
        """
        Refit model if enough new data since last fit.
        
        Returns:
            True if refitted, False otherwise
        """
        if self.fitted_model is None:
            self.fit(returns)
            return True
        
        # Check if enough new observations
        new_obs = returns.loc[self.last_fit_idx:current_idx]
        new_obs = new_obs.dropna()
        
        if len(new_obs) >= self.refit_frequency:
            self.fit(returns)
            return True
        
        return False
    
    def forecast_conditional_vol(
        self,
        prices: pd.Series,
        horizon: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Forecast conditional volatility for multiple horizons.
        
        Args:
            prices: Price series (used to compute returns for forecasting)
            horizon: Forecast horizon (default: self.forecast_horizon)
            
        Returns:
            DataFrame with forecasts for each horizon
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        if horizon is None:
            horizon = self.forecast_horizon
        
        # Get returns up to last fit
        returns = prices.pct_change().dropna()
        returns = returns.loc[:self.last_fit_idx]
        
        # Forecast
        forecast = self.fitted_model.forecast(horizon=horizon, reindex=False)
        cond_var = forecast.variance.iloc[-1]  # Last row has forecasts
        
        # Convert to volatility (annualized)
        # arch returns variance of scaled returns, so we need to rescale
        scale = 100 if returns.std() < 0.01 else 1
        cond_vol = np.sqrt(cond_var) / scale * np.sqrt(8760)  # Annualize
        
        # Create DataFrame with forecast horizons
        forecast_index = pd.RangeIndex(1, horizon + 1)
        forecast_df = pd.DataFrame(
            {h: cond_vol[h-1] for h in forecast_index},
            index=[self.last_fit_idx]
        )
        
        return forecast_df
    
    def forecast_rolling(
        self,
        prices: pd.Series,
        horizon: Optional[int] = None,
        step: int = 1
    ) -> pd.DataFrame:
        """
        Generate rolling forecasts with periodic refitting.
        
        Args:
            prices: Full price series
            horizon: Forecast horizon
            step: Step size for rolling (1 = every period)
            
        Returns:
            DataFrame with forecasts aligned to price index
        """
        if horizon is None:
            horizon = self.forecast_horizon
        
        returns = prices.pct_change().dropna()
        forecasts = []
        forecast_times = []
        
        # Initial fit
        initial_returns = returns.iloc[:self.min_obs]
        self.fit(initial_returns)
        
        # Rolling forecast
        for i in range(self.min_obs, len(returns), step):
            current_time = returns.index[i]
            
            # Refit if needed
            self.refit_if_needed(returns.iloc[:i+1], current_time)
            
            # Forecast
            forecast = self.forecast_conditional_vol(prices, horizon)
            forecasts.append(forecast.iloc[0].values)
            forecast_times.append(current_time)
        
        # Create result DataFrame
        forecast_df = pd.DataFrame(
            forecasts,
            index=forecast_times,
            columns=[f"garch_forecast_{h}h" for h in range(1, horizon+1)]
        )
        
        return forecast_df
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get in-sample conditional volatility."""
        if self.conditional_vol is None:
            raise ValueError("Model must be fitted first")
        return self.conditional_vol
    
    def simulate_paths(
        self,
        n_paths: int = 1000,
        horizon: int = 24,
        initial_value: float = 0.0
    ) -> np.ndarray:
        """
        Simulate future return paths.
        
        Returns:
            Array of shape (n_paths, horizon)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        simulations = self.fitted_model.simulate(
            params=self.fitted_model.params,
            nobs=horizon,
            nsim=n_paths,
            random_state=self.random_seed
        )
        
        # Rescale
        scale = 100 if self.fitted_model.resid.std() < 0.01 else 1
        return simulations.data / scale

def garch_forecast_pipeline(
    prices: pd.Series,
    config: Dict,
    expanding: bool = True
) -> pd.DataFrame:
    """
    Full GARCH forecasting pipeline.
    
    Args:
        prices: Price series
        config: GARCH configuration
        expanding: Whether to use expanding window (True) or rolling refit
        
    Returns:
        DataFrame with forecasts
    """
    forecaster = GARCHForecaster(
        p=config.get("p", 1),
        q=config.get("q", 1),
        dist=config.get("dist", "normal"),
        mean_model=config.get("mean_model", "constant"),
        forecast_horizon=config.get("forecast_horizon", 24),
        min_obs=config.get("min_obs", 500),
        refit_frequency=config.get("refit_frequency", 24)
    )
    
    if expanding:
        return forecaster.forecast_rolling(prices, step=config.get("refit_frequency", 24))
    else:
        forecaster.fit(prices.pct_change().dropna())
        return forecaster.forecast_conditional_vol(prices)
