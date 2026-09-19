"""XGBoost 3-class (short/neutral/long) direction model -- current main model."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score, classification_report
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

class XGBoostDirectionModel:
    """
    XGBoost 3-class direction classifier with time-series CV.
    
    Classes: 0=Short, 1=Neutral, 2=Long
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.classes_ = [0, 1, 2]
        self.best_iteration = 0
        self.feature_importance = None
        self.cv_scores = None
        
    def _get_params(self) -> Dict:
        """Get XGBoost parameters from config."""
        params = {
            "objective": self.config.get("objective", "multi:softprob"),
            "num_class": self.config.get("num_class", 3),
            "eval_metric": self.config.get("eval_metric", "mlogloss"),
            "eta": self.config.get("eta", 0.03),
            "max_depth": self.config.get("max_depth", 6),
            "min_child_weight": self.config.get("min_child_weight", 10),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "gamma": self.config.get("gamma", 0.1),
            "reg_alpha": self.config.get("reg_alpha", 0.1),
            "reg_lambda": self.config.get("reg_lambda", 1.0),
            "n_estimators": self.config.get("n_estimators", 2000),
            "early_stopping_rounds": self.config.get("early_stopping_rounds", 100),
            "random_state": self.config.get("random_seed", 42),
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        # Class weights
        class_weights = self.config.get("class_weights", "balanced")
        if class_weights == "balanced":
            # Will be set in fit()
            pass
        elif class_weights == "custom":
            custom = self.config.get("custom_weights", [1.0, 0.5, 1.0])
            params["scale_pos_weight"] = custom  # Not directly supported for multi-class
        
        return params
    
    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> xgb.DMatrix:
        """Prepare DMatrix with proper class weights."""
        if sample_weight is not None:
            dmatrix = xgb.DMatrix(X, label=y, weight=sample_weight, feature_names=X.columns.tolist())
        else:
            dmatrix = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
        return dmatrix
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> "XGBoostDirectionModel":
        """
        Train the model with optional validation set.
        
        Args:
            X_train: Training features
            y_train: Training labels (0, 1, 2)
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Sample weights for training
            verbose: Print training progress
            
        Returns:
            Self
        """
        self.feature_names = X_train.columns.tolist()
        
        # Calculate class weights if balanced
        class_weights = self.config.get("class_weights", "balanced")
        if class_weights == "balanced" and sample_weight is None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y_train)
        
        # Prepare data
        dtrain = self._prepare_data(X_train, y_train, sample_weight)
        
        # Validation set
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = self._prepare_data(X_val, y_val)
            evals.append((dval, "val"))
        
        # Train
        params = self._get_params()
        n_estimators = params.pop("n_estimators")
        early_stopping = params.pop("early_stopping_rounds")
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=100 if verbose else False
        )
        
        self.best_iteration = self.model.best_iteration
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": [self.model.get_score(importance_type="gain").get(f, 0) 
                          for f in self.feature_names]
        }).sort_values("importance", ascending=False)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        probs = self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))
        return probs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence filtering.
        
        Returns:
            (predictions, confident_mask)
        """
        probs = self.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        confident = max_probs >= confidence_threshold
        return preds, confident
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        purge_gap: int = 24,
        embargo_pct: float = 0.01
    ) -> Dict:
        """
        Time-series cross-validation with purging and embargo.
        
        Returns:
            Dictionary with CV scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {"logloss": [], "accuracy": [], "per_class_acc": []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Apply purge gap
            if purge_gap > 0:
                train_idx = train_idx[:-purge_gap] if len(train_idx) > purge_gap else train_idx
                val_idx = val_idx[purge_gap:] if len(val_idx) > purge_gap else val_idx
            
            # Apply embargo
            if embargo_pct > 0:
                embargo_size = int(len(val_idx) * embargo_pct)
                val_idx = val_idx[embargo_size:]
            
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold model
            fold_model = XGBoostDirectionModel(self.config)
            fold_model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            val_probs = fold_model.predict_proba(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            
            ll = log_loss(y_val, val_probs)
            acc = accuracy_score(y_val, val_preds)
            
            # Per-class accuracy
            class_acc = {}
            for cls in self.classes_:
                mask = y_val == cls
                if mask.sum() > 0:
                    class_acc[cls] = accuracy_score(y_val[mask], val_preds[mask])
                else:
                    class_acc[cls] = 0
            
            scores["logloss"].append(ll)
            scores["accuracy"].append(acc)
            scores["per_class_acc"].append(class_acc)
            
            if verbose:
                print(f"Fold {fold+1}: LogLoss={ll:.4f}, Acc={acc:.4f}")
        
        # Aggregate
        self.cv_scores = {
            "mean_logloss": np.mean(scores["logloss"]),
            "std_logloss": np.std(scores["logloss"]),
            "mean_accuracy": np.mean(scores["accuracy"]),
            "std_accuracy": np.std(scores["accuracy"]),
            "per_class_accuracy": {
                cls: np.mean([s[cls] for s in scores["per_class_acc"]])
                for cls in self.classes_
            }
        }
        
        return self.cv_scores
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save XGBoost model
        self.model.save_model(f"{path}.json")
        
        # Save metadata
        metadata = {
            "config": self.config,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "feature_importance": self.feature_importance.to_dict("records") if self.feature_importance is not None else None,
            "cv_scores": self.cv_scores
        }
        with open(f"{path}_meta.json", "w") as f:
            json.dump(metadata, f, default=str)
    
    @classmethod
    def load(cls, path: str) -> "XGBoostDirectionModel":
        """Load model from disk."""
        # Load metadata
        with open(f"{path}_meta.json", "r") as f:
            metadata = json.load(f)
        
        # Create instance
        model = cls(metadata["config"])
        model.feature_names = metadata["feature_names"]
        model.best_iteration = metadata["best_iteration"]
        model.cv_scores = metadata.get("cv_scores")
        
        if metadata.get("feature_importance"):
            model.feature_importance = pd.DataFrame(metadata["feature_importance"])
        
        # Load XGBoost model
        model.model = xgb.Booster()
        model.model.load_model(f"{path}.json")
        
        return model

def train_xgboost_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    config: Dict = None
) -> XGBoostDirectionModel:
    """
    Convenience function to train XGBoost model.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        config: Model configuration
        
    Returns:
        Trained XGBoostDirectionModel
    """
    if config is None:
        config = {}
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    model = XGBoostDirectionModel(config)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    return model
