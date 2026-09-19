"""Combines XGBoost + neural net outputs (averaging / agreement-filter)."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from src.v1.models.direction.xgboost_classifier import XGBoostDirectionModel
from src.v1.models.direction.neural_net_classifier import NeuralNetDirectionModel

class DirectionEnsemble:
    """
    Ensemble of direction models with multiple combination strategies.
    
    Strategies:
    - weighted_average: Weighted average of probabilities
    - agreement_filter: Only predict when models agree above threshold
    - stacking: Meta-learner on model predictions
    """
    
    def __init__(
        self,
        xgb_model: XGBoostDirectionModel,
        nn_model: NeuralNetDirectionModel,
        config: Dict
    ):
        self.xgb_model = xgb_model
        self.nn_model = nn_model
        self.config = config
        self.method = config.get("method", "weighted_average")
        self.xgb_weight = config.get("xgb_weight", 0.7)
        self.nn_weight = config.get("nn_weight", 0.3)
        self.agreement_threshold = config.get("agreement_threshold", 0.6)
        self.meta_learner = None
        self.classes_ = [0, 1, 2]
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ensemble probabilities.
        
        Returns:
            Array of shape (n_samples, 3) with class probabilities
        """
        # Get predictions from both models
        xgb_probs = self.xgb_model.predict_proba(X)
        nn_probs = self.nn_model.predict_proba(X)
        
        if self.method == "weighted_average":
            return self._weighted_average(xgb_probs, nn_probs)
        elif self.method == "agreement_filter":
            return self._agreement_filter(xgb_probs, nn_probs)
        elif self.method == "stacking":
            return self._stacking_predict(xgb_probs, nn_probs)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def _weighted_average(
        self,
        xgb_probs: np.ndarray,
        nn_probs: np.ndarray
    ) -> np.ndarray:
        """Weighted average of probabilities."""
        return self.xgb_weight * xgb_probs + self.nn_weight * nn_probs
    
    def _agreement_filter(
        self,
        xgb_probs: np.ndarray,
        nn_probs: np.ndarray
    ) -> np.ndarray:
        """
        Agreement filter: only predict when both models agree on top class
        with sufficient confidence.
        """
        xgb_preds = np.argmax(xgb_probs, axis=1)
        nn_preds = np.argmax(nn_probs, axis=1)
        xgb_conf = np.max(xgb_probs, axis=1)
        nn_conf = np.max(nn_probs, axis=1)
        
        # Agreement mask
        agree = (xgb_preds == nn_preds) & (xgb_conf >= self.agreement_threshold) & (nn_conf >= self.agreement_threshold)
        
        # Combined probabilities
        combined = self._weighted_average(xgb_probs, nn_probs)
        
        # For non-agreeing samples, return uniform (neutral) or low confidence
        # Here we return the weighted average but the predict() method will handle filtering
        return combined
    
    def _stacking_predict(
        self,
        xgb_probs: np.ndarray,
        nn_probs: np.ndarray
    ) -> np.ndarray:
        """Stacking with meta-learner."""
        if self.meta_learner is None:
            raise ValueError("Meta-learner not fitted. Call fit_stacking() first.")
        
        # Stack predictions as features
        stacked_features = np.hstack([xgb_probs, nn_probs])
        return self.meta_learner.predict_proba(stacked_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        
        if self.method == "agreement_filter":
            # For agreement filter, only predict when models agree
            xgb_probs = self.xgb_model.predict_proba(X)
            nn_probs = self.nn_model.predict_proba(X)
            xgb_preds = np.argmax(xgb_probs, axis=1)
            nn_preds = np.argmax(nn_probs, axis=1)
            xgb_conf = np.max(xgb_probs, axis=1)
            nn_conf = np.max(nn_probs, axis=1)
            
            agree = (xgb_preds == nn_preds) & (xgb_conf >= self.agreement_threshold) & (nn_conf >= self.agreement_threshold)
            
            preds = np.argmax(probs, axis=1)
            # Set non-agreeing to neutral (1)
            preds[~agree] = 1
            return preds
        else:
            return np.argmax(probs, axis=1)
    
    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence filtering."""
        probs = self.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        confident = max_probs >= confidence_threshold
        return preds, confident
    
    def fit_stacking(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        meta_learner_type: str = "logistic"
    ) -> "DirectionEnsemble":
        """
        Fit stacking meta-learner.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for meta-learner training)
            y_val: Validation labels
            meta_learner_type: 'logistic', 'rf', 'xgb'
        """
        # Get base model predictions on validation set
        xgb_val_probs = self.xgb_model.predict_proba(X_val)
        nn_val_probs = self.nn_model.predict_proba(X_val)
        
        # Stack features
        stacked_val = np.hstack([xgb_val_probs, nn_val_probs])
        
        # Train meta-learner
        if meta_learner_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                random_state=42
            )
        elif meta_learner_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        elif meta_learner_type == "xgb":
            import xgboost as xgb
            self.meta_learner = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner_type}")
        
        self.meta_learner.fit(stacked_val, y_val)
        return self
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        confidence_threshold: float = 0.0
    ) -> Dict:
        """
        Evaluate ensemble performance.
        
        Returns:
            Dictionary with metrics
        """
        from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
        
        probs = self.predict_proba(X)
        preds = self.predict(X)
        
        # Overall metrics
        ll = log_loss(y, probs)
        acc = accuracy_score(y, preds)
        
        # Per-class metrics
        report = classification_report(y, preds, target_names=["Short", "Neutral", "Long"], output_dict=True)
        cm = confusion_matrix(y, preds)
        
        # Confidence-based metrics
        if confidence_threshold > 0:
            confident_preds, confident_mask = self.predict_with_confidence(X, confidence_threshold)
            if confident_mask.sum() > 0:
                confident_acc = accuracy_score(y[confident_mask], confident_preds[confident_mask])
                coverage = confident_mask.mean()
            else:
                confident_acc = 0
                coverage = 0
        else:
            confident_acc = acc
            coverage = 1.0
        
        return {
            "log_loss": ll,
            "accuracy": acc,
            "confident_accuracy": confident_acc,
            "coverage": coverage,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "per_class_accuracy": {
                "short": report["Short"]["recall"],
                "neutral": report["Neutral"]["recall"],
                "long": report["Long"]["recall"]
            }
        }

def create_ensemble(
    xgb_model: XGBoostDirectionModel,
    nn_model: NeuralNetDirectionModel,
    config: Dict
) -> DirectionEnsemble:
    """Factory function to create ensemble."""
    return DirectionEnsemble(xgb_model, nn_model, config)
