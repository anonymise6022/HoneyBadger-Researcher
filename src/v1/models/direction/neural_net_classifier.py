"""PyTorch neural-net direction model -- kept for ensembling, not the primary model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

class DirectionNN(nn.Module):
    """Neural network for 3-class direction classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
        num_classes: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class NeuralNetDirectionModel:
    """
    PyTorch neural network for direction classification.
    
    Includes proper time-series CV, early stopping, and calibration.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.classes_ = [0, 1, 2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.cv_scores = None
        
    def _prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare tensors from DataFrame."""
        X_array = X.values.astype(np.float32)
        
        if fit_scaler:
            X_array = self.scaler.fit_transform(X_array)
        else:
            X_array = self.scaler.transform(X_array)
        
        X_tensor = torch.FloatTensor(X_array).to(self.device)
        y_tensor = torch.LongTensor(y.values.astype(np.int64)).to(self.device)
        
        return X_tensor, y_tensor
    
    def _create_dataloader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool = False
    ) -> DataLoader:
        """Create DataLoader."""
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> "NeuralNetDirectionModel":
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weight: Sample weights (not directly supported, use weighted sampler)
            verbose: Print training progress
            
        Returns:
            Self
        """
        self.feature_names = X_train.columns.tolist()
        input_dim = len(self.feature_names)
        
        # Initialize model
        self.model = DirectionNN(
            input_dim=input_dim,
            hidden_layers=self.config.get("hidden_layers", [128, 64, 32]),
            dropout=self.config.get("dropout", 0.3),
            activation=self.config.get("activation", "relu"),
            batch_norm=self.config.get("batch_norm", True),
            num_classes=3
        ).to(self.device)
        
        # Loss and optimizer
        class_weights = self.config.get("class_weights", "balanced")
        if class_weights == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weights_tensor = torch.FloatTensor(weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=verbose
        )
        
        # Prepare data
        X_train_t, y_train_t = self._prepare_data(X_train, y_train, fit_scaler=True)
        train_loader = self._create_dataloader(
            X_train_t, y_train_t,
            batch_size=self.config.get("batch_size", 256),
            shuffle=True
        )
        
        if X_val is not None:
            X_val_t, y_val_t = self._prepare_data(X_val, y_val)
            val_loader = self._create_dataloader(
                X_val_t, y_val_t,
                batch_size=self.config.get("batch_size", 256),
                shuffle=False
            )
        else:
            val_loader = None
        
        # Training loop
        epochs = self.config.get("epochs", 200)
        patience = self.config.get("early_stopping_patience", 20)
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation
            val_loss = 0
            val_acc = 0
            if val_loader is not None:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * batch_X.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                val_loss /= val_total
                val_acc = val_correct / val_total
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                val_loss = train_loss
                val_acc = train_acc
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Load best model
        if val_loader is not None and "best_model_state" in locals():
            self.model.load_state_dict(best_model_state)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        self.model.eval()
        X_tensor, _ = self._prepare_data(X, pd.Series(np.zeros(len(X))), fit_scaler=False)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        
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
        """Predict with confidence filtering."""
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
        """Time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = {"logloss": [], "accuracy": [], "per_class_acc": []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Apply purge and embargo
            if purge_gap > 0:
                train_idx = train_idx[:-purge_gap] if len(train_idx) > purge_gap else train_idx
                val_idx = val_idx[purge_gap:] if len(val_idx) > purge_gap else val_idx
            
            if embargo_pct > 0:
                embargo_size = int(len(val_idx) * embargo_pct)
                val_idx = val_idx[embargo_size:]
            
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold model
            fold_model = NeuralNetDirectionModel(self.config)
            fold_model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate
            val_probs = fold_model.predict_proba(X_val)
            val_preds = np.argmax(val_probs, axis=1)
            
            ll = log_loss(y_val, val_probs)
            acc = accuracy_score(y_val, val_preds)
            
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
        
        # Save model state
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "history": self.history,
            "cv_scores": self.cv_scores
        }, f"{path}.pt")
    
    @classmethod
    def load(cls, path: str) -> "NeuralNetDirectionModel":
        """Load model from disk."""
        checkpoint = torch.load(f"{path}.pt", map_location="cpu")
        
        model = cls(checkpoint["config"])
        model.feature_names = checkpoint["feature_names"]
        model.history = checkpoint["history"]
        model.cv_scores = checkpoint.get("cv_scores")
        
        # Recreate model architecture
        input_dim = len(model.feature_names)
        model.model = DirectionNN(
            input_dim=input_dim,
            hidden_layers=model.config.get("hidden_layers", [128, 64, 32]),
            dropout=model.config.get("dropout", 0.3),
            activation=model.config.get("activation", "relu"),
            batch_norm=model.config.get("batch_norm", True)
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore scaler
        model.scaler.mean_ = checkpoint["scaler_mean"]
        model.scaler.scale_ = checkpoint["scaler_scale"]
        model.scaler.n_samples_seen_ = 1  # Dummy value
        
        return model

def train_neural_net_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    config: Dict = None
) -> NeuralNetDirectionModel:
    """Convenience function to train neural net model."""
    if config is None:
        config = {}
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    model = NeuralNetDirectionModel(config)
    model.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    return model
