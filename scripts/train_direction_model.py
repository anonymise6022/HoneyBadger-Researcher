#!/usr/bin/env python3
"""
Train direction classification models (XGBoost, Neural Net, Ensemble).
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v1.data.loaders import load_ohlcv, load_config, merge_data_sources
from v1.features.pipeline import build_features, prepare_train_val_test, get_feature_columns
from v1.models.direction.xgboost_classifier import XGBoostDirectionModel, train_xgboost_model
from v1.models.direction.neural_net_classifier import NeuralNetDirectionModel, train_neural_net_model
from v1.models.direction.ensemble import DirectionEnsemble, create_ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train direction models")
    parser.add_argument("--data", type=str, required=True, help="Path to OHLCV data")
    parser.add_argument("--config", type=str, default="configs/direction_model.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="models/direction", help="Output directory")
    parser.add_argument("--model", type=str, default="ensemble", 
                       choices=["xgboost", "neural_net", "ensemble"], help="Model type")
    parser.add_argument("--start", type=str, help="Start date")
    parser.add_argument("--end", type=str, help="End date")
    parser.add_argument("--oi", type=str, help="Path to open interest data")
    parser.add_argument("--options", type=str, help="Path to options data")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load data
    logger.info(f"Loading OHLCV data from {args.data}")
    ohlcv = load_ohlcv(args.data, start=args.start, end=args.end)
    
    # Load additional data if provided
    oi = None
    if args.oi:
        logger.info(f"Loading OI data from {args.oi}")
        oi = load_ohlcv(args.oi)  # Reuse loader
    
    options = None
    if args.options:
        logger.info(f"Loading options data from {args.options}")
        from v1.data.loaders import load_options_chain
        options = load_options_chain(args.options)
    
    # Merge data
    df = merge_data_sources(ohlcv, oi, options)
    logger.info(f"Merged data shape: {df.shape}")
    
    # Build features
    logger.info("Building features...")
    features_df = build_features(
        df,
        config,
        target_horizon=config.get("target", {}).get("horizon", 4),
        threshold_pct=config.get("target", {}).get("threshold_pct", 0.015),
        labeling_method=config.get("target", {}).get("labeling_method", "terciles"),
        is_training=True
    )
    logger.info(f"Features shape: {features_df.shape}")
    
    # Get feature columns
    feature_cols = get_feature_columns(features_df)
    logger.info(f"Number of features: {len(feature_cols)}")
    
    # Split data
    train_frac = config.get("training", {}).get("train_frac", 0.7)
    val_frac = config.get("training", {}).get("val_frac", 0.15)
    purge_gap = config.get("training", {}).get("purge_gap", 24)
    embargo_pct = config.get("training", {}).get("embargo_pct", 0.01)
    
    train_df, val_df, test_df = prepare_train_val_test(
        features_df,
        train_frac=train_frac,
        val_frac=val_frac,
        purge_gap=purge_gap,
        embargo_pct=embargo_pct
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train models
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model in ["xgboost", "ensemble"]:
        logger.info("Training XGBoost model...")
        xgb_model = train_xgboost_model(
            train_df, val_df, feature_cols, "target", config.get("xgboost", {})
        )
        xgb_model.save(str(output_dir / "xgboost_model"))
        logger.info("XGBoost model saved")
    
    if args.model in ["neural_net", "ensemble"]:
        logger.info("Training Neural Net model...")
        nn_model = train_neural_net_model(
            train_df, val_df, feature_cols, "target", config.get("neural_net", {})
        )
        nn_model.save(str(output_dir / "neural_net_model"))
        logger.info("Neural Net model saved")
    
    if args.model == "ensemble":
        logger.info("Creating ensemble...")
        # Load models if not already in memory
        if "xgb_model" not in locals():
            xgb_model = XGBoostDirectionModel.load(str(output_dir / "xgboost_model"))
        if "nn_model" not in locals():
            nn_model = NeuralNetDirectionModel.load(str(output_dir / "neural_net_model"))
        
        ensemble = create_ensemble(xgb_model, nn_model, config.get("ensemble", {}))
        
        # Fit stacking if configured
        if config.get("ensemble", {}).get("method") == "stacking":
            logger.info("Fitting stacking meta-learner...")
            ensemble.fit_stacking(
                train_df[feature_cols], train_df["target"],
                val_df[feature_cols], val_df["target"],
                meta_learner_type=config.get("ensemble", {}).get("meta_learner", "logistic")
            )
        
        # Evaluate on test set
        logger.info("Evaluating ensemble on test set...")
        test_metrics = ensemble.evaluate(test_df[feature_cols], test_df["target"])
        logger.info(f"Test LogLoss: {test_metrics['log_loss']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Per-class accuracy: {test_metrics['per_class_accuracy']}")
        
        # Save ensemble config
        ensemble_config = {
            "method": ensemble.method,
            "xgb_weight": ensemble.xgb_weight,
            "nn_weight": ensemble.nn_weight,
            "agreement_threshold": ensemble.agreement_threshold,
            "meta_learner_type": config.get("ensemble", {}).get("meta_learner", "logistic")
        }
        with open(output_dir / "ensemble_config.yaml", "w") as f:
            yaml.dump(ensemble_config, f)
        
        logger.info("Ensemble saved")

if __name__ == "__main__":
    main()
