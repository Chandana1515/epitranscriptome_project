#!/usr/bin/env python3
"""
Step 5: Train Explainable Machine Learning Model
=================================================
Trains XGBoost and/or Random Forest classifiers to predict which sites
are susceptible to modification change. The model learns which biological
features explain differential modification.

Input:  Feature matrix from Step 4
Output: Trained model, performance metrics, feature importances
"""

import os
import sys
import json
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging, load_dataframe, save_dataframe


def load_feature_matrix(config):
    """Load feature matrix and column names."""
    features_dir = config["output"]["features_dir"]
    matrix_path = os.path.join(features_dir, "feature_matrix.csv")
    cols_path = os.path.join(features_dir, "feature_columns.txt")

    df = load_dataframe(matrix_path)

    with open(cols_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Ensure all feature columns exist
    available = [c for c in feature_cols if c in df.columns]
    return df, available


def prepare_data(df, feature_cols, config, logger):
    """Prepare X, y for training."""
    X = df[feature_cols].copy()
    y = df["label"].copy()

    # Handle class imbalance info
    n_pos = y.sum()
    n_neg = (y == 0).sum()
    logger.info(f"Class distribution: positive={n_pos}, negative={n_neg}")
    logger.info(f"Imbalance ratio: {n_neg / max(n_pos, 1):.1f}:1")

    # Train/test split
    test_size = config["ml"]["test_size"]
    random_state = config["ml"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set:  {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train, config, logger):
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed. Run: pip install xgboost")
        return None

    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=config["ml"]["n_estimators"],
        max_depth=config["ml"]["max_depth"],
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=config["ml"]["random_state"],
        eval_metric="logloss",
        use_label_encoder=False,
    )

    logger.info("Training XGBoost classifier...")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, config, logger):
    """Train Random Forest classifier."""
    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()

    model = RandomForestClassifier(
        n_estimators=config["ml"]["n_estimators"],
        max_depth=config["ml"]["max_depth"],
        class_weight="balanced",
        random_state=config["ml"]["random_state"],
        n_jobs=config["processing"]["n_jobs"],
    )

    logger.info("Training Random Forest classifier...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name, feature_cols,
                   output_dir, logger):
    """Evaluate model and save metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"\n{model_name} Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        logger.info(f"ROC AUC: {roc_auc:.4f}")
    except ValueError:
        roc_auc = np.nan

    # Average Precision
    try:
        avg_prec = average_precision_score(y_test, y_proba)
        logger.info(f"Average Precision: {avg_prec:.4f}")
    except ValueError:
        avg_prec = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Feature importances (built-in)
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    logger.info(f"\nTop 10 Feature Importances ({model_name}):")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save metrics
    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "avg_precision": float(avg_prec) if not np.isnan(avg_prec) else None,
        "accuracy": report["accuracy"],
        "precision_1": report.get("1", {}).get("precision", 0),
        "recall_1": report.get("1", {}).get("recall", 0),
        "f1_1": report.get("1", {}).get("f1-score", 0),
    }

    # Save outputs
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    importance_path = os.path.join(output_dir, f"{model_name}_importances.csv")
    importance_df.to_csv(importance_path, index=False)

    # Save ROC curve data
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        roc_df.to_csv(
            os.path.join(output_dir, f"{model_name}_roc_curve.csv"),
            index=False
        )
    except Exception:
        pass

    # Save PR curve data
    try:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_df = pd.DataFrame({"precision": prec, "recall": rec})
        pr_df.to_csv(
            os.path.join(output_dir, f"{model_name}_pr_curve.csv"),
            index=False
        )
    except Exception:
        pass

    return metrics, importance_df


def cross_validate_model(model_class, X, y, feature_cols, config, logger,
                         model_name="model", **model_kwargs):
    """Perform stratified k-fold cross-validation."""
    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=config["ml"]["random_state"]
    )

    model = model_class(**model_kwargs)

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(
        f"{model_name} 5-fold CV ROC AUC: "
        f"{scores.mean():.4f} +/- {scores.std():.4f}"
    )
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Train explainable ML models for modification prediction"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["model_dir"]
    logger = setup_logging(os.path.join(output_dir, "model_training.log"))

    logger.info("=" * 60)
    logger.info("Step 5: Train Explainable Machine Learning Model")
    logger.info("=" * 60)

    # Load data
    df, feature_cols = load_feature_matrix(config)
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")

    if df["label"].nunique() < 2:
        logger.error(
            "Need both positive and negative samples for training. "
            "Check if differential analysis found significant sites."
        )
        sys.exit(1)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        df, feature_cols, config, logger
    )

    model_type = config["ml"]["model_type"]
    trained_models = {}

    # Train XGBoost
    if model_type in ["xgboost", "both"]:
        xgb_model = train_xgboost(X_train, y_train, config, logger)
        if xgb_model:
            metrics, importances = evaluate_model(
                xgb_model, X_test, y_test, "xgboost", feature_cols,
                output_dir, logger
            )
            trained_models["xgboost"] = xgb_model

            # Save model
            model_path = os.path.join(output_dir, "xgboost_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(xgb_model, f)
            logger.info(f"XGBoost model saved to {model_path}")

    # Train Random Forest
    if model_type in ["random_forest", "both"]:
        rf_model = train_random_forest(X_train, y_train, config, logger)
        metrics, importances = evaluate_model(
            rf_model, X_test, y_test, "random_forest", feature_cols,
            output_dir, logger
        )
        trained_models["random_forest"] = rf_model

        # Save model
        model_path = os.path.join(output_dir, "random_forest_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(rf_model, f)
        logger.info(f"Random Forest model saved to {model_path}")

    # Cross-validation
    logger.info("\n--- Cross-Validation ---")
    X_full = df[feature_cols]
    y_full = df["label"]

    if "xgboost" in trained_models:
        try:
            import xgboost as xgb
            n_pos = y_full.sum()
            n_neg = (y_full == 0).sum()
            cross_validate_model(
                xgb.XGBClassifier, X_full, y_full, feature_cols, config,
                logger, model_name="XGBoost",
                n_estimators=config["ml"]["n_estimators"],
                max_depth=config["ml"]["max_depth"],
                scale_pos_weight=n_neg / max(n_pos, 1),
                eval_metric="logloss", use_label_encoder=False,
                random_state=config["ml"]["random_state"]
            )
        except Exception as e:
            logger.warning(f"XGBoost CV failed: {e}")

    if "random_forest" in trained_models:
        cross_validate_model(
            RandomForestClassifier, X_full, y_full, feature_cols, config,
            logger, model_name="RandomForest",
            n_estimators=config["ml"]["n_estimators"],
            max_depth=config["ml"]["max_depth"],
            class_weight="balanced",
            random_state=config["ml"]["random_state"]
        )

    # Save test data for SHAP analysis
    test_data = pd.DataFrame(X_test, columns=feature_cols)
    test_data["y_true"] = y_test.values
    test_path = os.path.join(output_dir, "test_data.csv")
    test_data.to_csv(test_path, index=False)

    train_data = pd.DataFrame(X_train, columns=feature_cols)
    train_data["y_true"] = y_train.values
    train_path = os.path.join(output_dir, "train_data.csv")
    train_data.to_csv(train_path, index=False)

    logger.info("\nStep 5 complete.")


if __name__ == "__main__":
    main()
