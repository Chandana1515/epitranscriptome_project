#!/usr/bin/env python3
"""
Step 6: SHAP-Based Feature Interpretation
==========================================
Uses SHAP (SHapley Additive exPlanations) to interpret the trained ML model
and identify which biological features contribute to modification susceptibility.

Outputs:
  - Global feature importance (SHAP summary plot data)
  - Local explanations for individual sites
  - Feature dependence analysis
"""

import os
import sys
import json
import argparse
import logging
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging, save_dataframe


def load_model_and_data(config, model_name="xgboost"):
    """Load trained model and test data."""
    model_dir = config["output"]["model_dir"]
    features_dir = config["output"]["features_dir"]

    # Load model
    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load feature columns
    cols_path = os.path.join(features_dir, "feature_columns.txt")
    with open(cols_path, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    # Load test data
    test_path = os.path.join(model_dir, "test_data.csv")
    test_df = pd.read_csv(test_path)

    # Load full feature matrix for context
    matrix_path = os.path.join(features_dir, "feature_matrix.csv")
    if os.path.exists(matrix_path):
        full_df = pd.read_csv(matrix_path)
    else:
        full_df = None

    X_test = test_df[[c for c in feature_cols if c in test_df.columns]]
    y_test = test_df["y_true"] if "y_true" in test_df.columns else None

    return model, X_test, y_test, feature_cols, full_df


def compute_shap_values(model, X, model_name, logger):
    """
    Compute SHAP values for the given model and data.
    Uses TreeExplainer for tree-based models (fast, exact).
    """
    import shap

    logger.info(f"Computing SHAP values for {model_name}...")
    logger.info(f"  Data shape: {X.shape}")

    if model_name in ["xgboost", "random_forest"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X).values

    # For binary classification, shap_values might be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take class 1 (modified)

    logger.info(f"  SHAP values shape: {shap_values.shape}")
    return shap_values, explainer


def global_feature_importance(shap_values, feature_cols, output_dir, logger):
    """
    Compute and save global SHAP feature importance.
    Mean absolute SHAP value for each feature.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_cols[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Add rank
    importance_df["rank"] = range(1, len(importance_df) + 1)

    out_path = os.path.join(output_dir, "shap_global_importance.csv")
    importance_df.to_csv(out_path, index=False)

    logger.info("\nGlobal SHAP Feature Importance:")
    for _, row in importance_df.head(15).iterrows():
        logger.info(
            f"  {row['rank']:2d}. {row['feature']:40s} "
            f"SHAP={row['mean_abs_shap']:.6f}"
        )

    return importance_df


def shap_summary_data(shap_values, X, feature_cols, output_dir):
    """Save SHAP summary plot data for visualization."""
    summary_records = []
    for i, col in enumerate(feature_cols[:shap_values.shape[1]]):
        for j in range(len(X)):
            summary_records.append({
                "feature": col,
                "shap_value": shap_values[j, i],
                "feature_value": X.iloc[j, i] if i < X.shape[1] else np.nan,
            })

    summary_df = pd.DataFrame(summary_records)
    out_path = os.path.join(output_dir, "shap_summary_data.csv")
    summary_df.to_csv(out_path, index=False)
    return out_path


def local_explanations(shap_values, X, feature_cols, full_df,
                       output_dir, logger, top_n=20):
    """
    Generate local SHAP explanations for the most extreme predictions.
    Explains why specific sites were predicted as susceptible to change.
    """
    if full_df is None:
        logger.warning("Full feature matrix not available for local explanations")
        return

    # Find sites with highest positive SHAP sum (most likely to change)
    shap_sums = shap_values.sum(axis=1)
    top_indices = np.argsort(shap_sums)[-top_n:][::-1]

    local_records = []
    for rank, idx in enumerate(top_indices, 1):
        record = {"rank": rank, "sample_index": idx, "shap_sum": shap_sums[idx]}

        for i, col in enumerate(feature_cols[:shap_values.shape[1]]):
            record[f"{col}_value"] = X.iloc[idx, i] if i < X.shape[1] else np.nan
            record[f"{col}_shap"] = shap_values[idx, i]

        local_records.append(record)

    local_df = pd.DataFrame(local_records)
    out_path = os.path.join(output_dir, "shap_local_explanations_top.csv")
    local_df.to_csv(out_path, index=False)

    logger.info(f"\nTop {top_n} most susceptible sites saved to {out_path}")

    return local_df


def feature_dependence_analysis(shap_values, X, feature_cols,
                                 output_dir, logger, top_n=5):
    """
    Analyze SHAP dependence for top features.
    Shows how feature values relate to SHAP values.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[-top_n:][::-1]

    for feat_idx in top_features:
        if feat_idx >= len(feature_cols) or feat_idx >= X.shape[1]:
            continue

        feat_name = feature_cols[feat_idx]
        dep_df = pd.DataFrame({
            "feature_value": X.iloc[:, feat_idx],
            "shap_value": shap_values[:, feat_idx],
        })

        out_path = os.path.join(
            output_dir, f"shap_dependence_{feat_name}.csv"
        )
        dep_df.to_csv(out_path, index=False)

    logger.info(
        f"Feature dependence data saved for top {top_n} features"
    )


def generate_shap_plots(shap_values, X, feature_cols, output_dir, logger):
    """
    Generate SHAP visualization plots and save as PNG files.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # 1. Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_cols[:shap_values.shape[1]],
            plot_type="bar", show=False, max_display=20
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, "shap_summary_bar.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info("  Saved SHAP summary bar plot")

        # 2. Summary beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_cols[:shap_values.shape[1]],
            show=False, max_display=20
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, "shap_summary_beeswarm.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close()
        logger.info("  Saved SHAP beeswarm plot")

        # 3. Feature dependence plots for top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features = np.argsort(mean_abs_shap)[-5:][::-1]

        for feat_idx in top_features:
            if feat_idx >= len(feature_cols):
                continue
            feat_name = feature_cols[feat_idx]

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.dependence_plot(
                feat_idx, shap_values, X,
                feature_names=feature_cols[:shap_values.shape[1]],
                show=False, ax=ax
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(figures_dir, f"shap_dependence_{feat_name}.png"),
                dpi=150, bbox_inches="tight"
            )
            plt.close()

        logger.info("  Saved SHAP dependence plots")

    except Exception as e:
        logger.warning(f"Could not generate SHAP plots: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SHAP-based model interpretation"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model", default="xgboost",
        choices=["xgboost", "random_forest"],
        help="Which trained model to explain"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples for SHAP (for speed on large datasets)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["shap_dir"]
    logger = setup_logging(os.path.join(output_dir, "shap.log"))

    logger.info("=" * 60)
    logger.info("Step 6: SHAP-Based Feature Interpretation")
    logger.info("=" * 60)

    # Load model and data
    try:
        model, X_test, y_test, feature_cols, full_df = load_model_and_data(
            config, args.model
        )
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        sys.exit(1)

    logger.info(f"Model: {args.model}")
    logger.info(f"Test samples: {len(X_test)}")

    # Subsample if needed
    if args.max_samples and len(X_test) > args.max_samples:
        X_test = X_test.sample(args.max_samples, random_state=42)
        logger.info(f"Subsampled to {args.max_samples} for SHAP computation")

    # Compute SHAP values
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        sys.exit(1)

    shap_values, explainer = compute_shap_values(
        model, X_test, args.model, logger
    )

    # Save raw SHAP values
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_cols[:shap_values.shape[1]]
    )
    shap_path = os.path.join(output_dir, "shap_values.csv")
    shap_df.to_csv(shap_path, index=False)
    logger.info(f"SHAP values saved to {shap_path}")

    # Global importance
    global_feature_importance(shap_values, feature_cols, output_dir, logger)

    # Summary data for visualization
    shap_summary_data(shap_values, X_test, feature_cols, output_dir)

    # Local explanations
    local_explanations(
        shap_values, X_test, feature_cols, full_df, output_dir, logger
    )

    # Feature dependence
    feature_dependence_analysis(
        shap_values, X_test, feature_cols, output_dir, logger
    )

    # Generate plots
    generate_shap_plots(shap_values, X_test, feature_cols, output_dir, logger)

    logger.info("\nStep 6 complete.")


if __name__ == "__main__":
    main()
