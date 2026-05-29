#!/usr/bin/env python3
"""
Step 3: Differential RNA Modification Analysis
===============================================
Compares modification stoichiometry between two conditions to identify
sites with significant changes.

Statistical methods:
  - Fisher's exact test (default, robust for count data)
  - Logistic regression (models probability as function of condition)
  - Beta-binomial regression (accounts for overdispersion)

Input:  Site-level stoichiometry tables from Step 2
Output: Differential modification results with p-values, FDR, effect sizes
"""

import os
import sys
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging, load_dataframe, save_dataframe


def merge_conditions(control_df, treatment_df, control_name, treatment_name):
    """
    Merge stoichiometry data from two conditions for comparison.
    Only keeps sites present in both conditions.
    """
    # Merge on site identity
    merge_cols = ["reference_name", "ref_pos", "mod_type", "strand"]

    control_cols = merge_cols + [
        "total_reads", "modified_reads", "stoichiometry",
        "mean_probability", "median_probability"
    ]
    treatment_cols = merge_cols + [
        "total_reads", "modified_reads", "stoichiometry",
        "mean_probability", "median_probability"
    ]

    ctrl = control_df[control_cols].copy()
    treat = treatment_df[treatment_cols].copy()

    merged = ctrl.merge(
        treat,
        on=merge_cols,
        suffixes=("_ctrl", "_treat"),
        how="inner"
    )

    merged["control"] = control_name
    merged["treatment"] = treatment_name

    return merged


def fishers_exact_test(row):
    """
    Perform Fisher's exact test for a single site.
    Compares modified/unmodified counts between conditions.
    """
    mod_ctrl = int(row["modified_reads_ctrl"])
    unmod_ctrl = int(row["total_reads_ctrl"] - row["modified_reads_ctrl"])
    mod_treat = int(row["modified_reads_treat"])
    unmod_treat = int(row["total_reads_treat"] - row["modified_reads_treat"])

    table = [[mod_ctrl, unmod_ctrl], [mod_treat, unmod_treat]]

    try:
        odds_ratio, pvalue = stats.fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio, pvalue = np.nan, 1.0

    return odds_ratio, pvalue


def logistic_regression_test(row):
    """
    Perform logistic regression test for a single site.
    Models modification probability as a function of condition.
    """
    try:
        import statsmodels.api as sm

        # Construct data: condition (0=ctrl, 1=treat) vs modified (0/1)
        mod_ctrl = int(row["modified_reads_ctrl"])
        total_ctrl = int(row["total_reads_ctrl"])
        mod_treat = int(row["modified_reads_treat"])
        total_treat = int(row["total_reads_treat"])

        y = np.array(
            [1] * mod_ctrl + [0] * (total_ctrl - mod_ctrl) +
            [1] * mod_treat + [0] * (total_treat - mod_treat)
        )
        x = np.array(
            [0] * total_ctrl + [1] * total_treat
        )

        X = sm.add_constant(x)
        model = sm.Logit(y, X)
        result = model.fit(disp=False, maxiter=100)

        coef = result.params[1]
        pvalue = result.pvalues[1]
        return coef, pvalue
    except Exception:
        return np.nan, 1.0


def run_differential_analysis(merged_df, method="fisher", logger=None):
    """
    Run differential analysis on all merged sites.

    Args:
        merged_df: Merged DataFrame from merge_conditions()
        method: "fisher" or "logistic"

    Returns:
        DataFrame with test results added
    """
    results = merged_df.copy()
    n_sites = len(results)

    if logger:
        logger.info(f"Testing {n_sites:,} sites using {method} method...")

    pvalues = []
    effect_sizes = []
    test_stats = []

    for idx, row in results.iterrows():
        if method == "fisher":
            stat, pval = fishers_exact_test(row)
        elif method == "logistic":
            stat, pval = logistic_regression_test(row)
        else:
            stat, pval = fishers_exact_test(row)

        test_stats.append(stat)
        pvalues.append(pval)

    results["test_statistic"] = test_stats
    results["pvalue"] = pvalues

    # Compute delta stoichiometry (effect size)
    results["delta_stoichiometry"] = (
        results["stoichiometry_treat"] - results["stoichiometry_ctrl"]
    )
    results["abs_delta_stoichiometry"] = results["delta_stoichiometry"].abs()

    # Multiple testing correction (Benjamini-Hochberg)
    valid_mask = results["pvalue"].notna()
    if valid_mask.sum() > 0:
        reject, fdr, _, _ = multipletests(
            results.loc[valid_mask, "pvalue"],
            method="fdr_bh",
            alpha=0.05
        )
        results.loc[valid_mask, "fdr"] = fdr
        results.loc[valid_mask, "significant_bh"] = reject
    else:
        results["fdr"] = np.nan
        results["significant_bh"] = False

    results["fdr"] = results["fdr"].fillna(1.0)
    results["significant_bh"] = results["significant_bh"].fillna(False)

    return results


def classify_significant_sites(results_df, fdr_threshold, min_delta):
    """
    Classify sites as significantly altered based on FDR and effect size.
    """
    results_df = results_df.copy()
    results_df["is_significant"] = (
        (results_df["fdr"] < fdr_threshold) &
        (results_df["abs_delta_stoichiometry"] >= min_delta)
    )

    # Direction of change
    results_df["direction"] = "unchanged"
    sig_mask = results_df["is_significant"]
    results_df.loc[
        sig_mask & (results_df["delta_stoichiometry"] > 0), "direction"
    ] = "increased"
    results_df.loc[
        sig_mask & (results_df["delta_stoichiometry"] < 0), "direction"
    ] = "decreased"

    return results_df


def generate_volcano_data(results_df, output_dir, comparison_name):
    """Save data formatted for volcano plot visualization."""
    volcano = results_df[
        ["reference_name", "ref_pos", "mod_type",
         "delta_stoichiometry", "pvalue", "fdr", "is_significant", "direction"]
    ].copy()
    volcano["-log10_pvalue"] = -np.log10(volcano["pvalue"].clip(lower=1e-300))
    volcano["-log10_fdr"] = -np.log10(volcano["fdr"].clip(lower=1e-300))

    out_path = os.path.join(output_dir, f"{comparison_name}_volcano_data.csv")
    volcano.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Differential RNA modification analysis"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--method", default="fisher",
        choices=["fisher", "logistic"],
        help="Statistical test method"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stoich_dir = config["output"]["stoichiometry_dir"]
    output_dir = config["output"]["differential_dir"]
    fdr_threshold = config["thresholds"]["fdr_threshold"]
    min_delta = config["thresholds"]["min_delta_stoichiometry"]

    logger = setup_logging(os.path.join(output_dir, "differential.log"))

    logger.info("=" * 60)
    logger.info("Step 3: Differential RNA Modification Analysis")
    logger.info("=" * 60)
    logger.info(f"Method: {args.method}")
    logger.info(f"FDR threshold: {fdr_threshold}")
    logger.info(f"Min delta stoichiometry: {min_delta}")

    comparisons = config.get("comparisons", [])
    if not comparisons:
        logger.error("No comparisons defined in config.yaml")
        sys.exit(1)

    all_results = []

    for comp in comparisons:
        ctrl_name = comp["control"]
        treat_name = comp["treatment"]
        comparison_name = f"{ctrl_name}_vs_{treat_name}"

        logger.info(f"\n--- Comparison: {comparison_name} ---")

        # Load stoichiometry data
        ctrl_path = os.path.join(stoich_dir, f"{ctrl_name}_stoichiometry.csv")
        treat_path = os.path.join(stoich_dir, f"{treat_name}_stoichiometry.csv")

        if not os.path.exists(ctrl_path):
            logger.error(f"Control data not found: {ctrl_path}")
            continue
        if not os.path.exists(treat_path):
            logger.error(f"Treatment data not found: {treat_path}")
            continue

        ctrl_df = load_dataframe(ctrl_path)
        treat_df = load_dataframe(treat_path)

        logger.info(f"  Control sites:   {len(ctrl_df):,}")
        logger.info(f"  Treatment sites: {len(treat_df):,}")

        # Merge conditions
        merged = merge_conditions(ctrl_df, treat_df, ctrl_name, treat_name)
        logger.info(f"  Common sites:    {len(merged):,}")

        if merged.empty:
            logger.warning("No common sites found. Skipping comparison.")
            continue

        # Run differential analysis per modification type
        for mod_type in merged["mod_type"].unique():
            logger.info(f"\n  Mod type: {mod_type}")
            subset = merged[merged["mod_type"] == mod_type].copy()

            if len(subset) < 2:
                logger.warning(f"  Too few sites for {mod_type}. Skipping.")
                continue

            # Run statistical tests
            results = run_differential_analysis(
                subset, method=args.method, logger=logger
            )

            # Classify significant sites
            results = classify_significant_sites(
                results, fdr_threshold, min_delta
            )

            n_sig = results["is_significant"].sum()
            n_up = (results["direction"] == "increased").sum()
            n_down = (results["direction"] == "decreased").sum()

            logger.info(f"  Total tested:    {len(results):,}")
            logger.info(f"  Significant:     {n_sig:,}")
            logger.info(f"    Increased:     {n_up:,}")
            logger.info(f"    Decreased:     {n_down:,}")

            # Save results
            comp_mod_name = f"{comparison_name}_{mod_type}"
            out_path = os.path.join(
                output_dir, f"{comp_mod_name}_differential.csv"
            )
            save_dataframe(results, out_path)
            logger.info(f"  Saved to {out_path}")

            # Save significant sites
            sig_df = results[results["is_significant"]].sort_values("fdr")
            sig_path = os.path.join(
                output_dir, f"{comp_mod_name}_significant_sites.csv"
            )
            save_dataframe(sig_df, sig_path)

            # Volcano plot data
            volcano_path = generate_volcano_data(
                results, output_dir, comp_mod_name
            )
            logger.info(f"  Volcano data: {volcano_path}")

            all_results.append(results)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(
            output_dir, "all_differential_results.csv"
        )
        save_dataframe(combined, combined_path)
        logger.info(f"\nCombined results: {combined_path}")

    logger.info("\nStep 3 complete.")


if __name__ == "__main__":
    main()
