"""
Result Loader
=============
Loads pipeline output files into DataFrames and summary dicts
for the dashboard and AI chat context.
"""

import os
import json
import yaml
import pandas as pd
import streamlit as st


def find_results_dir(pipeline_dir: str) -> str:
    """Locate the results directory from runtime_config, falling back to ./results/."""
    runtime = os.path.join(pipeline_dir, "runtime_config.yaml")
    if os.path.exists(runtime):
        try:
            with open(runtime) as f:
                cfg = yaml.safe_load(f)
            base = cfg.get("output", {}).get("base_dir", "")
            if base:
                if not os.path.isabs(base):
                    base = os.path.join(pipeline_dir, base)
                if os.path.isdir(base):
                    return base
        except Exception:
            pass
    results_dir = os.path.join(pipeline_dir, "results")
    if os.path.isdir(results_dir):
        return results_dir
    return pipeline_dir


def load_csv_safe(path: str) -> pd.DataFrame:
    """Load a CSV, returning empty DataFrame on failure."""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        parquet = path.replace(".csv", ".parquet")
        if os.path.exists(parquet):
            return pd.read_parquet(parquet)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
    return pd.DataFrame()


def load_json_safe(path: str) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


@st.cache_data(ttl=300)
def load_all_results(results_dir: str) -> dict:
    """
    Load all pipeline results into a structured dict.
    Pass the results directory directly (e.g. results_A375/).
    Cached for 5 minutes to avoid re-reading on every interaction.
    """
    results = {}

    # ── Extraction summaries ────────────────────────────────
    ext_dir = os.path.join(results_dir, "01_extraction")
    if os.path.isdir(ext_dir):
        summaries = []
        for f in os.listdir(ext_dir):
            if f.endswith("_extraction_summary.csv"):
                summaries.append(load_csv_safe(os.path.join(ext_dir, f)))
        if summaries:
            results["extraction_summary"] = pd.concat(summaries, ignore_index=True)

    # ── Stoichiometry ───────────────────────────────────────
    stoich_dir = os.path.join(results_dir, "02_stoichiometry")
    if os.path.isdir(stoich_dir):
        combined = os.path.join(stoich_dir, "all_samples_stoichiometry.csv")
        results["stoichiometry"] = load_csv_safe(combined)

        # Per-sample stoichiometry
        results["stoichiometry_per_sample"] = {}
        for f in os.listdir(stoich_dir):
            if f.endswith("_stoichiometry.csv") and not f.startswith("all_"):
                sample = f.replace("_stoichiometry.csv", "")
                results["stoichiometry_per_sample"][sample] = load_csv_safe(
                    os.path.join(stoich_dir, f)
                )

        # Summary stats
        if not results["stoichiometry"].empty:
            df = results["stoichiometry"]
            results["stoichiometry_summary"] = {
                "total_sites": len(df),
                "unique_transcripts": df["reference_name"].nunique() if "reference_name" in df.columns else 0,
                "mean_stoichiometry": round(df["stoichiometry"].mean(), 4) if "stoichiometry" in df.columns else 0,
                "median_coverage": round(df["total_reads"].median(), 1) if "total_reads" in df.columns else 0,
            }

    # ── Differential analysis ───────────────────────────────
    diff_dir = os.path.join(results_dir, "03_differential")
    if os.path.isdir(diff_dir):
        all_diff = os.path.join(diff_dir, "all_differential_results.csv")
        results["differential"] = load_csv_safe(all_diff)

        # Significant sites
        if not results["differential"].empty and "is_significant" in results["differential"].columns:
            sig = results["differential"][results["differential"]["is_significant"] == True]
            results["significant_sites"] = sig.sort_values("fdr") if "fdr" in sig.columns else sig
        else:
            results["significant_sites"] = pd.DataFrame()

        # Volcano data
        for f in os.listdir(diff_dir):
            if f.endswith("_volcano_data.csv"):
                results["volcano_data"] = load_csv_safe(os.path.join(diff_dir, f))
                break

    # ── Feature matrix ──────────────────────────────────────
    feat_dir = os.path.join(results_dir, "04_features")
    if os.path.isdir(feat_dir):
        results["feature_matrix"] = load_csv_safe(
            os.path.join(feat_dir, "feature_matrix.csv")
        )

    # ── Model metrics ───────────────────────────────────────
    model_dir = os.path.join(results_dir, "05_model")
    if os.path.isdir(model_dir):
        for model_name in ["xgboost", "random_forest"]:
            metrics = load_json_safe(
                os.path.join(model_dir, f"{model_name}_metrics.json")
            )
            if metrics:
                results["model_metrics"] = metrics

            imp = load_csv_safe(
                os.path.join(model_dir, f"{model_name}_importances.csv")
            )
            if not imp.empty:
                results["feature_importances"] = imp

            roc = load_csv_safe(
                os.path.join(model_dir, f"{model_name}_roc_curve.csv")
            )
            if not roc.empty:
                results["roc_curve"] = roc

    # ── SHAP ────────────────────────────────────────────────
    shap_dir = os.path.join(results_dir, "06_shap")
    if os.path.isdir(shap_dir):
        results["shap_importance"] = load_csv_safe(
            os.path.join(shap_dir, "shap_global_importance.csv")
        )
        results["shap_values"] = load_csv_safe(
            os.path.join(shap_dir, "shap_values.csv")
        )

    # ── Pathway enrichment ──────────────────────────────────
    path_dir = os.path.join(results_dir, "07_pathway")
    if os.path.isdir(path_dir):
        curated = os.path.join(path_dir, "curated_pathway_enrichment.csv")
        results["pathway_enrichment"] = load_csv_safe(curated)

    # ── Metatranscript ──────────────────────────────────────
    meta_dir = os.path.join(results_dir, "08_metatranscript")
    if os.path.isdir(meta_dir):
        results["metatranscript"] = load_csv_safe(
            os.path.join(meta_dir, "metatranscript_positions.csv")
        )
        results["metatranscript_summary"] = load_csv_safe(
            os.path.join(meta_dir, "metatranscript_summary.csv")
        )

    # ── Build overall summary ───────────────────────────────
    summary = {}
    if "differential" in results and not results["differential"].empty:
        summary["total_tested_sites"] = len(results["differential"])
    if "significant_sites" in results:
        summary["significant_sites"] = len(results["significant_sites"])
    if "model_metrics" in results:
        summary["model_roc_auc"] = results["model_metrics"].get("roc_auc", "N/A")
        summary["model_type"] = results["model_metrics"].get("model", "N/A")
    if "pathway_enrichment" in results and not results["pathway_enrichment"].empty:
        pe = results["pathway_enrichment"]
        if "significant" in pe.columns:
            summary["enriched_pathways"] = pe["significant"].sum()
    results["summary"] = summary

    return results


def get_results_overview(results: dict) -> dict:
    """Get a quick overview of what results are available."""
    overview = {}
    for key in [
        "extraction_summary", "stoichiometry", "differential",
        "significant_sites", "feature_matrix", "model_metrics",
        "shap_importance", "pathway_enrichment"
    ]:
        if key in results:
            val = results[key]
            if isinstance(val, pd.DataFrame):
                overview[key] = f"{len(val)} rows" if not val.empty else "empty"
            elif isinstance(val, dict):
                overview[key] = "loaded"
            else:
                overview[key] = "loaded"
        else:
            overview[key] = "not found"
    return overview
