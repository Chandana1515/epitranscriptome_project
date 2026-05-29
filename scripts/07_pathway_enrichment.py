#!/usr/bin/env python3
"""
Step 7: Pathway Enrichment Analysis
====================================
Performs Gene Ontology (GO) and KEGG pathway enrichment analysis on genes
associated with significantly altered modification sites.

Uses gseapy (Gene Set Enrichment Analysis in Python) for enrichment testing.
Falls back to scipy-based Fisher's exact test if gseapy is unavailable.

Input:  Differential analysis results (significant sites)
Output: Enriched pathways, GO terms, visualization data
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging, load_dataframe, save_dataframe


def get_significant_genes(config, logger):
    """
    Extract gene names from significantly altered modification sites.
    """
    diff_dir = config["output"]["differential_dir"]
    features_dir = config["output"]["features_dir"]

    # Try to load from feature matrix (has gene names)
    matrix_path = os.path.join(features_dir, "feature_matrix.csv")
    if os.path.exists(matrix_path):
        df = load_dataframe(matrix_path)
        if "is_significant" in df.columns and "gene_name" in df.columns:
            sig_genes = df[df["is_significant"] == True]["gene_name"].unique()
            all_genes = df["gene_name"].unique()
            logger.info(f"Significant genes: {len(sig_genes)}")
            logger.info(f"Background genes:  {len(all_genes)}")
            return list(sig_genes), list(all_genes), df

    # Fallback: load from differential results
    all_path = os.path.join(diff_dir, "all_differential_results.csv")
    if os.path.exists(all_path):
        df = load_dataframe(all_path)
        # Extract gene names from reference_name
        df["gene_name"] = df["reference_name"].apply(
            lambda x: str(x).split("|")[5] if "|" in str(x) and len(str(x).split("|")) > 5 else str(x)
        )
        if "is_significant" in df.columns:
            sig_genes = df[df["is_significant"] == True]["gene_name"].unique()
            all_genes = df["gene_name"].unique()
            return list(sig_genes), list(all_genes), df

    logger.warning("Could not find significant genes")
    return [], [], pd.DataFrame()


def run_gseapy_enrichment(sig_genes, all_genes, output_dir, logger):
    """
    Run enrichment analysis using gseapy library.
    Tests against GO_Biological_Process, GO_Molecular_Function, KEGG.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.warning(
            "gseapy not installed. Run: pip install gseapy\n"
            "Falling back to curated pathway analysis."
        )
        return None

    results = {}
    gene_sets = [
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "KEGG_2021_Human",
        "Reactome_2022",
    ]

    for gs_name in gene_sets:
        logger.info(f"  Testing against: {gs_name}")
        try:
            enr = gp.enrichr(
                gene_list=list(sig_genes),
                gene_sets=gs_name,
                organism="Human",
                outdir=os.path.join(output_dir, gs_name),
                cutoff=0.05,
                no_plot=True,
            )
            if enr.results is not None and len(enr.results) > 0:
                results[gs_name] = enr.results
                n_sig = (enr.results["Adjusted P-value"] < 0.05).sum()
                logger.info(f"    Found {n_sig} significant terms")
            else:
                logger.info(f"    No significant terms found")
        except Exception as e:
            logger.warning(f"    Failed for {gs_name}: {e}")

    return results


def curated_pathway_enrichment(sig_genes, all_genes, logger):
    """
    Perform enrichment analysis using curated pathway gene sets.
    Uses Fisher's exact test for each pathway.
    """
    # Curated gene sets relevant to plasma biology and skin cancer
    pathway_sets = {
        "Oxidative_Stress_Response": {
            "NFE2L2", "NRF2", "SOD1", "SOD2", "CAT", "GPX1", "GPX4",
            "HMOX1", "NQO1", "TXNRD1", "PRDX1", "PRDX2", "PRDX3",
            "GSR", "GCLC", "GCLM", "TXN", "TXNRD2", "SRXN1", "KEAP1"
        },
        "Apoptosis": {
            "TP53", "BAX", "BCL2", "BCL2L1", "CASP3", "CASP7", "CASP8",
            "CASP9", "APAF1", "BID", "BAK1", "MCL1", "BBC3", "PMAIP1",
            "CYCS", "DIABLO", "XIAP", "BIRC5", "FAS", "FASLG", "TRAIL"
        },
        "DNA_Damage_Response": {
            "ATM", "ATR", "BRCA1", "BRCA2", "RAD51", "XRCC1", "XRCC4",
            "PARP1", "H2AFX", "MDM2", "CHEK1", "CHEK2", "TP53BP1",
            "RPA1", "RAD50", "MRE11", "NBN", "GADD45A", "GADD45B"
        },
        "Cell_Cycle": {
            "CDKN1A", "CDKN2A", "CDKN1B", "CDK2", "CDK4", "CDK6",
            "CCND1", "CCNE1", "CCNA2", "CCNB1", "RB1", "E2F1", "E2F2",
            "MYC", "CDC25A", "CDC25C", "PLK1", "AURKA", "AURKB"
        },
        "Immune_Regulation": {
            "CD274", "PDCD1LG2", "IFNG", "IFNB1", "TNF", "IL6", "IL1B",
            "CXCL8", "STAT1", "STAT3", "JAK1", "JAK2", "IRF1", "IRF3",
            "NFKB1", "RELA", "CXCL10", "CCL2", "IL10", "TGFB1"
        },
        "m6A_RNA_Modification": {
            "METTL3", "METTL14", "WTAP", "RBM15", "RBM15B", "VIRMA",
            "ZC3H13", "FTO", "ALKBH5", "YTHDF1", "YTHDF2", "YTHDF3",
            "YTHDC1", "YTHDC2", "IGF2BP1", "IGF2BP2", "IGF2BP3",
            "HNRNPA2B1", "HNRNPC"
        },
        "Melanoma_Signaling": {
            "BRAF", "NRAS", "KIT", "MITF", "PTEN", "CDKN2A", "AKT1",
            "MAPK1", "MAPK3", "RAF1", "MEK1", "ERK2", "SOX10",
            "MC1R", "GNAQ", "GNA11", "NF1", "RAS"
        },
        "Hypoxia_Response": {
            "HIF1A", "EPAS1", "ARNT", "VHL", "VEGFA", "VEGFB",
            "EPO", "LDHA", "PDK1", "SLC2A1", "SLC2A3", "CA9",
            "ADM", "BNIP3", "BNIP3L", "EGLN1", "EGLN2", "EGLN3"
        },
    }

    sig_gene_set = set(str(g).upper() for g in sig_genes)
    all_gene_set = set(str(g).upper() for g in all_genes)
    n_total = len(all_gene_set)
    n_sig = len(sig_gene_set)

    results = []
    for pathway_name, pathway_genes in pathway_sets.items():
        # Count overlaps
        pathway_in_background = pathway_genes & all_gene_set
        n_pathway = len(pathway_in_background)

        if n_pathway == 0:
            continue

        overlap = sig_gene_set & pathway_in_background
        n_overlap = len(overlap)

        # Fisher's exact test
        # Contingency table:
        #              In pathway    Not in pathway
        # Significant  n_overlap     n_sig - n_overlap
        # Not sig      n_pathway-n_overlap  n_total-n_sig-n_pathway+n_overlap
        a = n_overlap
        b = n_sig - n_overlap
        c = n_pathway - n_overlap
        d = n_total - n_sig - n_pathway + n_overlap

        table = [[a, b], [c, d]]
        try:
            odds_ratio, pvalue = stats.fisher_exact(table, alternative="greater")
        except Exception:
            odds_ratio, pvalue = 1.0, 1.0

        results.append({
            "pathway": pathway_name,
            "pathway_size": n_pathway,
            "overlap_count": n_overlap,
            "overlap_genes": "; ".join(sorted(overlap)) if overlap else "",
            "expected": n_sig * n_pathway / max(n_total, 1),
            "fold_enrichment": (n_overlap / max(n_sig, 1)) / (n_pathway / max(n_total, 1)) if n_pathway > 0 else 0,
            "odds_ratio": odds_ratio,
            "pvalue": pvalue,
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Multiple testing correction
        reject, fdr, _, _ = multipletests(
            results_df["pvalue"], method="fdr_bh", alpha=0.05
        )
        results_df["fdr"] = fdr
        results_df["significant"] = reject
        results_df = results_df.sort_values("pvalue")

    return results_df


def generate_enrichment_summary(results, output_dir, logger):
    """Generate summary of enrichment results."""
    logger.info("\n--- Pathway Enrichment Summary ---")

    if isinstance(results, dict):
        # gseapy results
        for gs_name, result_df in results.items():
            sig = result_df[result_df["Adjusted P-value"] < 0.05]
            logger.info(f"\n{gs_name}: {len(sig)} significant terms")
            for _, row in sig.head(10).iterrows():
                logger.info(
                    f"  {row['Term']}: "
                    f"p={row['Adjusted P-value']:.4e}, "
                    f"genes={row.get('Genes', 'N/A')}"
                )

            # Save
            out_path = os.path.join(output_dir, f"enrichment_{gs_name}.csv")
            result_df.to_csv(out_path, index=False)

    elif isinstance(results, pd.DataFrame):
        # Curated pathway results
        for _, row in results.iterrows():
            status = "*" if row.get("significant", False) else " "
            logger.info(
                f"  {status} {row['pathway']:30s} "
                f"overlap={row['overlap_count']}/{row['pathway_size']} "
                f"fold={row['fold_enrichment']:.2f} "
                f"p={row['pvalue']:.4e} "
                f"FDR={row.get('fdr', 1.0):.4e}"
            )

        out_path = os.path.join(output_dir, "curated_pathway_enrichment.csv")
        results.to_csv(out_path, index=False)
        logger.info(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pathway enrichment analysis for plasma-responsive genes"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--method", default="both",
        choices=["gseapy", "curated", "both"],
        help="Enrichment method"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["pathway_dir"]
    logger = setup_logging(os.path.join(output_dir, "pathway.log"))

    logger.info("=" * 60)
    logger.info("Step 7: Pathway Enrichment Analysis")
    logger.info("=" * 60)

    # Get significant genes
    sig_genes, all_genes, df = get_significant_genes(config, logger)

    if not sig_genes:
        logger.warning(
            "No significant genes found. "
            "This may be expected if comparing similar conditions."
        )
        # Still run curated analysis to show the framework
        logger.info("Running curated pathway analysis with all genes as demo...")
        if all_genes:
            # Use top genes by effect size as "significant" for demo
            if "abs_delta_stoichiometry" in df.columns:
                top_genes = df.nlargest(
                    min(100, len(df)), "abs_delta_stoichiometry"
                )
                if "gene_name" in top_genes.columns:
                    sig_genes = top_genes["gene_name"].unique().tolist()

    # Run curated pathway enrichment (always available)
    if args.method in ["curated", "both"]:
        logger.info("\n--- Curated Pathway Enrichment ---")
        curated_results = curated_pathway_enrichment(
            sig_genes, all_genes, logger
        )
        generate_enrichment_summary(curated_results, output_dir, logger)

    # Run gseapy enrichment
    if args.method in ["gseapy", "both"]:
        logger.info("\n--- Gene Set Enrichment (gseapy) ---")
        gseapy_results = run_gseapy_enrichment(
            sig_genes, all_genes, output_dir, logger
        )
        if gseapy_results:
            generate_enrichment_summary(gseapy_results, output_dir, logger)

    logger.info("\nStep 7 complete.")


if __name__ == "__main__":
    main()
