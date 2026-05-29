#!/usr/bin/env python3
"""
Step 4: Biological Feature Construction
========================================
Builds a feature matrix for each modification site by extracting biological
annotations from sequence context, transcript structure, and gene metadata.

Features constructed:
  - baseline_stoichiometry (from control condition)
  - transcript_region (5'UTR / CDS / 3'UTR)
  - GC_content (local sequence context)
  - DRACH_motif (canonical m6A motif presence)
  - gene_length
  - relative_position (position within transcript)
  - sequence_context (k-mer around the site)
  - coverage_control / coverage_treatment

Input:  Differential analysis results from Step 3
        (Optional) GTF annotation, reference FASTA
Output: Feature matrix for machine learning
"""

import os
import sys
import re
import argparse
import logging
import numpy as np
import pandas as pd
import pyranges as pr
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging, load_dataframe, save_dataframe


# ============================================================================
# Feature extraction functions
# ============================================================================

def extract_transcript_features(df):
    """
    Extract transcript-level features from reference_name.
    For transcriptome-aligned data, the reference_name often encodes
    transcript/gene information.
    """
    features = df.copy()

    # Parse transcript info from reference_name if ENST format
    features["transcript_id"] = features["reference_name"].apply(
        lambda x: x.split("|")[0] if "|" in str(x) else str(x)
    )

    # Extract gene name if present in reference name (GENCODE format)
    # Format: ENST...|ENSG...|OTTHUMG...|OTTHUMT...|GENE_NAME-N|GENE_NAME|...
    def parse_gene_name(ref_name):
        parts = str(ref_name).split("|")
        if len(parts) >= 6:
            return parts[5]  # Gene name in GENCODE format
        elif len(parts) >= 2:
            return parts[1]
        return ref_name

    features["gene_name"] = features["reference_name"]

    return features


def compute_gc_content_from_context(sequence, window=50):
    """Compute GC content of a sequence window."""
    if not sequence or len(sequence) == 0:
        return np.nan
    seq = sequence.upper()
    gc = sum(1 for b in seq if b in "GC")
    return gc / len(seq)


def check_drach_motif(sequence, position=None):
    """
    Check if the site falls within a DRACH motif.
    DRACH = [D=A/G/U][R=A/G][A][C][H=A/C/U]
    For RNA: D={A,G,U}, R={A,G}, A, C, H={A,C,U}
    In DNA terms: D={A,G,T}, R={A,G}, A, C, H={A,C,T}
    """
    if not sequence:
        return False

    seq = sequence.upper().replace("U", "T")
    drach_pattern = r"[AGT][AG]AC[ACT]"
    return bool(re.search(drach_pattern, seq))


def compute_sequence_features(df, reference_fasta=None):
    """
    Compute sequence-context features.
    If reference FASTA is available, extract actual sequences.
    Otherwise, use placeholder features.
    """
    features = df.copy()

    if reference_fasta and os.path.exists(reference_fasta):
        try:
            import pysam
            fasta = pysam.FastaFile(reference_fasta)

            gc_contents = []
            drach_motifs = []
            kmer_contexts = []

            for _, row in features.iterrows():
                ref = row["reference_name"]
                pos = int(row["ref_pos"])

                try:
                    ref_len = fasta.get_reference_length(ref)

                    # Extract 50bp window for GC content
                    start = max(0, pos - 25)
                    end = min(ref_len, pos + 25)
                    seq_window = fasta.fetch(ref, start, end)
                    gc_contents.append(compute_gc_content_from_context(seq_window))

                    # Extract 5bp window for DRACH check
                    start_d = max(0, pos - 2)
                    end_d = min(ref_len, pos + 3)
                    seq_drach = fasta.fetch(ref, start_d, end_d)
                    drach_motifs.append(check_drach_motif(seq_drach))

                    # Extract 11-mer context
                    start_k = max(0, pos - 5)
                    end_k = min(ref_len, pos + 6)
                    kmer = fasta.fetch(ref, start_k, end_k)
                    kmer_contexts.append(kmer.upper())

                except (ValueError, KeyError):
                    gc_contents.append(np.nan)
                    drach_motifs.append(False)
                    kmer_contexts.append("")

            features["gc_content"] = gc_contents
            features["drach_motif"] = drach_motifs
            features["kmer_context"] = kmer_contexts

            fasta.close()

        except ImportError:
            logging.warning("pysam not available; skipping sequence features")
            features["gc_content"] = np.nan
            features["drach_motif"] = False
            features["kmer_context"] = ""
    else:
        # Placeholder features when no reference is available
        features["gc_content"] = np.nan
        features["drach_motif"] = False
        features["kmer_context"] = ""

    return features


def compute_positional_features(df, reference_fasta=None):
    """
    Compute position-related features.
    If GTF annotations already added relative_position and transcript_region,
    this function just fills any gaps.
    """
    if "relative_position" not in df.columns:
        df["relative_position"] = 0.5
 
    if "transcript_region" not in df.columns:
        df["transcript_region"] = "unknown"
 
    return df


def load_gene_annotation(gtf_path, logger=None):
    """
    Load a GTF file and build a gene-level annotation table using pyranges.
    Returns a pyranges object with gene boundaries and metadata.
    """
    if not gtf_path or not os.path.exists(gtf_path):
        if logger:
            logger.warning(f"GTF not found: {gtf_path}")
        return None
 
    if logger:
        logger.info(f"  Loading GTF: {gtf_path}")
 
    try:
        gtf = pr.read_gtf(gtf_path)
    except Exception as e:
        if logger:
            logger.warning(f"  Failed to read GTF with pyranges: {e}")
        return None
 
    # Filter to gene features only (not transcripts, exons, etc.)
    genes = gtf[gtf.Feature == "gene"]
 
    if logger:
        logger.info(f"  Loaded {len(genes)} gene records from GTF")
 
    return genes
 
 
def add_gtf_annotations(df, gtf_path, logger=None):
    """
    Map genomic coordinates to gene names using pyranges interval overlap.
    
    For each site (chr, position), find which gene it falls within and
    annotate with gene_name, gene_biotype, gene_length, and relative position
    within the gene.
    """
    if not gtf_path or not os.path.exists(gtf_path):
        df["gene_name"] = df["reference_name"]
        df["gene_biotype"] = "unknown"
        df["gene_length"] = np.nan
        return df
 
    genes = load_gene_annotation(gtf_path, logger)
    if genes is None:
        df["gene_name"] = df["reference_name"]
        df["gene_biotype"] = "unknown"
        df["gene_length"] = np.nan
        return df
 
    # Build a pyranges object from our sites
    sites_pr = pr.PyRanges(pd.DataFrame({
        "Chromosome": df["reference_name"].values,
        "Start": df["ref_pos"].astype(int).values,
        "End": (df["ref_pos"].astype(int) + 1).values,
        "idx": range(len(df)),
    }))
 
    if logger:
        logger.info(f"  Overlapping {len(df):,} sites with {len(genes):,} genes...")
 
    # Find overlaps
    try:
        overlaps = sites_pr.join(genes, how="left")
        overlap_df = overlaps.df
    except Exception as e:
        if logger:
            logger.warning(f"  Overlap failed: {e}")
        df["gene_name"] = df["reference_name"]
        df["gene_biotype"] = "unknown"
        df["gene_length"] = np.nan
        return df
 
    if logger:
        logger.info(f"  Found {len(overlap_df):,} overlaps")
 
    # For sites mapping to multiple genes, keep the one with smallest gene
    # (most specific annotation)
    if "gene_name" in overlap_df.columns:
        overlap_df["_gene_length"] = overlap_df["End_b"] - overlap_df["Start_b"]
 
        # Deduplicate: keep shortest gene per site
        overlap_df = overlap_df.sort_values("_gene_length")
        overlap_df = overlap_df.drop_duplicates(subset=["idx"], keep="first")
 
        # Build mapping
        idx_to_gene = overlap_df.set_index("idx")
 
        df["gene_name"] = df.index.map(
            lambda i: idx_to_gene.loc[i, "gene_name"] 
            if i in idx_to_gene.index else df.loc[i, "reference_name"]
        ) if len(idx_to_gene) > 0 else df["reference_name"]
 
        # Gene biotype
        biotype_col = None
        for col in ["gene_type", "gene_biotype"]:
            if col in idx_to_gene.columns:
                biotype_col = col
                break
 
        if biotype_col:
            df["gene_biotype"] = df.index.map(
                lambda i: idx_to_gene.loc[i, biotype_col]
                if i in idx_to_gene.index else "unknown"
            )
        else:
            df["gene_biotype"] = "unknown"
 
        # Gene length
        df["gene_length"] = df.index.map(
            lambda i: idx_to_gene.loc[i, "_gene_length"]
            if i in idx_to_gene.index else np.nan
        )
 
        # Relative position within gene (0 = start, 1 = end)
        def _get_rel_pos(i):
            if i not in idx_to_gene.index:
                return 0.5
            row = idx_to_gene.loc[i]
            gene_start = row["Start_b"]
            gene_end = row["End_b"]
            pos = df.loc[i, "ref_pos"]
            length = gene_end - gene_start
            if length <= 0:
                return 0.5
            return (pos - gene_start) / length
 
        df["relative_position"] = [_get_rel_pos(i) for i in df.index]
 
        # Estimate transcript region from relative position
        df["transcript_region"] = pd.cut(
            df["relative_position"],
            bins=[0, 0.15, 0.85, 1.0],
            labels=["5UTR", "CDS", "3UTR"],
            include_lowest=True,
        ).astype(str).fillna("unknown")
 
        n_annotated = (df["gene_name"] != df["reference_name"]).sum()
        if logger:
            logger.info(f"  Annotated {n_annotated:,} / {len(df):,} sites with gene names")
            logger.info(f"  Gene biotypes: {df['gene_biotype'].value_counts().head(5).to_dict()}")
    else:
        if logger:
            logger.warning("  No gene_name column in GTF overlap results")
        df["gene_name"] = df["reference_name"]
        df["gene_biotype"] = "unknown"
        df["gene_length"] = np.nan
 
    return df


def add_pathway_annotations(df):
    """
    Add curated pathway annotations for known cancer-related genes.
    This serves as a starting feature set; expand with GO/KEGG enrichment later.
    """
    # Curated pathway associations for common skin cancer / stress genes
    pathway_map = {
        # Oxidative stress response
        "NFE2L2": "oxidative_stress", "NRF2": "oxidative_stress",
        "SOD1": "oxidative_stress", "SOD2": "oxidative_stress",
        "CAT": "oxidative_stress", "GPX1": "oxidative_stress",
        "GPX4": "oxidative_stress", "HMOX1": "oxidative_stress",
        "NQO1": "oxidative_stress", "TXNRD1": "oxidative_stress",
        "PRDX1": "oxidative_stress", "GSR": "oxidative_stress",

        # Apoptosis
        "TP53": "apoptosis", "BAX": "apoptosis", "BCL2": "apoptosis",
        "CASP3": "apoptosis", "CASP8": "apoptosis", "CASP9": "apoptosis",
        "APAF1": "apoptosis", "BID": "apoptosis", "BAK1": "apoptosis",
        "MCL1": "apoptosis", "PUMA": "apoptosis", "NOXA": "apoptosis",
        "CYCS": "apoptosis", "DIABLO": "apoptosis",

        # DNA damage response
        "ATM": "dna_damage", "ATR": "dna_damage", "BRCA1": "dna_damage",
        "BRCA2": "dna_damage", "RAD51": "dna_damage", "XRCC1": "dna_damage",
        "PARP1": "dna_damage", "H2AFX": "dna_damage", "MDM2": "dna_damage",
        "CHEK1": "dna_damage", "CHEK2": "dna_damage",

        # Cell cycle
        "CDKN1A": "cell_cycle", "CDKN2A": "cell_cycle",
        "CDK2": "cell_cycle", "CDK4": "cell_cycle",
        "CCND1": "cell_cycle", "CCNE1": "cell_cycle",
        "RB1": "cell_cycle", "E2F1": "cell_cycle",

        # Immune regulation
        "CD274": "immune", "PDCD1LG2": "immune",
        "IFNG": "immune", "TNF": "immune",
        "IL6": "immune", "IL1B": "immune",
        "CXCL8": "immune", "STAT1": "immune",
        "STAT3": "immune", "JAK2": "immune",

        # m6A machinery
        "METTL3": "m6a_machinery", "METTL14": "m6a_machinery",
        "WTAP": "m6a_machinery", "FTO": "m6a_machinery",
        "ALKBH5": "m6a_machinery", "YTHDF1": "m6a_machinery",
        "YTHDF2": "m6a_machinery", "YTHDF3": "m6a_machinery",
        "YTHDC1": "m6a_machinery", "YTHDC2": "m6a_machinery",

        # Melanoma / skin cancer specific
        "BRAF": "melanoma_driver", "NRAS": "melanoma_driver",
        "KIT": "melanoma_driver", "MITF": "melanoma_driver",
        "PTEN": "melanoma_driver", "CDKN2A": "melanoma_driver",
    }

    df["pathway_annotation"] = df["gene_name"].map(
        lambda g: pathway_map.get(str(g).upper(), "other")
    )

    return df


def build_feature_matrix(df, config, logger):
    """
    Build the complete feature matrix from differential analysis results.
    """
    logger.info("Building feature matrix...")

    # Step 1: Transcript features
    logger.info("  Extracting transcript features...")
    df = extract_transcript_features(df)

    # Step 2: Sequence features
    ref_fasta = config.get("reference_fasta")
    logger.info("  Computing sequence features...")
    df = compute_sequence_features(df, ref_fasta)

    # Step 3: Positional features
    logger.info("  Computing positional features...")
    df = compute_positional_features(df, ref_fasta)

    # Step 4: GTF annotations
    gtf_path = config.get("gtf_annotation")
    logger.info("  Adding GTF annotations...")
    df = add_gtf_annotations(df, gtf_path, logger=logger)

    # Step 5: Pathway annotations
    logger.info("  Adding pathway annotations...")
    df = add_pathway_annotations(df)

    # Step 6: Coverage features
    df["log_coverage_ctrl"] = np.log1p(df["total_reads_ctrl"])
    df["log_coverage_treat"] = np.log1p(df["total_reads_treat"])
    df["coverage_ratio"] = (
        df["total_reads_treat"] / df["total_reads_ctrl"].clip(lower=1)
    )

    # Step 7: Stoichiometry features
    df["baseline_stoichiometry"] = df["stoichiometry_ctrl"]
    df["stoichiometry_change_ratio"] = (
        df["stoichiometry_treat"] / df["stoichiometry_ctrl"].clip(lower=0.001)
    )

    return df


def prepare_ml_features(df, logger):
    """
    Prepare features for machine learning:
    - Encode categorical features
    - Handle missing values
    - Select feature columns
    """
    feature_df = df.copy()

    # Encode categorical features
    categorical_cols = [
        "transcript_region", "pathway_annotation", "gene_biotype"
    ]
    for col in categorical_cols:
        if col in feature_df.columns:
            dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=False)
            feature_df = pd.concat([feature_df, dummies], axis=1)

    # Boolean to int
    if "drach_motif" in feature_df.columns:
        feature_df["drach_motif"] = feature_df["drach_motif"].astype(int)

    # Define numeric feature columns for ML
    numeric_features = [
        "baseline_stoichiometry",
        "gc_content",
        "drach_motif",
        "relative_position",
        "gene_length",
        "log_coverage_ctrl",
        "log_coverage_treat",
        "coverage_ratio",
        "mean_probability_ctrl",
        "mean_probability_treat",
    ]

    # Add encoded categorical features
    for col in feature_df.columns:
        if any(col.startswith(f"{cat}_") for cat in categorical_cols):
            numeric_features.append(col)

    # Keep only available features
    available_features = [f for f in numeric_features if f in feature_df.columns]
    logger.info(f"  Available features: {len(available_features)}")
    for f in available_features:
        logger.info(f"    - {f}")

    # Handle missing values
    for col in available_features:
        if feature_df[col].isna().any():
            median_val = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(median_val)
            logger.info(f"    Filled NaN in {col} with median={median_val:.4f}")

    return feature_df, available_features


def main():
    parser = argparse.ArgumentParser(
        description="Build biological feature matrix for ML"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    diff_dir = config["output"]["differential_dir"]
    output_dir = config["output"]["features_dir"]

    logger = setup_logging(os.path.join(output_dir, "features.log"))

    logger.info("=" * 60)
    logger.info("Step 4: Biological Feature Construction")
    logger.info("=" * 60)

    # Load differential analysis results
    all_diff_path = os.path.join(diff_dir, "all_differential_results.csv")

    if not os.path.exists(all_diff_path):
        # Try to find individual comparison files
        diff_files = [
            f for f in os.listdir(diff_dir)
            if f.endswith("_differential.csv") and not f.startswith("all_")
        ]
        if diff_files:
            dfs = [load_dataframe(os.path.join(diff_dir, f)) for f in diff_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            logger.error("No differential analysis results found!")
            sys.exit(1)
    else:
        df = load_dataframe(all_diff_path)

    logger.info(f"Loaded {len(df):,} sites from differential analysis")

    # Build feature matrix
    feature_df = build_feature_matrix(df, config, logger)

    # Prepare ML-ready features
    feature_df, feature_cols = prepare_ml_features(feature_df, logger)

    # Define target variable
    feature_df["label"] = feature_df["is_significant"].astype(int)

    # Save complete feature matrix
    out_path = os.path.join(output_dir, "feature_matrix.csv")
    save_dataframe(feature_df, out_path)
    logger.info(f"\nFeature matrix saved to {out_path}")
    logger.info(f"  Total sites: {len(feature_df):,}")
    logger.info(f"  Positive (significant): {feature_df['label'].sum():,}")
    logger.info(
        f"  Negative (non-significant): "
        f"{(feature_df['label'] == 0).sum():,}"
    )

    # Save feature column names for ML step
    feature_cols_path = os.path.join(output_dir, "feature_columns.txt")
    with open(feature_cols_path, "w") as f:
        f.write("\n".join(feature_cols))
    logger.info(f"  Feature columns saved to {feature_cols_path}")

    logger.info("\nStep 4 complete.")


if __name__ == "__main__":
    main()
