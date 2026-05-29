#!/usr/bin/env python3
"""
Step 2: Site-Level Modification Stoichiometry
=============================================
Aggregates per-read modification probabilities to compute site-level
stoichiometry (fraction of modified reads at each position).

Features:
  - Filters to m6A only (configurable) to reduce data 3x
  - Processes chunks one at a time (memory-safe)
  - Checkpoints every 50 chunks so crashes don't lose progress
  - Resumes from last checkpoint automatically

Input:  Per-read modification chunks from Step 1
Output: Site-level stoichiometry tables per sample

Usage:
  python scripts/02_site_stoichiometry.py --config config.yaml
  
  # Run in background (survives terminal/screen disconnect):
  nohup python scripts/02_site_stoichiometry.py --config config.yaml > step2_log.txt 2>&1 &
  tail -f step2_log.txt
"""

import os
import sys
import argparse
import logging
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_config, setup_logging, load_dataframe, save_dataframe
)

# Column names matching the tuple order from Step 1's _flush_chunk
CHUNK_COLUMNS = [
    "sample", "read_id", "reference_name", "ref_pos",
    "mod_type", "mod_code", "probability", "strand"
]

# m6A modification codes from different Dorado models
M6A_CODES = {"m6A", "a"}


def find_sample_data(extraction_dir, sample_name):
    """Find extraction data files for a sample (single file or chunks)."""
    for ext in [".parquet", ".csv"]:
        single = os.path.join(extraction_dir, f"{sample_name}_modifications{ext}")
        if os.path.exists(single):
            return [single]

    chunks = sorted([
        os.path.join(extraction_dir, f)
        for f in os.listdir(extraction_dir)
        if f.startswith(f"{sample_name}_chunk_") and f.endswith(".parquet")
    ])
    return chunks


def normalize_chunk_columns(df):
    """Ensure chunk has proper named columns (handles numeric column names)."""
    col0 = str(df.columns[0])
    if col0 in ("sample", "read_id", "reference_name"):
        return df
    if col0.isdigit() or isinstance(df.columns[0], int):
        if len(df.columns) >= 8:
            df = df.iloc[:, :8]
            df.columns = CHUNK_COLUMNS
        else:
            raise ValueError(f"Chunk has {len(df.columns)} cols, need 8")
    return df


def process_chunk_to_site_counts(df, threshold, mod_filter=None):
    """
    Process a single chunk into per-site counts.
    
    Args:
        df: Raw chunk DataFrame
        threshold: Probability threshold for calling modified
        mod_filter: Set of mod_type values to keep (e.g. {"m6A"}).
                    None means keep all.
    """
    df = normalize_chunk_columns(df)

    if "probability" not in df.columns:
        raise ValueError(f"No 'probability' column. Columns: {df.columns.tolist()}")

    # Filter to specific modification type(s)
    if mod_filter and "mod_type" in df.columns:
        before = len(df)
        df = df[df["mod_type"].isin(mod_filter)]
        if df.empty:
            return pd.DataFrame()

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(0.0)
    df["is_modified"] = df["probability"] >= threshold

    group_cols = ["reference_name", "ref_pos", "mod_type", "strand"]
    available_cols = [c for c in group_cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame()

    site_counts = df.groupby(available_cols).agg(
        total_count=("is_modified", "count"),
        modified_count=("is_modified", "sum"),
        prob_sum=("probability", "sum"),
        prob_sq_sum=("probability", lambda x: (x ** 2).sum()),
    ).reset_index()

    return site_counts


def merge_site_counts(accumulated, new_counts):
    """Merge new site counts into accumulated totals."""
    if accumulated is None or accumulated.empty:
        return new_counts
    if new_counts is None or new_counts.empty:
        return accumulated

    group_cols = ["reference_name", "ref_pos", "mod_type", "strand"]
    available_cols = [c for c in group_cols if c in accumulated.columns]

    merged = pd.concat([accumulated, new_counts], ignore_index=True)
    result = merged.groupby(available_cols).agg(
        total_count=("total_count", "sum"),
        modified_count=("modified_count", "sum"),
        prob_sum=("prob_sum", "sum"),
        prob_sq_sum=("prob_sq_sum", "sum"),
    ).reset_index()

    return result


# ── Checkpoint helpers ──────────────────────────────────────

def get_checkpoint_path(output_dir, sample_name):
    return os.path.join(output_dir, f"{sample_name}_checkpoint.parquet")


def get_checkpoint_index_path(output_dir, sample_name):
    return os.path.join(output_dir, f"{sample_name}_checkpoint_idx.txt")


def save_checkpoint(accumulated, chunk_index, output_dir, sample_name, logger):
    """Save current accumulated counts and chunk index to disk."""
    if accumulated is None or accumulated.empty:
        return
    ckpt_path = get_checkpoint_path(output_dir, sample_name)
    idx_path = get_checkpoint_index_path(output_dir, sample_name)

    accumulated.to_parquet(ckpt_path, index=False)
    with open(idx_path, "w") as f:
        f.write(str(chunk_index))

    logger.info(
        f"  CHECKPOINT saved at chunk {chunk_index}: "
        f"{len(accumulated):,} sites -> {os.path.basename(ckpt_path)}"
    )


def load_checkpoint(output_dir, sample_name, logger):
    """
    Load checkpoint if it exists.
    Returns (accumulated_counts, start_chunk_index) or (None, 0).
    """
    ckpt_path = get_checkpoint_path(output_dir, sample_name)
    idx_path = get_checkpoint_index_path(output_dir, sample_name)

    if os.path.exists(ckpt_path) and os.path.exists(idx_path):
        try:
            accumulated = pd.read_parquet(ckpt_path)
            with open(idx_path) as f:
                start_idx = int(f.read().strip())
            logger.info(
                f"  RESUMING from checkpoint: chunk {start_idx}, "
                f"{len(accumulated):,} sites loaded"
            )
            return accumulated, start_idx
        except Exception as e:
            logger.warning(f"  Checkpoint corrupted, starting fresh: {e}")

    return None, 0


def clear_checkpoint(output_dir, sample_name):
    """Remove checkpoint files after successful completion."""
    for path in [
        get_checkpoint_path(output_dir, sample_name),
        get_checkpoint_index_path(output_dir, sample_name),
    ]:
        if os.path.exists(path):
            os.remove(path)


# ── Finalize ────────────────────────────────────────────────

def finalize_stoichiometry(site_counts, sample_name, min_coverage):
    """Convert accumulated site counts into final stoichiometry table."""
    df = site_counts.copy()

    df["total_reads"] = df["total_count"].astype(int)
    df["modified_reads"] = df["modified_count"].astype(int)
    df["stoichiometry"] = df["modified_reads"] / df["total_reads"]
    df["mean_probability"] = df["prob_sum"] / df["total_reads"]

    # Var = E[X^2] - (E[X])^2
    df["variance"] = (df["prob_sq_sum"] / df["total_reads"]) - (df["mean_probability"] ** 2)
    df["variance"] = df["variance"].clip(lower=0)
    df["std_probability"] = np.sqrt(df["variance"])
    df["median_probability"] = df["mean_probability"]  # approximation
    df["sample"] = sample_name

    before = len(df)
    df = df[df["total_reads"] >= min_coverage].copy()
    after = len(df)

    final_cols = [
        "sample", "reference_name", "ref_pos", "mod_type", "strand",
        "total_reads", "modified_reads", "stoichiometry",
        "mean_probability", "median_probability", "std_probability",
    ]
    df = df[[c for c in final_cols if c in df.columns]]

    return df, before, after


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute site-level modification stoichiometry"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument(
        "--mod-filter", default="m6A",
        help="Comma-separated mod types to keep (e.g. 'm6A' or 'm6A,inosine' or 'all')"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=50,
        help="Save checkpoint every N chunks (default: 50)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    extraction_dir = config["output"]["extraction_dir"]
    output_dir = config["output"]["stoichiometry_dir"]
    threshold_uint8 = config["thresholds"]["modification_probability"]
    threshold_float = threshold_uint8 / 255.0
    min_coverage = config["thresholds"]["min_coverage"]

    logger = setup_logging(os.path.join(output_dir, "stoichiometry.log"))

    # Parse modification filter
    if args.mod_filter.lower() == "all":
        mod_filter = None
        mod_filter_str = "all modifications"
    else:
        mod_filter = set(args.mod_filter.split(","))
        mod_filter_str = ", ".join(mod_filter)

    logger.info("=" * 60)
    logger.info("Step 2: Site-Level Modification Stoichiometry")
    logger.info("=" * 60)
    logger.info(f"Probability threshold: {threshold_float:.3f} ({threshold_uint8}/255)")
    logger.info(f"Minimum coverage: {min_coverage}")
    logger.info(f"Modification filter: {mod_filter_str}")
    logger.info(f"Checkpoint every: {args.checkpoint_every} chunks")

    samples = config["samples"]
    if args.sample:
        samples = {args.sample: samples[args.sample]}

    all_stoichiometry = []

    for sample_name in samples:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {sample_name}...")
        logger.info(f"{'='*50}")

        data_files = find_sample_data(extraction_dir, sample_name)
        if not data_files:
            logger.warning(f"No extraction data found for {sample_name}")
            continue

        logger.info(f"  Found {len(data_files)} data file(s)")

        # Try to resume from checkpoint
        accumulated_counts, start_idx = load_checkpoint(
            output_dir, sample_name, logger
        )

        total_records = 0
        total_kept = 0
        n_errors = 0

        for i, filepath in enumerate(tqdm(
            data_files, desc=f"  {sample_name}",
            initial=start_idx, total=len(data_files)
        )):
            # Skip already-processed chunks
            if i < start_idx:
                continue

            try:
                chunk = pd.read_parquet(filepath) if filepath.endswith(".parquet") else pd.read_csv(filepath)
                total_records += len(chunk)

                chunk_counts = process_chunk_to_site_counts(
                    chunk, threshold_float, mod_filter=mod_filter
                )
                kept = len(chunk_counts) if chunk_counts is not None else 0
                total_kept += kept
                del chunk

                if chunk_counts is not None and not chunk_counts.empty:
                    accumulated_counts = merge_site_counts(
                        accumulated_counts, chunk_counts
                    )
                del chunk_counts

                # Checkpoint
                if (i + 1) % args.checkpoint_every == 0:
                    gc.collect()
                    save_checkpoint(
                        accumulated_counts, i + 1,
                        output_dir, sample_name, logger
                    )
                    logger.info(
                        f"  Progress: {i+1}/{len(data_files)} chunks, "
                        f"{total_records:,} records read, "
                        f"{len(accumulated_counts):,} unique sites"
                    )

            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    logger.warning(f"  Error chunk {i} ({os.path.basename(filepath)}): {e}")
                elif n_errors == 4:
                    logger.warning(f"  (suppressing further errors)")
                continue

        if accumulated_counts is None or accumulated_counts.empty:
            logger.warning(f"  No site data for {sample_name}")
            continue

        logger.info(f"  Total records read:      {total_records:,}")
        logger.info(f"  Chunk errors:            {n_errors}")
        logger.info(f"  Unique sites (raw):      {len(accumulated_counts):,}")

        # Finalize
        site_df, n_before, n_after = finalize_stoichiometry(
            accumulated_counts, sample_name, min_coverage
        )
        del accumulated_counts
        gc.collect()

        logger.info(f"  Sites after filter (>={min_coverage}x): {n_after:,}")

        # Save
        out_path = os.path.join(output_dir, f"{sample_name}_stoichiometry.csv")
        save_dataframe(site_df, out_path)
        logger.info(f"  Saved to {out_path}")

        # Clean up checkpoint
        clear_checkpoint(output_dir, sample_name)
        logger.info(f"  Checkpoint cleared (complete)")

        # Per mod-type summary
        if "mod_type" in site_df.columns:
            for mt in sorted(site_df["mod_type"].unique()):
                sub = site_df[site_df["mod_type"] == mt]
                logger.info(
                    f"  {mt}: {len(sub):,} sites, "
                    f"mean stoich={sub['stoichiometry'].mean():.4f}, "
                    f"median cov={sub['total_reads'].median():.0f}"
                )

        all_stoichiometry.append(site_df)

    # Save combined
    if all_stoichiometry:
        combined = pd.concat(all_stoichiometry, ignore_index=True)
        combined_path = os.path.join(output_dir, "all_samples_stoichiometry.csv")
        save_dataframe(combined, combined_path)
        logger.info(f"\nCombined: {len(combined):,} sites")
        logger.info(f"Samples: {combined['sample'].unique().tolist()}")
        if "mod_type" in combined.columns:
            logger.info(f"Mod types: {combined['mod_type'].unique().tolist()}")

    logger.info("\nStep 2 complete.")


if __name__ == "__main__":
    main()