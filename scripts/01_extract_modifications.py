#!/usr/bin/env python3
"""
Step 1: Extract Modification Information from BAM Files
========================================================
Reads modification-aware BAM files (with MM/ML tags from Dorado) and extracts
per-read modification probabilities at each genomic position.

Input:  BAM file with MM and ML tags
Output: CSV/Parquet table with columns:
        [sample, read_id, reference_name, ref_pos, mod_type, probability]
"""

import os
import sys
import argparse
import logging
import pysam
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_config, setup_logging, parse_mm_tag,
    get_modification_positions, assign_ml_probabilities,
    mod_code_to_name, save_dataframe, prob_to_float
)


def _save_incremental(records, sample_name, output_dir, logger):
    """Save records collected so far to a parquet file as backup."""
    try:
        backup_path = os.path.join(
            output_dir, f"{sample_name}_modifications_partial.parquet"
        )
        df = pd.DataFrame(records)
        df.to_parquet(backup_path, index=False)
        logger.info(f"  Incremental save: {len(records):,} records -> {backup_path}")
    except Exception as e:
        logger.warning(f"  Incremental save failed: {e}")


_CHUNK_COLUMNS = [
    "sample", "read_id", "reference_name", "ref_pos",
    "mod_type", "mod_code", "probability", "strand",
]


def _flush_chunk(records, sample_name, output_dir, chunk_num):
    """Write a chunk of records to a numbered parquet file and free memory."""
    if not records:
        return
    df = pd.DataFrame(records, columns=_CHUNK_COLUMNS)
    chunk_path = os.path.join(
        output_dir, f"{sample_name}_chunk_{chunk_num:04d}.parquet"
    )
    df.to_parquet(chunk_path, index=False)
    return chunk_path


def extract_modifications_pysam_native(bam_path, sample_name, config, logger):
    """
    Extract modifications using manual MM/ML tag parsing.
    Writes records in chunks to disk to avoid running out of memory.
    """
    logger.info(f"Extracting modifications from: {bam_path}")
    logger.info(f"Sample: {sample_name}")

    records = []
    n_reads = 0
    n_mapped_with_mods = 0
    n_unmapped = 0
    n_no_mm = 0
    n_errors = 0
    n_total_calls = 0
    chunk_num = 0
    output_dir = config["output"]["extraction_dir"]

    # Flush to disk every N reads — read from config, default 5000
    FLUSH_EVERY = int(config.get("processing", {}).get("chunk_size", 5000))

    # Clean up any previous chunk files
    for f in os.listdir(output_dir):
        if f.startswith(f"{sample_name}_chunk_") and f.endswith(".parquet"):
            os.remove(os.path.join(output_dir, f))

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in tqdm(bam.fetch(until_eof=True), desc=f"Processing {sample_name}"):
            n_reads += 1

            if read.is_unmapped:
                n_unmapped += 1
                continue

            if not read.has_tag("MM") or not read.has_tag("ML"):
                n_no_mm += 1
                continue

            try:
                read_id = read.query_name
                ref_name = read.reference_name
                query_seq = read.query_sequence

                if query_seq is None:
                    n_errors += 1
                    continue

                mm_string = read.get_tag("MM")
                ml_array = list(read.get_tag("ML"))

                mm_entries = parse_mm_tag(mm_string)
                mod_positions = get_modification_positions(read, mm_entries)
                mod_positions = assign_ml_probabilities(
                    mod_positions, ml_array, mm_entries
                )

                strand = "-" if read.is_reverse else "+"

                for mp in mod_positions:
                    records.append((
                        sample_name,
                        read_id,
                        ref_name,
                        mp["ref_pos"],
                        mod_code_to_name(mp["mod_code"]),
                        mp["mod_code"],
                        mp.get("probability", 0.0),
                        strand,
                    ))

                n_mapped_with_mods += 1

            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    logger.warning(f"  Skipped bad read: {e}")
                elif n_errors == 6:
                    logger.warning(f"  (suppressing further error messages)")
                continue

            # Flush chunk to disk to free memory
            if n_mapped_with_mods % FLUSH_EVERY == 0 and records:
                n_total_calls += len(records)
                _flush_chunk(records, sample_name, output_dir, chunk_num)
                chunk_num += 1
                logger.info(
                    f"  Chunk {chunk_num}: {n_reads:,} reads processed, "
                    f"{n_total_calls:,} total calls, "
                    f"{n_errors} errors"
                )
                records = []  # FREE MEMORY

    # Flush remaining records
    if records:
        n_total_calls += len(records)
        _flush_chunk(records, sample_name, output_dir, chunk_num)
        chunk_num += 1
        records = []

    logger.info(f"  Skipped {n_errors:,} malformed reads")
    logger.info(f"Finished processing {sample_name}:")
    logger.info(f"  Total reads:    {n_reads:,}")
    logger.info(f"  Mapped w/ mods: {n_mapped_with_mods:,}")
    logger.info(f"  Unmapped reads: {n_unmapped:,}")
    logger.info(f"  No MM/ML tags:  {n_no_mm:,}")
    logger.info(f"  Mod calls:      {n_total_calls:,}")
    logger.info(f"  Chunks written: {chunk_num}")

    # Don't merge chunks — 1.8 billion rows won't fit in RAM.
    # Step 2 will process chunks directly.
    logger.info(f"  Chunks saved to: {output_dir}/{sample_name}_chunk_*.parquet")
    return None  # Signal that data is in chunks, not a DataFrame


def extract_modifications_manual(bam_path, sample_name, config, logger):
    """
    Manual extraction using MM/ML tag parsing.
    Use this if pysam's modified_bases is not available.
    """
    logger.info(f"Using manual MM/ML parsing for: {bam_path}")

    records = []
    n_reads = 0

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in tqdm(bam.fetch(until_eof=True), desc=f"Processing {sample_name}"):
            n_reads += 1

            if read.is_unmapped:
                continue

            try:
                mm_string = read.get_tag("MM")
                ml_array = list(read.get_tag("ML"))
            except (KeyError, TypeError):
                continue

            read_id = read.query_name
            ref_name = read.reference_name

            mm_entries = parse_mm_tag(mm_string)
            mod_positions = get_modification_positions(read, mm_entries)
            mod_positions = assign_ml_probabilities(
                mod_positions, ml_array, mm_entries
            )

            for mp in mod_positions:
                records.append({
                    "sample": sample_name,
                    "read_id": read_id,
                    "reference_name": ref_name,
                    "ref_pos": mp["ref_pos"],
                    "mod_type": mod_code_to_name(mp["mod_code"]),
                    "mod_code": mp["mod_code"],
                    "probability": mp.get("probability", 0.0),
                    "strand": "-" if read.is_reverse else "+"
                })

    logger.info(f"Processed {n_reads:,} reads -> {len(records):,} mod calls")
    return pd.DataFrame(records)


def generate_summary(df, sample_name, output_dir, logger):
    """Generate extraction summary statistics."""
    if df.empty:
        logger.warning(f"No modification calls extracted for {sample_name}")
        return

    summary = {
        "sample": sample_name,
        "total_mod_calls": len(df),
        "unique_reads": df["read_id"].nunique(),
        "unique_sites": df.groupby(["reference_name", "ref_pos"]).ngroups,
        "unique_transcripts": df["reference_name"].nunique(),
    }

    # Per-modification-type summary
    for mod_type in df["mod_type"].unique():
        subset = df[df["mod_type"] == mod_type]
        summary[f"{mod_type}_calls"] = len(subset)
        summary[f"{mod_type}_mean_prob"] = subset["probability"].mean()
        summary[f"{mod_type}_median_prob"] = subset["probability"].median()

    summary_df = pd.DataFrame([summary])
    out_path = os.path.join(output_dir, f"{sample_name}_extraction_summary.csv")
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Summary saved to {out_path}")

    # Log key stats
    for key, val in summary.items():
        if key != "sample":
            logger.info(f"  {key}: {val}")


def generate_summary_from_chunks(chunk_paths, sample_name, output_dir, logger):
    """Aggregate extraction stats across all chunks without loading them all at once."""
    if not chunk_paths:
        logger.warning(f"No chunks found for {sample_name}")
        return

    total_calls = 0
    unique_reads = set()
    unique_sites = set()
    unique_transcripts = set()
    per_mod_counts = defaultdict(int)
    per_mod_prob_sum = defaultdict(float)
    per_mod_probs = defaultdict(list)  # for median (sampled)
    MAX_PROB_SAMPLES = 2_000_000  # cap memory for median calc

    for path in sorted(chunk_paths):
        df = pd.read_parquet(
            path,
            columns=["read_id", "reference_name", "ref_pos", "mod_type", "probability"],
        )
        total_calls += len(df)
        unique_reads.update(df["read_id"].unique())
        unique_transcripts.update(df["reference_name"].unique())
        unique_sites.update(
            map(tuple, df[["reference_name", "ref_pos"]].drop_duplicates().to_numpy())
        )
        for mod_type, sub in df.groupby("mod_type", sort=False):
            per_mod_counts[mod_type] += len(sub)
            per_mod_prob_sum[mod_type] += float(sub["probability"].sum())
            if len(per_mod_probs[mod_type]) < MAX_PROB_SAMPLES:
                per_mod_probs[mod_type].extend(
                    sub["probability"].to_numpy().tolist()[: MAX_PROB_SAMPLES - len(per_mod_probs[mod_type])]
                )
        del df

    summary = {
        "sample": sample_name,
        "total_mod_calls": total_calls,
        "unique_reads": len(unique_reads),
        "unique_sites": len(unique_sites),
        "unique_transcripts": len(unique_transcripts),
        "n_chunks": len(chunk_paths),
    }
    for mod_type, count in per_mod_counts.items():
        summary[f"{mod_type}_calls"] = count
        summary[f"{mod_type}_mean_prob"] = (
            per_mod_prob_sum[mod_type] / count if count else 0.0
        )
        summary[f"{mod_type}_median_prob"] = (
            float(np.median(per_mod_probs[mod_type])) if per_mod_probs[mod_type] else 0.0
        )

    summary_df = pd.DataFrame([summary])
    out_path = os.path.join(output_dir, f"{sample_name}_extraction_summary.csv")
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Summary saved to {out_path}")

    for key, val in summary.items():
        if key != "sample":
            logger.info(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract RNA modification data from BAM files"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--sample", default=None,
        help="Process only this sample (default: all samples)"
    )
    parser.add_argument(
        "--method", default="auto", choices=["auto", "native", "manual"],
        help="Extraction method: auto, native (pysam), or manual"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["extraction_dir"]
    logger = setup_logging(
        os.path.join(output_dir, "extraction.log")
    )

    logger.info("=" * 60)
    logger.info("Step 1: Extract Modification Information from BAM Files")
    logger.info("=" * 60)

    samples_to_process = config["samples"]
    if args.sample:
        if args.sample not in samples_to_process:
            logger.error(f"Sample '{args.sample}' not found in config")
            sys.exit(1)
        samples_to_process = {args.sample: samples_to_process[args.sample]}

    for sample_name, sample_info in samples_to_process.items():
        bam_path = sample_info["bam"]

        if not os.path.exists(bam_path):
            logger.error(f"BAM file not found: {bam_path}")
            continue

        # Check for index
        bai_path = bam_path + ".bai"
        if not os.path.exists(bai_path):
            alt_bai = bam_path.replace(".bam", ".bai")
            if not os.path.exists(alt_bai):
                logger.warning(
                    f"BAM index not found. Consider running: "
                    f"samtools index {bam_path}"
                )

        # Extract modifications
        if args.method == "manual":
            df = extract_modifications_manual(
                bam_path, sample_name, config, logger
            )
        else:
            df = extract_modifications_pysam_native(
                bam_path, sample_name, config, logger
            )

        # Check if data was extracted
        # df is None when chunks were written directly to disk (large files)
        # df is a DataFrame for smaller files processed by manual method
        if df is None:
            # Chunks on disk — verify they exist
            chunk_files = [
                f for f in os.listdir(output_dir)
                if f.startswith(f"{sample_name}_chunk_") and f.endswith(".parquet")
            ]
            if chunk_files:
                logger.info(f"Extraction complete: {len(chunk_files)} chunks on disk")
                chunk_paths = [os.path.join(output_dir, f) for f in chunk_files]
                generate_summary_from_chunks(
                    chunk_paths, sample_name, output_dir, logger
                )
            else:
                logger.warning(f"No data extracted for {sample_name}")
        elif df.empty:
            logger.warning(f"No data extracted for {sample_name}")
        else:
            # Small enough to fit in memory — save as single file
            out_path = os.path.join(output_dir, f"{sample_name}_modifications.parquet")
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved extraction to {out_path}")
            generate_summary(df, sample_name, output_dir, logger)
            del df

    logger.info("Step 1 complete.")


if __name__ == "__main__":
    main()