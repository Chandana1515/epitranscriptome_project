#!/usr/bin/env python3
"""
Step 8: Metatranscript Position Analysis
=========================================
Maps differential modification sites onto a normalised transcript coordinate
(5'UTR → CDS → 3'UTR) to reveal whether modifications preferentially occur
at the beginning or end of transcripts — regions linked to distinct functions.

Output:
  results/08_metatranscript/metatranscript_positions.csv
  results/08_metatranscript/metatranscript_summary.csv
"""

import os
import sys
import re
import argparse
import logging
import bisect
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging


# ── GTF parsing ────────────────────────────────────────────────────────────────

WANTED_FEATURES = {"five_prime_utr", "three_prime_utr", "CDS", "transcript"}
_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def _parse_attrs(attr_str):
    return {k: v for k, v in _ATTR_RE.findall(attr_str)}


def parse_gtf(gtf_path, logger):
    """Read GTF and return a DataFrame of relevant features only."""
    logger.info(f"Parsing GTF: {gtf_path}")
    records = []
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in WANTED_FEATURES:
                continue
            attrs = _parse_attrs(parts[8])
            records.append({
                "chrom":         parts[0],
                "start":         int(parts[3]) - 1,   # → 0-based half-open
                "end":           int(parts[4]),
                "strand":        parts[6],
                "feature":       feature,
                "transcript_id": attrs.get("transcript_id", ""),
                "gene_name":     attrs.get("gene_name", ""),
            })
    df = pd.DataFrame(records)
    logger.info(f"  Loaded {len(df):,} feature records")
    return df


# ── Interval index ─────────────────────────────────────────────────────────────

def build_index(gtf_df):
    """Build a per-(chrom, strand) sorted interval list for fast lookup."""
    index = defaultdict(list)
    for row in gtf_df.itertuples(index=False):
        key = (row.chrom, row.strand)
        index[key].append((row.start, row.end, row.feature,
                           row.transcript_id, row.gene_name))
    for key in index:
        index[key].sort()
    return index


def query_interval(index, chrom, pos, strand):
    """Return all features overlapping (chrom, pos, strand)."""
    key = (chrom, strand)
    intervals = index.get(key, [])
    if not intervals:
        return []
    # Binary search: find first interval that could overlap pos
    starts = [iv[0] for iv in intervals]
    i = bisect.bisect_right(starts, pos) - 1
    hits = []
    while i >= 0 and intervals[i][1] > pos:
        hits.append(intervals[i])
        i -= 1
    # Also scan forward from bisect point (overlaps that start ≤ pos)
    # already covered; also scan right for same-start intervals
    j = bisect.bisect_right(starts, pos)
    while j < len(intervals) and intervals[j][0] <= pos:
        if intervals[j][1] > pos:
            hits.append(intervals[j])
        j += 1
    return hits


# ── Transcript model per transcript_id ────────────────────────────────────────

def build_transcript_models(gtf_df):
    """
    For each transcript_id build a model:
      { transcript_id: { chrom, strand, tx_start, tx_end,
                         utr5_regions, cds_regions, utr3_regions } }
    """
    models = {}
    tx_df = gtf_df[gtf_df["feature"] == "transcript"]
    for row in tx_df.itertuples(index=False):
        models[row.transcript_id] = {
            "chrom":       row.chrom,
            "strand":      row.strand,
            "gene_name":   row.gene_name,
            "tx_start":    row.start,
            "tx_end":      row.end,
            "utr5":        [],
            "cds":         [],
            "utr3":        [],
        }

    for row in gtf_df[gtf_df["feature"] != "transcript"].itertuples(index=False):
        tid = row.transcript_id
        if tid not in models:
            continue
        if row.feature == "five_prime_utr":
            models[tid]["utr5"].append((row.start, row.end))
        elif row.feature == "CDS":
            models[tid]["cds"].append((row.start, row.end))
        elif row.feature == "three_prime_utr":
            models[tid]["utr3"].append((row.start, row.end))

    return models


def _total_len(regions):
    return sum(e - s for s, e in regions)


def metatranscript_pos(pos, tx_model):
    """
    Map a genomic position to a normalised metatranscript coordinate [0, 3].
      0–1 : 5'UTR   (position within 5'UTR, normalised)
      1–2 : CDS
      2–3 : 3'UTR
    Returns (metatranscript_pos, region_label) or (None, None) if unmappable.
    """
    strand   = tx_model["strand"]
    utr5     = sorted(tx_model["utr5"])
    cds      = sorted(tx_model["cds"])
    utr3     = sorted(tx_model["utr3"])

    utr5_len = _total_len(utr5)
    cds_len  = _total_len(cds)
    utr3_len = _total_len(utr3)

    def offset_in_regions(regions, genomic_pos, reverse):
        """Bases from the 5' end of the region set to genomic_pos."""
        sorted_r = sorted(regions, reverse=reverse)
        acc = 0
        for s, e in sorted_r:
            if reverse:
                # minus strand: iterate high→low
                if genomic_pos >= s and genomic_pos < e:
                    return acc + (e - 1 - genomic_pos)
                acc += (e - s)
            else:
                if genomic_pos >= s and genomic_pos < e:
                    return acc + (genomic_pos - s)
                acc += (e - s)
        return None

    reverse = (strand == "-")

    # Try 5'UTR
    if utr5 and utr5_len > 0:
        off = offset_in_regions(utr5, pos, reverse)
        if off is not None:
            return off / utr5_len, "5UTR"

    # Try CDS
    if cds and cds_len > 0:
        off = offset_in_regions(cds, pos, reverse)
        if off is not None:
            return 1.0 + off / cds_len, "CDS"

    # Try 3'UTR
    if utr3 and utr3_len > 0:
        off = offset_in_regions(utr3, pos, reverse)
        if off is not None:
            return 2.0 + off / utr3_len, "3UTR"

    return None, None


# ── Main annotation logic ──────────────────────────────────────────────────────

def annotate_sites(sites_df, index, tx_models, logger):
    """
    Annotate each site with its metatranscript position.
    Picks the first transcript that gives a valid region assignment.
    """
    meta_pos_list  = []
    region_list    = []
    gene_name_list = []
    tx_id_list     = []

    for _, row in sites_df.iterrows():
        chrom  = str(row["reference_name"])
        pos    = int(row["ref_pos"])
        strand = str(row.get("strand", "+"))

        hits = query_interval(index, chrom, pos, strand)

        # Filter to transcript-level hits to get transcript_ids
        tx_hits = [h for h in hits if h[2] == "transcript"]

        assigned = False
        for h in tx_hits:
            tid = h[3]
            gene = h[4]
            model = tx_models.get(tid)
            if model is None:
                continue
            mpos, region = metatranscript_pos(pos, model)
            if mpos is not None:
                meta_pos_list.append(mpos)
                region_list.append(region)
                gene_name_list.append(gene)
                tx_id_list.append(tid)
                assigned = True
                break

        if not assigned:
            meta_pos_list.append(np.nan)
            region_list.append("intergenic")
            gene_name_list.append("")
            tx_id_list.append("")

    sites_df = sites_df.copy()
    sites_df["metatranscript_pos"] = meta_pos_list
    sites_df["region"]             = region_list
    sites_df["gene_name"]          = gene_name_list
    sites_df["transcript_id"]      = tx_id_list

    assigned_n = sum(1 for r in region_list if r != "intergenic")
    logger.info(f"  Annotated {assigned_n:,} / {len(sites_df):,} sites to transcript regions")
    return sites_df


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Metatranscript position analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config     = load_config(args.config)
    output_dir = os.path.join(
        config["output"].get("base_dir", "./results"),
        "08_metatranscript",
    )
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(os.path.join(output_dir, "metatranscript.log"))
    logger.info("=" * 60)
    logger.info("Step 8: Metatranscript Position Analysis")
    logger.info("=" * 60)

    # ── GTF ───────────────────────────────────────────────────
    gtf_path = config.get("gtf_annotation")
    if not gtf_path or not os.path.exists(gtf_path):
        logger.error("gtf_annotation not set or file missing — cannot run step 8")
        sys.exit(1)

    gtf_df   = parse_gtf(gtf_path, logger)
    index    = build_index(gtf_df)
    tx_models = build_transcript_models(gtf_df)
    logger.info(f"  Transcript models: {len(tx_models):,}")

    # ── Differential sites ────────────────────────────────────
    diff_dir = config["output"]["differential_dir"]
    diff_path = os.path.join(diff_dir, "all_differential_results.csv")
    if not os.path.exists(diff_path):
        logger.error(f"Differential results not found: {diff_path}")
        sys.exit(1)

    diff_df = pd.read_csv(diff_path)
    logger.info(f"Loaded {len(diff_df):,} differential sites")

    # ── Annotate ──────────────────────────────────────────────
    annotated = annotate_sites(diff_df, index, tx_models, logger)

    out_path = os.path.join(output_dir, "metatranscript_positions.csv")
    annotated.to_csv(out_path, index=False)
    logger.info(f"Saved annotated sites → {out_path}")

    # ── Summary ───────────────────────────────────────────────
    region_order = ["5UTR", "CDS", "3UTR", "intergenic"]
    summary_rows = []

    for mod_type in annotated["mod_type"].unique() if "mod_type" in annotated.columns else [""]:
        sub = annotated[annotated["mod_type"] == mod_type] if "mod_type" in annotated.columns else annotated
        for region in region_order:
            r_sub = sub[sub["region"] == region]
            sig_sub = r_sub[r_sub.get("is_significant", pd.Series(False, index=r_sub.index))]
            summary_rows.append({
                "mod_type":       mod_type,
                "region":         region,
                "total_sites":    len(r_sub),
                "significant":    int(r_sub["is_significant"].sum()) if "is_significant" in r_sub.columns else 0,
                "increased":      int((r_sub.get("direction", "") == "increased").sum()),
                "decreased":      int((r_sub.get("direction", "") == "decreased").sum()),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "metatranscript_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary → {summary_path}")

    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['mod_type']} | {row['region']}: "
            f"{row['total_sites']} sites, {row['significant']} significant"
        )

    logger.info("Step 8 complete.")


if __name__ == "__main__":
    main()
