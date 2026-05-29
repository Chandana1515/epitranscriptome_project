"""
Utility functions for the Epitranscriptome Analysis Pipeline.
Handles configuration loading, MM/ML tag parsing, and common operations.
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging for the pipeline."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load and validate the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create output directories
    for key, path in config["output"].items():
        os.makedirs(path, exist_ok=True)

    return config


def prob_to_float(prob_uint8):
    """Convert ML tag probability (0-255 uint8) to float (0.0-1.0)."""
    return prob_uint8 / 255.0


def float_to_uint8(prob_float):
    """Convert float probability (0.0-1.0) to uint8 (0-255)."""
    return int(round(prob_float * 255))


def parse_mm_tag(mm_string):
    """
    Parse the MM tag from a BAM record.
 
    The MM tag format is:
        BASE+MOD[.?],SKIP1,SKIP2,...;BASE+MOD[.?],SKIP1,...;...
    
    The delimiter after MOD can be '.' (all bases listed) or '?' (subset).
 
    Returns a list of dicts:
        [{'base': 'A', 'mod_code': 'a', 'skips': [0, 0, 3, ...]}, ...]
    """
    if not mm_string:
        return []
 
    entries = []
    # Split by semicolons to get each modification type
    parts = mm_string.rstrip(";").split(";")
 
    for part in parts:
        part = part.strip()
        if not part:
            continue
 
        # Find the delimiter: either '.' or '?'
        # Format: A+a. or A+a? or A+17596. or A+17596?
        delim_idx = -1
        for delim in [".", "?"]:
            idx = part.find(delim)
            if idx >= 0:
                if delim_idx < 0 or idx < delim_idx:
                    delim_idx = idx
 
        if delim_idx < 0:
            # No delimiter found — skip this entry
            continue
 
        header = part[:delim_idx]
        skips_str = part[delim_idx + 1:]  # everything after the delimiter
 
        # Parse header: e.g. "A+a" or "A+17596"
        if "+" not in header:
            continue
 
        base = header[0]
        mod_code = header[2:]  # skip the '+'
 
        # Parse skip values
        if skips_str.startswith(","):
            skips_str = skips_str[1:]
 
        skips = []
        if skips_str:
            try:
                skips = [int(s) for s in skips_str.split(",") if s]
            except ValueError:
                continue
 
        entries.append({
            "base": base,
            "mod_code": mod_code,
            "skips": skips,
            "n_calls": len(skips)
        })
 
    return entries


def get_modification_positions(read, mm_entries):
    """
    Given a pysam AlignedSegment and parsed MM entries,
    compute the reference positions for each modification call.

    Returns a list of dicts:
        [{'mod_code': 'a', 'ref_pos': 12345, 'query_pos': 50}, ...]
    """
    if read.is_unmapped:
        return []

    query_seq = read.query_sequence
    if query_seq is None:
        return []

    # Build mapping from query position to reference position
    aligned_pairs = read.get_aligned_pairs(matches_only=False)
    query_to_ref = {}
    for qpos, rpos in aligned_pairs:
        if qpos is not None and rpos is not None:
            query_to_ref[qpos] = rpos

    is_reverse = read.is_reverse
    results = []

    for entry in mm_entries:
        base = entry["base"]
        mod_code = entry["mod_code"]
        skips = entry["skips"]

        if not skips:
            continue

        # Find all positions of the canonical base in the query sequence
        # For reverse strand reads, the MM tag refers to the original strand
        if is_reverse:
            # Complement the base for reverse strand
            complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
            target_base = complement.get(base, base)
            # Positions are in reverse order for reverse strand
            base_positions = [
                i for i, b in enumerate(query_seq)
                if b.upper() == target_base
            ]
            base_positions = base_positions[::-1]
        else:
            base_positions = [
                i for i, b in enumerate(query_seq)
                if b.upper() == base
            ]

        # Walk through skip values to find modified positions
        base_idx = 0
        for skip in skips:
            base_idx += skip  # skip unmodified bases
            if base_idx >= len(base_positions):
                break

            query_pos = base_positions[base_idx]
            ref_pos = query_to_ref.get(query_pos)

            if ref_pos is not None:
                results.append({
                    "mod_code": mod_code,
                    "query_pos": query_pos,
                    "ref_pos": ref_pos
                })

            base_idx += 1  # move past the modified base

    return results


def assign_ml_probabilities(mod_positions, ml_array, mm_entries):
    """
    Assign ML tag probabilities to each modification position.

    The ML array contains probabilities in the same order as MM entries.
    For multiple modification types, probabilities are interleaved per-position.

    Returns mod_positions with 'probability' field added.
    """
    if not ml_array or not mod_positions:
        return mod_positions

    # The ML array has values for each modification call, grouped by type
    # and then by position. With multiple mod types per base, they interleave.
    #
    # For the inosine_m6A_2OmeA model, each A position gets 3 probabilities
    # (one per modification type), interleaved in the ML array.

    n_types = len(mm_entries)
    n_calls_per_type = [e["n_calls"] for e in mm_entries]

    # Check if modifications are interleaved or sequential
    # Dorado uses interleaved format when multiple mods are on the same base
    same_base = len(set(e["base"] for e in mm_entries)) == 1

    if same_base and n_types > 1:
        # Interleaved: for each position, n_types probabilities in sequence
        # Total ML values = n_calls * n_types (all types have same n_calls)
        n_positions = n_calls_per_type[0]  # all should be equal
        ml_idx = 0
        pos_idx = 0

        # Group mod_positions by type
        by_type = {}
        for mp in mod_positions:
            code = mp["mod_code"]
            if code not in by_type:
                by_type[code] = []
            by_type[code].append(mp)

        type_order = [e["mod_code"] for e in mm_entries]

        for i in range(n_positions):
            for t_idx, mod_code in enumerate(type_order):
                if ml_idx < len(ml_array):
                    # Find the corresponding position entry
                    type_positions = by_type.get(mod_code, [])
                    if i < len(type_positions):
                        type_positions[i]["probability"] = prob_to_float(
                            ml_array[ml_idx]
                        )
                    ml_idx += 1
    else:
        # Sequential: all calls for type 1, then all for type 2, etc.
        ml_idx = 0
        type_start = {}
        for entry in mm_entries:
            type_start[entry["mod_code"]] = ml_idx
            ml_idx += entry["n_calls"]

        type_counters = {e["mod_code"]: 0 for e in mm_entries}

        for mp in mod_positions:
            code = mp["mod_code"]
            offset = type_start.get(code, 0) + type_counters.get(code, 0)
            if offset < len(ml_array):
                mp["probability"] = prob_to_float(ml_array[offset])
            type_counters[code] = type_counters.get(code, 0) + 1

    # Fill any missing probabilities
    for mp in mod_positions:
        if "probability" not in mp:
            mp["probability"] = 0.0

    return mod_positions


def mod_code_to_name(code):
    """Map modification code to human-readable name."""
    mapping = {
        "a": "m6A",
        "17596": "inosine",
        "69426": "2OmeA",
        "m": "5mC",
        "h": "5hmC",
    }
    return mapping.get(code, f"mod_{code}")


def compute_stoichiometry(modified_count, total_count):
    """Compute modification stoichiometry with zero-division safety."""
    if total_count == 0:
        return np.nan
    return modified_count / total_count


def save_dataframe(df, path, index=False):
    """Save DataFrame to CSV and optionally parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    # Also save parquet for faster loading
    parquet_path = path.replace(".csv", ".parquet")
    try:
        df.to_parquet(parquet_path, index=index)
    except Exception:
        pass  # parquet optional
    return path


def load_dataframe(path):
    """Load DataFrame, preferring parquet if available."""
    parquet_path = path.replace(".csv", ".parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.read_csv(path)


def get_timestamp():
    """Get current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
