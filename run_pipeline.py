#!/usr/bin/env python3
"""
Epitranscriptome Pipeline - Master Runner
Run with: python run_pipeline.py [--from-step 1] [--to-step 8] [--method fisher]
"""

import argparse
import json
import os
import sys
import subprocess
import yaml

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
_BASE = os.path.dirname(os.path.abspath(__file__))

DONE_FILE = os.path.join(_BASE, "pipeline_done.json")

# All pipeline outputs live on the E: drive (plenty of space). Use the WSL
# path when running under Linux/WSL, and the Windows drive path otherwise.
RESULTS_ROOT = r"E:\results" if os.name == "nt" else "/mnt/e/results"


def _resolve_config(explicit=None):
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(_BASE, explicit)
        if os.path.exists(path):
            return path
        print(f"ERROR: Config file not found: {path}")
        sys.exit(1)

    runtime = os.path.join(_BASE, "runtime_config.yaml")
    fallback = os.path.join(_BASE, "config.yaml")

    if os.path.exists(runtime) and os.path.exists(fallback):
        # Use the most recently modified one
        chosen = runtime if os.path.getmtime(runtime) >= os.path.getmtime(fallback) else fallback
        print(f"Both configs found — using most recently modified: {os.path.basename(chosen)}")
        print(f"  (pass --config <file> to choose explicitly)")
        return chosen

    if os.path.exists(runtime):
        return runtime
    if os.path.exists(fallback):
        return fallback

    print("ERROR: No config file found (runtime_config.yaml or config.yaml)")
    sys.exit(1)


def _detect_cell_line(config):
    """
    Derive cell line name.
    Priority 1: extract from output.base_dir if it already encodes a name
                (e.g. './results_A375' → 'A375').
    Priority 2: parse from sample keys (e.g. 'A2058_normoxia' → 'A2058').
    """
    import re

    # Priority 1 — base_dir set by UI (e.g. "./results_A375")
    base = config.get("output", {}).get("base_dir", "")
    m = re.search(r"results_(.+)$", base)
    if m:
        candidate = m.group(1).strip("/").strip("\\")
        # Only trust it if it doesn't look like a generic placeholder
        if candidate and candidate not in ("results",):
            return candidate

    # Priority 2 — derive from sample keys
    _KNOWN_SUFFIXES = [
        "_normoxia_plasma", "_hypoxia_plasma",
        "_plasma_normoxia", "_plasma_hypoxia",
        "_normoxia", "_hypoxia", "_plasma",
        "_treated", "_untreated",
    ]
    # Also skip purely-generic key names used by the UI
    _GENERIC = {"untreated", "plasma", "plasma_treated"}

    names = set()
    for key in config.get("samples", {}):
        name = key
        for suf in _KNOWN_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        if name not in _GENERIC:
            names.add(name)

    if len(names) == 1:
        return names.pop()
    if names:
        common = os.path.commonprefix(sorted(names)).rstrip("_")
        if common:
            return common

    return "results"


def _set_output_dirs(config, cell_line):
    """Override output dirs in config to {RESULTS_ROOT}/results_{cell_line}/..."""
    base = f"{RESULTS_ROOT}/results_{cell_line}".replace("\\", "/")
    os.makedirs(base, exist_ok=True)
    config["output"] = {
        "base_dir":        base,
        "extraction_dir":  f"{base}/01_extraction",
        "stoichiometry_dir": f"{base}/02_stoichiometry",
        "differential_dir": f"{base}/03_differential",
        "features_dir":    f"{base}/04_features",
        "model_dir":       f"{base}/05_model",
        "shap_dir":        f"{base}/06_shap",
        "pathway_dir":     f"{base}/07_pathway",
        "figures_dir":     f"{base}/figures",
    }
    return config


def _write_done(success, failed_step=None):
    with open(DONE_FILE, "w") as f:
        json.dump({"success": success, "failed_step": failed_step}, f)


def check_bam_alignment(config):
    try:
        import pysam
    except ImportError:
        print("WARNING: pysam not installed, skipping alignment check")
        return True

    all_ok = True
    for sample_name, sample_info in config["samples"].items():
        bam_path = sample_info["bam"]
        if not os.path.exists(bam_path):
            print(f"  ERROR: BAM not found: {bam_path}")
            all_ok = False
            continue

        n_mapped = 0
        n_checked = 0
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            for read in bam.fetch(until_eof=True):
                n_checked += 1
                if not read.is_unmapped:
                    n_mapped += 1
                if n_checked >= 100:
                    break

        if n_mapped == 0:
            print(f"  {sample_name}: UNALIGNED (0/{n_checked} reads mapped)")
            all_ok = False
        else:
            print(f"  {sample_name}: OK ({n_mapped}/{n_checked} reads mapped)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Epitranscriptome pipeline runner")
    parser.add_argument("--from-step", type=int, default=1, dest="from_step",
                        help="Start from this step number (1-8)")
    parser.add_argument("--to-step", type=int, default=8, dest="to_step",
                        help="Run through this step number (1-8)")
    parser.add_argument("--method", default="fisher",
                        choices=["fisher", "logistic"],
                        help="Statistical method for Step 3")
    parser.add_argument("--config", default=None,
                        help="Path to config file (runtime_config.yaml or config.yaml). "
                             "If omitted, uses the most recently modified of the two.")
    args = parser.parse_args()

    CONFIG = _resolve_config(args.config)
    from_step = max(1, min(args.from_step, 8))
    to_step = max(from_step, min(args.to_step, 8))

    # Remove stale done file at start
    if os.path.exists(DONE_FILE):
        os.remove(DONE_FILE)

    print("=" * 56)
    print("  Epitranscriptome Analysis Pipeline")
    print(f"  Config: {CONFIG}")
    print(f"  Steps:  {from_step} → {to_step}")
    print("=" * 56)

    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    cell_line = _detect_cell_line(config)
    config = _set_output_dirs(config, cell_line)
    print(f"  Cell line: {cell_line}  →  {config['output']['base_dir']}/")

    # Scripts run as subprocesses and re-read the config file from disk, so
    # the in-memory overrides must be written to an effective config that
    # every step reads.
    EFFECTIVE_CONFIG = os.path.join(_BASE, ".effective_config.yaml")
    with open(EFFECTIVE_CONFIG, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    if from_step == 1:
        print("\nChecking BAM files...")
        if not check_bam_alignment(config):
            print("\nYour BAM files appear unaligned. Align them first.")
            _write_done(False, failed_step=0)
            sys.exit(1)
        print("All BAM files are aligned. Starting pipeline...\n")

    all_steps = [
        ("1/8", "Extracting modifications from BAM files",
         [sys.executable, os.path.join(SCRIPTS_DIR, "01_extract_modifications.py"),
          "--config", EFFECTIVE_CONFIG]),

        ("2/8", "Computing site-level stoichiometry",
         [sys.executable, os.path.join(SCRIPTS_DIR, "02_site_stoichiometry.py"),
          "--config", EFFECTIVE_CONFIG]),

        ("3/8", "Running differential modification analysis",
         [sys.executable, os.path.join(SCRIPTS_DIR, "03_differential_analysis.py"),
          "--config", EFFECTIVE_CONFIG, "--method", args.method]),

        ("4/8", "Constructing biological feature matrix",
         [sys.executable, os.path.join(SCRIPTS_DIR, "04_feature_construction.py"),
          "--config", EFFECTIVE_CONFIG]),

        ("5/8", "Training explainable ML models",
         [sys.executable, os.path.join(SCRIPTS_DIR, "05_train_model.py"),
          "--config", EFFECTIVE_CONFIG]),

        ("6/8", "Running SHAP analysis",
         [sys.executable, os.path.join(SCRIPTS_DIR, "06_shap_analysis.py"),
          "--config", EFFECTIVE_CONFIG, "--model", "xgboost"]),

        ("7/8", "Pathway enrichment analysis",
         [sys.executable, os.path.join(SCRIPTS_DIR, "07_pathway_enrichment.py"),
          "--config", EFFECTIVE_CONFIG, "--method", "both"]),

        ("8/8", "Metatranscript position analysis",
         [sys.executable, os.path.join(SCRIPTS_DIR, "08_metatranscript_analysis.py"),
          "--config", EFFECTIVE_CONFIG]),
    ]

    steps_to_run = [(num, desc, cmd) for i, (num, desc, cmd) in enumerate(all_steps)
                    if from_step <= i + 1 <= to_step]

    for step_num_str, description, cmd in steps_to_run:
        step_num = int(step_num_str.split("/")[0])
        print(f"\n[Step {step_num_str}] {description}...")
        sys.stdout.flush()
        result = subprocess.run(cmd, cwd=_BASE)
        if result.returncode != 0:
            print(f"\nERROR: Step {step_num} failed with exit code {result.returncode}")
            _write_done(False, failed_step=step_num)
            sys.exit(1)

    print("\n" + "=" * 56)
    print("  Pipeline complete!")
    print("=" * 56)
    sys.stdout.flush()
    _write_done(True)


if __name__ == "__main__":
    main()
