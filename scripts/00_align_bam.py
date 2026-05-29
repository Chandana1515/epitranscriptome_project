#!/usr/bin/env python3
"""
Step 0: Align Basecalled BAM Files to Reference Transcriptome
==============================================================
Fixed version: runs each step separately to avoid memory issues
from piping millions of reads through minimap2 -> samtools sort.

Usage:
    python scripts/00_align_bam.py --reference gencode.v46.transcripts.fa --threads 4
"""

import os
import sys
import argparse
import subprocess
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config, setup_logging


def check_tools():
    """Verify minimap2 and samtools are installed."""
    for tool in ["minimap2", "samtools"]:
        if shutil.which(tool) is None:
            print(f"ERROR: '{tool}' not found. Install with: conda install -c bioconda {tool}")
            return False
        result = subprocess.run([tool, "--version"], capture_output=True, text=True)
        print(f"  Found: {result.stdout.strip().split(chr(10))[0]}")
    return True


def align_sample(input_bam, reference_fa, output_bam, threads=4, logger=None):
    """
    Align in 3 separate steps (no piping) to avoid memory issues:
      1. BAM -> FASTQ (preserving MM/ML tags)
      2. FASTQ -> aligned SAM via minimap2
      3. SAM -> sorted BAM via samtools
    """
    out_dir = os.path.dirname(output_bam)
    base = os.path.splitext(os.path.basename(output_bam))[0]

    # Use the same directory as output for temp files
    fastq_file = os.path.join(out_dir, f"{base}_temp.fastq")
    sam_file = os.path.join(out_dir, f"{base}_temp.sam")

    try:
        # ---- Step 1: BAM to FASTQ ----
        # For single-end nanopore reads, samtools fastq -o may route
        # reads incorrectly. Use shell redirect from stdout instead.
        print(f"\n  [1/4] Converting BAM to FASTQ (preserving MM/ML tags)...")
        cmd1 = f'samtools fastq -T "*" "{input_bam}" > "{fastq_file}"'
        print(f"    {cmd1}")
        r1 = subprocess.run(cmd1, shell=True)
        if r1.returncode != 0:
            print(f"  ERROR: samtools fastq failed (exit {r1.returncode})")
            return False

        fq_size = os.path.getsize(fastq_file) / (1024**3)
        print(f"    FASTQ written: {fq_size:.1f} GB")
        if fq_size < 0.01:
            print(f"  ERROR: FASTQ file is empty! Something went wrong.")
            return False

        # ---- Step 2: Align with minimap2 ----
        print(f"\n  [2/4] Aligning with minimap2 (this is the slow step)...")
        cmd2 = [
            "minimap2",
            "-ax", "map-ont",      # ONT preset
            "-N", "10",            # consider top 10 secondary for best mapping
            "--secondary=no",      # output only primary
            "-y",                  # copy tags from FASTQ comment to SAM
            "-t", str(threads),
            "-o", sam_file,
            reference_fa,
            fastq_file
        ]
        print(f"    {' '.join(cmd2)}")
        r2 = subprocess.run(cmd2)
        if r2.returncode != 0:
            print(f"  ERROR: minimap2 failed (exit {r2.returncode})")
            return False

        sam_size = os.path.getsize(sam_file) / (1024**3)
        print(f"    SAM written: {sam_size:.1f} GB")

        # Delete FASTQ to free space before sorting
        print(f"    Removing temp FASTQ to free disk space...")
        os.remove(fastq_file)
        fastq_file = None  # prevent double-delete in finally

        # ---- Step 3: Sort SAM to BAM ----
        print(f"\n  [3/4] Sorting and converting to BAM...")
        cmd3 = [
            "samtools", "sort",
            "-@", str(threads),
            "-m", "1G",            # max 1GB memory per thread
            "-o", output_bam,
            sam_file
        ]
        print(f"    {' '.join(cmd3)}")
        r3 = subprocess.run(cmd3)
        if r3.returncode != 0:
            print(f"  ERROR: samtools sort failed (exit {r3.returncode})")
            return False

        # Delete SAM to free space
        print(f"    Removing temp SAM...")
        os.remove(sam_file)
        sam_file = None

        # ---- Step 4: Index ----
        print(f"\n  [4/4] Indexing BAM...")
        cmd4 = ["samtools", "index", output_bam]
        subprocess.run(cmd4)

        # ---- Verify ----
        print(f"\n  Verifying alignment...")
        r_stat = subprocess.run(
            ["samtools", "flagstat", output_bam],
            capture_output=True, text=True
        )
        print(f"  {r_stat.stdout}")

        # Check MM/ML tags survived
        check = subprocess.run(
            f'samtools view "{output_bam}" | head -1 | tr "\\t" "\\n" | grep "^MM:Z:"',
            shell=True, capture_output=True, text=True
        )
        if check.stdout.strip():
            print(f"  MM/ML tags preserved: YES")
        else:
            print(f"  WARNING: MM/ML tags may not have been preserved!")

        return True

    finally:
        # Clean up temp files if they still exist
        for tmp in [fastq_file, sam_file]:
            if tmp and os.path.exists(tmp):
                print(f"  Cleaning up: {tmp}")
                os.remove(tmp)


def main():
    parser = argparse.ArgumentParser(
        description="Align unaligned Dorado BAM files to a reference transcriptome"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--sample", default=None,
                        help="Align only this sample")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input BAM)")
    args = parser.parse_args()

    os.makedirs("./results", exist_ok=True)
    logger = setup_logging("./results/00_alignment.log")

    logger.info("=" * 60)
    logger.info("Step 0: Align BAM Files to Reference Transcriptome")
    logger.info("=" * 60)

    print("\nChecking tools...")
    if not check_tools():
        sys.exit(1)

    if not os.path.exists(args.reference):
        print(f"\nERROR: Reference not found: {args.reference}")
        sys.exit(1)

    config = load_config(args.config)
    samples = config["samples"]
    if args.sample:
        samples = {args.sample: samples[args.sample]}

    print(f"\nReference: {args.reference}")
    print(f"Samples: {list(samples.keys())}")
    print(f"Threads: {args.threads}")

    # Check disk space
    for sample_name, info in samples.items():
        bam_path = info["bam"]
        if not os.path.exists(bam_path):
            continue
        bam_dir = os.path.dirname(bam_path)
        total, used, free = shutil.disk_usage(bam_dir)
        free_gb = free / (1024**3)
        bam_size_gb = os.path.getsize(bam_path) / (1024**3)
        needed_gb = bam_size_gb * 3  # FASTQ + SAM (not at same time thanks to cleanup)
        print(f"\n  {sample_name}:")
        print(f"    BAM size: {bam_size_gb:.1f} GB")
        print(f"    Free disk on {bam_dir}: {free_gb:.1f} GB")
        print(f"    Estimated peak temp space: ~{needed_gb:.0f} GB")
        if free_gb < needed_gb:
            print(f"    WARNING: Low disk space!")
            print(f"    Use --output-dir /path/to/bigger/drive")

    aligned_paths = {}
    for sample_name, info in samples.items():
        input_bam = info["bam"]
        if not os.path.exists(input_bam):
            print(f"\nERROR: BAM not found: {input_bam}")
            continue

        if args.output_dir:
            out_dir = args.output_dir
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(input_bam)

        base_name = os.path.basename(input_bam).replace(".bam", "_aligned.bam")
        output_bam = os.path.join(out_dir, base_name)

        print(f"\n{'='*50}")
        print(f"  Sample: {sample_name}")
        print(f"  Input:  {input_bam}")
        print(f"  Output: {output_bam}")
        print(f"{'='*50}")

        success = align_sample(
            input_bam, args.reference, output_bam,
            threads=args.threads, logger=logger
        )

        if success:
            aligned_paths[sample_name] = output_bam
        else:
            print(f"\n  FAILED: {sample_name}")

    if aligned_paths:
        print(f"\n{'='*60}")
        print("  ALIGNMENT COMPLETE")
        print(f"{'='*60}")
        print("\n  Update config.yaml with these BAM paths:\n")
        for sample_name, path in aligned_paths.items():
            info = config["samples"][sample_name]
            print(f'  {sample_name}:')
            print(f'    bam: "{path}"')
            print(f'    condition: "{info["condition"]}"')
            print(f'    oxygen: "{info["oxygen"]}"')
        print(f"\n  Then run: python run_pipeline.py")


if __name__ == "__main__":
    main()
