#!/usr/bin/env python3
"""
Project Setup Checker
=====================
Verifies all dependencies and creates output directories.
Run with: python setup.py
"""

import os
import sys
import subprocess
import shutil


def check_python():
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}", end="")
    if v.major >= 3 and v.minor >= 9:
        print(" ✓")
        return True
    print(" ✗ (need 3.9+)")
    return False


def check_tool(name):
    path = shutil.which(name)
    if path:
        r = subprocess.run([name, "--version"], capture_output=True, text=True)
        ver = (r.stdout or r.stderr).strip().split("\n")[0]
        print(f"  {name}: {ver} ✓")
        return True
    print(f"  {name}: NOT FOUND ✗")
    return False


def check_package(name, import_name=None):
    try:
        __import__(import_name or name)
        print(f"  {name}: OK ✓")
        return True
    except ImportError:
        print(f"  {name}: MISSING ✗")
        return False


def check_ollama():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  Ollama server: running ✓")
            print(f"  Ollama models: {models or '(none — run: ollama pull llama3.1:8b)'}")
            return True
    except Exception:
        pass
    print("  Ollama server: not running ✗")
    return False


def create_dirs():
    dirs = [
        "results/01_extraction", "results/02_stoichiometry",
        "results/03_differential", "results/04_features",
        "results/05_model", "results/06_shap",
        "results/07_pathway", "results/figures", "reference",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("  Output directories: created ✓")


def main():
    print("=" * 50)
    print("  EpiTranscriptome Project — Setup Check")
    print("=" * 50)

    print("\n[1] Python")
    check_python()

    print("\n[2] Bioinformatics tools")
    m = check_tool("minimap2")
    s = check_tool("samtools")
    if not m or not s:
        print("  Fix: conda install -c bioconda minimap2 samtools")

    print("\n[3] Python packages")
    packages = [
        ("pysam", "pysam"), ("numpy", "numpy"), ("pandas", "pandas"),
        ("scipy", "scipy"), ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"), ("xgboost", "xgboost"),
        ("shap", "shap"), ("streamlit", "streamlit"),
        ("plotly", "plotly"), ("pyyaml", "yaml"), ("tqdm", "tqdm"),
        ("requests", "requests"),
    ]
    missing = []
    for name, imp in packages:
        if not check_package(name, imp):
            missing.append(name)
    if missing:
        print(f"\n  Fix: pip install -r requirements.txt")

    print("\n[4] Ollama (AI assistant)")
    check_tool("ollama")
    check_ollama()

    print("\n[5] Directories")
    create_dirs()

    print("\n[6] Config & reference files")
    print(f"  config.yaml: {'found ✓' if os.path.exists('config.yaml') else 'NOT FOUND ✗'}")
    ref = "reference/gencode.v46.transcripts.fa"
    print(f"  Reference FASTA: {'found ✓' if os.path.exists(ref) else 'not downloaded yet'}")

    print("\n" + "=" * 50)
    print("  Setup check complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
