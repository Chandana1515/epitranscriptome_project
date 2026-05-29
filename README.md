# EpiTranscriptome Project

**Explainable AI for Detecting RNA Changes in Plasma-Treated Skin Cancer Using Nanopore Sequencing**

Master's Thesis — Chandana Nagaraju, Universität Rostock, 2026

---

## Overview

This project detects and interprets RNA m6A modification changes between conditions
(e.g. normoxia vs. hypoxia, untreated vs. plasma-treated) from Oxford Nanopore
modification-aware BAM files. It combines a bioinformatics pipeline with an
explainable machine-learning layer (XGBoost / Random Forest + SHAP) and a
Streamlit dashboard with an Ollama-powered AI chat for interpreting results.

---

## Folder Structure

```
epitranscriptome_project/
│
├── config.yaml              # Sample paths, thresholds, comparisons (template)
├── runtime_config.yaml      # Config written by the app's UI (used if newer)
├── requirements.txt         # Python dependencies
├── run_pipeline.py          # CLI pipeline runner (Steps 1–8)
├── setup.py                 # Setup / dependency verification script
├── pipeline_done.json       # Pipeline completion status (auto-written)
├── README.md
│
├── scripts/                 # Analysis pipeline
│   ├── utils.py
│   ├── 00_align_bam.py              # Align basecalled BAM to reference
│   ├── 01_extract_modifications.py  # Extract MM/ML mod probabilities
│   ├── 02_site_stoichiometry.py     # Site-level modification stoichiometry
│   ├── 03_differential_analysis.py  # Differential modification (Fisher/logistic)
│   ├── 04_feature_construction.py   # Biological feature matrix
│   ├── 05_train_model.py            # Train XGBoost / Random Forest
│   ├── 06_shap_analysis.py          # SHAP feature interpretation
│   ├── 07_pathway_enrichment.py     # GO / KEGG enrichment (gseapy)
│   ├── 08_metatranscript_analysis.py# Metatranscript position profiling
│   └── generate_report.py           # PDF report of all results
│
├── app/                     # Streamlit dashboard + AI chat
│   ├── app.py
│   ├── ollama_helper.py
│   ├── result_loader.py
│   └── _pages/
│       ├── upload_page.py
│       ├── pipeline_page.py
│       ├── results_page.py
│       └── chat_page.py
│
├── notebook/
│   └── epitranscriptome_analysis.ipynb
│
├── reference/               # Reference genome / annotation (download here)
│   ├── GRCh38.primary_assembly.genome.fa
│   └── GRCh38.primary_assembly.genome.fa.fai
│
└── results_<cell_line>/     # Pipeline outputs (auto-populated)
    ├── 01_extraction/  02_stoichiometry/  03_differential/
    ├── 04_features/    05_model/          06_shap/
    ├── 07_pathway/     figures/
```

> **Output location:** `run_pipeline.py` writes results to
> `E:\results\results_<cell_line>` (Windows) or `/mnt/e/results/results_<cell_line>`
> (WSL/Linux). The cell line is auto-detected from the config sample names or `output.base_dir`.

---

## Pipeline Steps

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `00_align_bam.py` | Align basecalled BAM to reference transcriptome (run manually before the pipeline) |
| 1 | `01_extract_modifications.py` | Extract per-read m6A probabilities from MM/ML tags |
| 2 | `02_site_stoichiometry.py` | Aggregate to site-level stoichiometry |
| 3 | `03_differential_analysis.py` | Identify significantly changed sites (Fisher or logistic) |
| 4 | `04_feature_construction.py` | Build biological feature matrix per site |
| 5 | `05_train_model.py` | Train explainable ML model (XGBoost / Random Forest) |
| 6 | `06_shap_analysis.py` | SHAP interpretation of feature importance |
| 7 | `07_pathway_enrichment.py` | GO / KEGG pathway enrichment |
| 8 | `08_metatranscript_analysis.py` | Map sites onto 5'UTR→CDS→3'UTR metatranscript |

---

## Setup

### 1. Environment

One conda environment is used for everything (pipeline + app + Ollama).

```bash
conda create -n thesis_env python=3.11 -y
conda activate thesis_env
conda install -c bioconda minimap2 samtools pysam -y
```

### 2. Install dependencies

```bash
cd /mnt/c/Users/Chandhu/Downloads/epitranscriptome_project
pip install -r requirements.txt
```

### 3. Verify setup

```bash
python setup.py
```

Checks Python, bioinformatics tools, Python packages, Ollama, and creates
output directories.

### 4. Reference genome

Place your reference FASTA in `reference/` and point `config.yaml` at it
(`reference_fasta` and `gtf_annotation`). The repo currently uses
`GRCh38.primary_assembly.genome.fa`.

### 5. Configure samples

Edit `config.yaml`:

```yaml
samples:
  SCL-1_normoxia:
    bam: "/mnt/e/SCL-1_normoxia/alignment/...bam"
    condition: "untreated"
    oxygen: "normoxia"
  SCL-1_hypoxia:
    bam: "/mnt/e/SCL-1_hypoxia/alignment/...bam"
    condition: "untreated"
    oxygen: "hypoxia"

comparisons:
  - control: "SCL-1_normoxia"
    treatment: "SCL-1_hypoxia"
```

---

## Running

### Align BAM files (Step 0, one-time per sample)

```bash
python scripts/00_align_bam.py \
  --reference reference/GRCh38.primary_assembly.genome.fa \
  --threads 4 --sample SCL-1_normoxia
```

Update `config.yaml` with the aligned BAM paths afterwards.

### Run the pipeline (Steps 1–8)

```bash
python run_pipeline.py
```

Useful flags:

```bash
python run_pipeline.py --from-step 3 --to-step 6   # run a subset of steps
python run_pipeline.py --method logistic           # Step 3 statistical method
python run_pipeline.py --config config.yaml        # choose config explicitly
```

If both `config.yaml` and `runtime_config.yaml` exist, the most recently
modified one is used unless `--config` is given.

### Generate a PDF report

```bash
python scripts/generate_report.py --config config.yaml
```

---

## Web App (Dashboard + AI Chat)

### Install Ollama (for AI interpretation)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

### Launch

```bash
streamlit run app/app.py
```

Opens at **http://localhost:8501**. The app provides upload, pipeline control,
results visualisation, and an AI chat page for interpreting findings.

---

## Adding Plasma-Treated Data Later

1. Place the new BAM files on the data drive.
2. Align: `python scripts/00_align_bam.py --reference reference/GRCh38.primary_assembly.genome.fa --sample SCL-1_normoxia_plasma`
3. Add the new samples and comparison pairs to `config.yaml`.
4. Re-run: `python run_pipeline.py`
5. Refresh the Streamlit app.
