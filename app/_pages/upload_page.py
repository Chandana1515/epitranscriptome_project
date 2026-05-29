"""Upload & Configure page — upload BAM files and configure the pipeline entirely from the UI."""

import os
import yaml
import streamlit as st

RUNTIME_CONFIG = "runtime_config.yaml"

# All pipeline outputs live on the E: drive. Use WSL path on Linux/WSL,
# Windows drive path otherwise.
RESULTS_ROOT = r"E:\results" if os.name == "nt" else "/mnt/e/results"


def _get_project_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _results_base(cell_line):
    """Forward-slash results base path for config files (YAML-friendly)."""
    return f"{RESULTS_ROOT}/results_{cell_line}".replace("\\", "/")



def _bam_card(label, condition, oxygen, key_prefix):
    """Render a BAM file input card. Returns BAM path or None."""
    st.markdown(f"**{label}**")
    st.caption(f"Condition: `{condition}` · Oxygen: `{oxygen}`")

    saved = st.session_state.get(f"{key_prefix}_path", "")
    path = st.text_input(
        "Path to BAM file",
        value=saved,
        key=f"{key_prefix}_input",
        placeholder="/mnt/e/A375_normoxia/alignment/sample_aligned.bam",
    )

    bam_path = None
    if path:
        st.session_state[f"{key_prefix}_path"] = path
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024 ** 3)
            st.success(f"Found — {size_gb:.1f} GB")
            bam_path = path
        else:
            st.error("File not found")

    return bam_path


def _build_runtime_config(samples, comparisons, thresholds, reference_path, cell_line="results"):
    """Build a complete pipeline config from UI inputs. Never touches config.yaml."""
    config = {
        "samples": {},
        "comparisons": comparisons,
        "modification_types": [
            {"code": "a", "name": "m6A", "chebi": None},
        ],
        "primary_modification": "m6A",
        "reference_fasta": reference_path,
        "gtf_annotation": os.path.join(_get_project_dir(), "reference", "gencode.v47.annotation.gtf"),
        "thresholds": thresholds,
        "ml": {
            "model_type": "both",
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 200,
            "max_depth": 6,
        },
        "output": {
            "base_dir": _results_base(cell_line),
            "extraction_dir": f"{_results_base(cell_line)}/01_extraction",
            "stoichiometry_dir": f"{_results_base(cell_line)}/02_stoichiometry",
            "differential_dir": f"{_results_base(cell_line)}/03_differential",
            "features_dir": f"{_results_base(cell_line)}/04_features",
            "model_dir": f"{_results_base(cell_line)}/05_model",
            "shap_dir": f"{_results_base(cell_line)}/06_shap",
            "pathway_dir": f"{_results_base(cell_line)}/07_pathway",
            "figures_dir": f"{_results_base(cell_line)}/figures",
        },
        "processing": {
            "n_jobs": 4,
            "chunk_size": 10000,
            "verbose": True,
        },
    }
    for name, info in samples.items():
        if info["bam"]:
            config["samples"][name] = {
                "bam": info["bam"],
                "condition": info["condition"],
                "oxygen": info["oxygen"],
            }
    return config


def _save_runtime_config(config):
    path = os.path.join(_get_project_dir(), RUNTIME_CONFIG)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path


def render():
    st.header("Upload & configure")
    st.markdown(
        "Add your BAM files for each condition below. "
        "The pipeline configuration is generated automatically."
    )

    project_dir = _get_project_dir()
    st.session_state.pipeline_dir = project_dir

    # ── Cell line name ───────────────────────────────────────
    cell_line = st.text_input(
        "Cell line name",
        value=st.session_state.get("cell_line", ""),
        placeholder="e.g. A375, A431, A2058, SCL1",
        help="Results will be saved to results_<cell_line>/",
    )
    if cell_line:
        st.session_state["cell_line"] = cell_line
        st.caption(f"Output directory: `{_results_base(cell_line)}/`")
    else:
        st.warning("Enter a cell line name to set the output directory.")

    # ── BAM File Inputs ─────────────────────────────────────
    st.subheader("Sample BAM files")
    st.caption("Provide aligned BAM files with modification tags (MM/ML from Dorado).")

    samples = {}

    # Row 1: Untreated
    st.markdown("---")
    st.markdown("##### Untreated (control)")
    col1, col2 = st.columns(2)

    with col1:
        bam = _bam_card(
            "Normoxia — untreated",
            condition="untreated", oxygen="normoxia",
            key_prefix="untreated_normoxia",
        )
        samples["untreated_normoxia"] = {
            "bam": bam, "condition": "untreated", "oxygen": "normoxia",
        }

    with col2:
        bam = _bam_card(
            "Hypoxia — untreated",
            condition="untreated", oxygen="hypoxia",
            key_prefix="untreated_hypoxia",
        )
        samples["untreated_hypoxia"] = {
            "bam": bam, "condition": "untreated", "oxygen": "hypoxia",
        }

    # Row 2: Plasma-treated
    st.markdown("---")
    st.markdown("##### Plasma-treated")
    col3, col4 = st.columns(2)

    with col3:
        bam = _bam_card(
            "Normoxia — plasma-treated",
            condition="plasma_treated", oxygen="normoxia",
            key_prefix="plasma_normoxia",
        )
        samples["plasma_normoxia"] = {
            "bam": bam, "condition": "plasma_treated", "oxygen": "normoxia",
        }

    with col4:
        bam = _bam_card(
            "Hypoxia — plasma-treated",
            condition="plasma_treated", oxygen="hypoxia",
            key_prefix="plasma_hypoxia",
        )
        samples["plasma_hypoxia"] = {
            "bam": bam, "condition": "plasma_treated", "oxygen": "hypoxia",
        }

    # ── Summary ─────────────────────────────────────────────
    loaded = {k: v for k, v in samples.items() if v["bam"]}
    st.markdown("---")
    st.markdown(f"**{len(loaded)} / 4 samples loaded** — minimum 2 needed to run")

    # ── Comparisons ─────────────────────────────────────────
    st.subheader("Comparisons")
    st.caption("Auto-generated from loaded samples.")

    comparisons = []
    auto_pairs = [
        ("untreated_normoxia", "plasma_normoxia", "Normoxia: untreated → plasma"),
        ("untreated_hypoxia", "plasma_hypoxia", "Hypoxia: untreated → plasma"),
        ("untreated_normoxia", "untreated_hypoxia", "Untreated: normoxia → hypoxia"),
        ("plasma_normoxia", "plasma_hypoxia", "Plasma: normoxia → hypoxia"),
    ]
    for ctrl, treat, label in auto_pairs:
        if ctrl in loaded and treat in loaded:
            enabled = st.checkbox(label, value=True, key=f"comp_{ctrl}_{treat}")
            if enabled:
                comparisons.append({"control": ctrl, "treatment": treat})

    if not comparisons:
        st.info("Load at least 2 samples to enable comparisons.")

    # ── Reference genome ────────────────────────────────────
    st.subheader("Reference genome")

    ref_dir = os.path.join(project_dir, "reference")
    ref_files = []
    if os.path.isdir(ref_dir):
        ref_files = [
            f for f in os.listdir(ref_dir)
            if f.endswith((".fa", ".fasta", ".fna")) and not f.startswith(".")
        ]

    if ref_files:
        selected_ref = st.selectbox("Reference FASTA", ref_files)
        reference_path = os.path.join(ref_dir, selected_ref)
        st.success(f"Using: `reference/{selected_ref}`")
    else:
        reference_path = None
        st.warning("No reference FASTA in `reference/` directory")

    # ── Thresholds ──────────────────────────────────────────
    st.subheader("Analysis thresholds")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mod_thresh = st.slider(
            "Mod probability", 0.50, 1.0, 0.75, 0.05,
            help="Reads above this are called modified",
        )
    with col2:
        min_cov = st.number_input("Min coverage", 1, 100, 10)
    with col3:
        fdr = st.number_input("FDR threshold", 0.001, 0.5, 0.05, 0.01, format="%.3f")
    with col4:
        min_delta = st.number_input("Min Δ stoich", 0.01, 0.5, 0.10, 0.01, format="%.2f")

    thresholds = {
        "modification_probability": int(round(mod_thresh * 255)),
        "min_coverage": min_cov,
        "fdr_threshold": fdr,
        "min_delta_stoichiometry": min_delta,
    }

    # ── Save ────────────────────────────────────────────────
    st.markdown("---")

    can_save = len(loaded) >= 2 and len(comparisons) >= 1
    if st.button(
        "💾  Save configuration",
        type="primary",
        use_container_width=True,
        disabled=not can_save,
    ):
        config = _build_runtime_config(samples, comparisons, thresholds, reference_path, cell_line or "results")
        path = _save_runtime_config(config)
        st.session_state.config = config
        st.session_state.runtime_config_path = path
        st.success(f"Saved to `{RUNTIME_CONFIG}` — go to **Run Pipeline** next")

        with st.expander("View generated config"):
            st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language="yaml")

    if not can_save:
        st.caption("Load at least 2 samples and enable a comparison to save.")

    # ── Existing results ────────────────────────────────────
    st.subheader("Existing results")

    step_dirs = {
        "01_extraction":   "Modification extraction",
        "02_stoichiometry": "Site stoichiometry",
        "03_differential": "Differential analysis",
        "04_features":     "Feature matrix",
        "05_model":        "ML models",
        "06_shap":         "SHAP analysis",
        "07_pathway":      "Pathway enrichment",
        "08_metatranscript": "Metatranscript analysis",
    }

    # Scan E:/results for results_* directories
    st.caption(f"Scanning: `{RESULTS_ROOT}/`")
    result_runs = []
    if os.path.isdir(RESULTS_ROOT):
        result_runs = sorted([
            d for d in os.listdir(RESULTS_ROOT)
            if d.startswith("results") and os.path.isdir(os.path.join(RESULTS_ROOT, d))
        ])

    if result_runs:
        selected_run = st.selectbox("Select run", result_runs)
        run_dir = os.path.join(RESULTS_ROOT, selected_run)
        for dirname, label in step_dirs.items():
            path = os.path.join(run_dir, dirname)
            if os.path.isdir(path):
                n = len([f for f in os.listdir(path) if not f.endswith(".log") and not f.startswith(".")])
                st.markdown(f"{'✅' if n > 0 else '⬜'} **{label}** — {n} output files")
            else:
                st.markdown(f"⬜ **{label}** — not run yet")
    else:
        st.info(f"No results yet in `{RESULTS_ROOT}/`.")