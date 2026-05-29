#!/usr/bin/env python3
"""
Generate PDF Report from Pipeline Results
==========================================
Every chart and table gets its own full-size page.
All values shown completely — no truncation.

Usage: python scripts/generate_report.py --config config.yaml
Output: results/report/epitranscriptome_report.pdf
"""

import os, sys, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config

sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})

PAGE = (11.69, 8.27)  # A4 landscape


def load_results(results_dir):
    results = {}
    def _csv(rel):
        p = os.path.join(results_dir, rel)
        for ext in [".csv", ".parquet"]:
            f = p if p.endswith(ext) else p.replace(".csv", ext)
            if os.path.exists(f):
                try:
                    return pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
                except Exception:
                    pass
        return pd.DataFrame()
    def _json(rel):
        p = os.path.join(results_dir, rel)
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
        return {}

    results["stoichiometry"] = _csv("02_stoichiometry/all_samples_stoichiometry.csv")
    results["differential"]  = _csv("03_differential/all_differential_results.csv")
    if not results["differential"].empty and "is_significant" in results["differential"].columns:
        results["significant"] = results["differential"][
            results["differential"]["is_significant"] == True
        ].sort_values("fdr" if "fdr" in results["differential"].columns else "pvalue")
    else:
        results["significant"] = pd.DataFrame()
    results["features"] = _csv("04_features/feature_matrix.csv")
    for m in ["xgboost", "random_forest"]:
        v = _json(f"05_model/{m}_metrics.json")
        if v: results["model_metrics"] = v
        imp = _csv(f"05_model/{m}_importances.csv")
        if not imp.empty: results["importances"] = imp
        roc = _csv(f"05_model/{m}_roc_curve.csv")
        if not roc.empty: results["roc"] = roc
    results["shap"]    = _csv("06_shap/shap_global_importance.csv")
    results["pathway"] = _csv("07_pathway/curated_pathway_enrichment.csv")

    # Step 8 — metatranscript
    meta = _csv("08_metatranscript/metatranscript_positions.csv")
    results["metatranscript_all"] = meta
    if not meta.empty and "is_significant" in meta.columns:
        results["metatranscript_sig"] = meta[meta["is_significant"] == True].copy()
    else:
        results["metatranscript_sig"] = pd.DataFrame()

    return results


# ── Helper to draw a table on a full page ───────────────────

def _draw_table(pdf, title, df, col_widths=None, highlight_col=None, highlight_thresh=None):
    """Draw a DataFrame as a table on its own full page."""
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, title, ha="center", fontsize=16, fontweight="bold")

    ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])
    ax.axis("off")

    cell_text = df.values.tolist()
    col_labels = df.columns.tolist()

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    n_rows = len(df)
    # Scale row height based on how many rows
    row_height = min(2.0, max(1.3, 18.0 / max(n_rows, 1)))
    table.scale(1, row_height)

    # Auto-size columns
    table.auto_set_column_width(list(range(len(col_labels))))

    # Style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")

        # Highlight rows where a column meets threshold
        if highlight_col is not None and highlight_thresh is not None and row > 0:
            try:
                col_idx = col_labels.index(highlight_col)
                val = float(cell_text[row - 1][col_idx])
                if val < highlight_thresh:
                    if row % 2 == 1:
                        cell.set_facecolor("#ffeaea")
                    else:
                        cell.set_facecolor("#ffe0e0")
            except (ValueError, IndexError):
                pass

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page functions ──────────────────────────────────────────

def page_title(pdf, config, results=None):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.65, "EpiTranscriptome Analysis Report",
             ha="center", fontsize=28, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.55,
             "Explainable AI for Detecting RNA Changes in\nPlasma-Treated Skin Cancer Using Nanopore Sequencing",
             ha="center", fontsize=15, color="#444444", linespacing=1.5)
    fig.text(0.5, 0.42, "Chandana Nagaraju\nMaster's Thesis — Universität Rostock",
             ha="center", fontsize=13, color="#666666", linespacing=1.5)

    # Derive sample names from the actual run results, not the (possibly stale)
    # config.yaml. Falls back to config when results are missing.
    sample_names = []
    if results is not None:
        stoich = results.get("stoichiometry", pd.DataFrame())
        if not stoich.empty and "sample" in stoich.columns:
            sample_names = sorted(stoich["sample"].dropna().unique().tolist())
    config_samples = config.get("samples", {})
    if not sample_names:
        sample_names = list(config_samples.keys())

    def _meta(name):
        info = config_samples.get(name)
        if info:
            return info.get("condition", "?"), info.get("oxygen", "?")
        # Infer from suffix
        lname = name.lower()
        oxy = "hypoxia" if "hypoxia" in lname else ("normoxia" if "normoxia" in lname else "?")
        cond = "plasma_treated" if "plasma" in lname else "untreated"
        return cond, oxy

    sample_strs = [f"{n} ({_meta(n)[0]}/{_meta(n)[1]})" for n in sample_names]
    fig.text(0.5, 0.30,
             "Samples: " + ", ".join(sample_strs),
             ha="center", fontsize=10, color="#888888")

    # Derive comparisons from differential results if possible.
    comps = []
    if results is not None:
        diff = results.get("differential", pd.DataFrame())
        if not diff.empty and {"sample_ctrl", "sample_treat"}.issubset(diff.columns):
            comps = [{"control": c, "treatment": t} for c, t in
                     diff[["sample_ctrl", "sample_treat"]].drop_duplicates().itertuples(index=False)]
    if not comps:
        # Fall back to config, filtered to samples that actually ran.
        sset = set(sample_names)
        for c in config.get("comparisons", []):
            if not sset or (c.get("control") in sset and c.get("treatment") in sset):
                comps.append(c)
    if comps:
        fig.text(0.5, 0.26,
                 "Comparisons: " + ", ".join(f"{c['control']} vs {c['treatment']}" for c in comps),
                 ha="center", fontsize=10, color="#888888")

    t = config.get("thresholds", {})
    fig.text(0.5, 0.22,
             f"Thresholds: mod_prob >= {t.get('modification_probability',191)/255:.2f}, "
             f"coverage >= {t.get('min_coverage',10)}, FDR < {t.get('fdr_threshold',0.05)}, "
             f"|delta stoich| >= {t.get('min_delta_stoichiometry',0.1)}",
             ha="center", fontsize=9, color="#888888")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_summary(pdf, results):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Summary Statistics", ha="center", fontsize=20, fontweight="bold")

    diff = results["differential"]
    sig = results["significant"]
    m = results.get("model_metrics", {})
    pw = results.get("pathway", pd.DataFrame())

    lines = []
    if not diff.empty: lines.append(f"Total sites tested:      {len(diff):,}")
    if not sig.empty:
        lines.append(f"Significant sites:       {len(sig):,}")
        if "direction" in sig.columns:
            lines.append(f"  - Increased:           {(sig['direction']=='increased').sum():,}")
            lines.append(f"  - Decreased:           {(sig['direction']=='decreased').sum():,}")
    if m:
        lines.append(f"")
        lines.append(f"ML Model:                {m.get('model','N/A')}")
        lines.append(f"ROC AUC:                 {m.get('roc_auc','N/A'):.4f}" if isinstance(m.get('roc_auc'), float) else f"ROC AUC:                 {m.get('roc_auc','N/A')}")
        lines.append(f"Precision (class 1):     {m.get('precision_1','N/A'):.4f}" if isinstance(m.get('precision_1'), float) else "")
        lines.append(f"Recall (class 1):        {m.get('recall_1','N/A'):.4f}" if isinstance(m.get('recall_1'), float) else "")
    if not pw.empty and "significant" in pw.columns:
        lines.append(f"")
        lines.append(f"Enriched pathways:       {int(pw['significant'].sum())}")

    fig.text(0.15, 0.78, "\n".join(lines), fontsize=14, va="top",
             fontfamily="monospace", linespacing=1.8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_stoichiometry(pdf, results):
    df = results["stoichiometry"]
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=PAGE)
    fig.suptitle("Modification Stoichiometry Distribution", fontsize=16, fontweight="bold", y=0.98)

    for sample in sorted(df["sample"].unique()):
        sub = df[df["sample"] == sample]
        if "mod_type" in sub.columns: sub = sub[sub["mod_type"] == "m6A"]
        axes[0].hist(sub["stoichiometry"], bins=60, alpha=0.6, label=sample, density=True)
    axes[0].set_xlabel("Stoichiometry")
    axes[0].set_ylabel("Density")
    axes[0].set_title("m6A Stoichiometry")
    axes[0].legend(fontsize=9)

    for sample in sorted(df["sample"].unique()):
        sub = df[df["sample"] == sample]
        axes[1].hist(sub["total_reads"].clip(upper=5000), bins=60, alpha=0.6, label=sample, density=True)
    axes[1].set_xlabel("Coverage (reads)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Read Coverage per Site")
    axes[1].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_volcano(pdf, results):
    df = results["differential"]
    if df.empty or "delta_stoichiometry" not in df.columns: return

    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("Volcano Plot — Differential RNA Modification", fontsize=16, fontweight="bold", y=0.98)

    if "fdr" in df.columns:
        df["_y"] = -np.log10(df["fdr"].clip(lower=1e-300))
        y_label = "-log10(FDR)"
    else:
        df["_y"] = -np.log10(df["pvalue"].clip(lower=1e-300))
        y_label = "-log10(p-value)"

    colors = {"unchanged": "#cccccc", "increased": "#e74c3c", "decreased": "#3498db"}
    if "direction" in df.columns:
        for d, c in colors.items():
            mask = df["direction"] == d
            ax.scatter(df.loc[mask, "delta_stoichiometry"], df.loc[mask, "_y"],
                       c=c, alpha=0.4, s=6, label=d, rasterized=True)
    else:
        ax.scatter(df["delta_stoichiometry"], df["_y"], c="#cccccc", alpha=0.4, s=6, rasterized=True)

    ax.axhline(-np.log10(0.05), color="grey", ls="--", alpha=0.5, lw=0.8)
    ax.axvline(0.10, color="grey", ls="--", alpha=0.5, lw=0.8)
    ax.axvline(-0.10, color="grey", ls="--", alpha=0.5, lw=0.8)
    ax.set_xlabel("Delta Stoichiometry (treatment - control)")
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_shap(pdf, results):
    df = results.get("shap", pd.DataFrame())
    if df.empty: return

    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("SHAP Feature Importance (Global)", fontsize=16, fontweight="bold", y=0.98)
    top = df.sort_values("mean_abs_shap", ascending=True).tail(20)
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2196F3", edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 20 Features by SHAP Importance")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_shap_table(pdf, results):
    df = results.get("shap", pd.DataFrame())
    if df.empty: return
    display = df[["feature", "mean_abs_shap", "rank"]].copy() if "rank" in df.columns else df[["feature", "mean_abs_shap"]].copy()
    display["mean_abs_shap"] = display["mean_abs_shap"].apply(lambda x: f"{x:.4f}")
    _draw_table(pdf, "SHAP Feature Importance — Full Table", display)


def page_model(pdf, results):
    roc = results.get("roc", pd.DataFrame())
    imp = results.get("importances", pd.DataFrame())
    m = results.get("model_metrics", {})
    if roc.empty and imp.empty: return

    fig, axes = plt.subplots(1, 2, figsize=PAGE)
    fig.suptitle("Machine Learning Model Performance", fontsize=16, fontweight="bold", y=0.98)

    if not roc.empty and "fpr" in roc.columns:
        auc = f"{m.get('roc_auc', '?'):.4f}" if isinstance(m.get("roc_auc"), float) else str(m.get("roc_auc", "?"))
        axes[0].plot(roc["fpr"], roc["tpr"], color="#4f46e5", lw=2,
                     label=f'{m.get("model","Model")} (AUC={auc})')
        axes[0].plot([0,1], [0,1], "k--", alpha=0.3, lw=1)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend(fontsize=9)

    if not imp.empty:
        top = imp.sort_values("importance", ascending=True).tail(15)
        axes[1].barh(top["feature"], top["importance"], color="#7c3aed", edgecolor="none")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Built-in Feature Importance (Top 15)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_pathway_chart(pdf, results):
    df = results.get("pathway", pd.DataFrame())
    if df.empty or "pvalue" not in df.columns: return

    fig, ax = plt.subplots(figsize=PAGE)
    fig.suptitle("Pathway Enrichment Analysis", fontsize=16, fontweight="bold", y=0.98)

    df = df.sort_values("pvalue")
    df["_y"] = -np.log10(df["pvalue"].clip(lower=1e-300))
    colors = ["#e74c3c" if s else "#95a5a6" for s in df.get("significant", [False]*len(df))]

    ax.barh(df["pathway"], df["_y"], color=colors, edgecolor="none")
    ax.axvline(-np.log10(0.05), color="grey", ls="--", alpha=0.5, lw=0.8)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["_y"] + 0.05, i,
                f"{int(row.get('overlap_count',0))}/{int(row.get('pathway_size',0))}",
                va="center", fontsize=9)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Curated Pathway Enrichment")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_pathway_table(pdf, results):
    """Full pathway table with ALL gene names visible."""
    df = results.get("pathway", pd.DataFrame())
    if df.empty or "pathway" not in df.columns: return

    df = df.sort_values("pvalue").copy()

    # Build table with gene names on separate rows for readability
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.96, "Pathway Enrichment — Detailed Results with Gene Names",
             ha="center", fontsize=16, fontweight="bold")

    # Format the data
    rows = []
    for _, row in df.iterrows():
        genes = str(row.get("overlap_genes", ""))
        # Wrap gene names to ~60 chars per line
        wrapped = textwrap.fill(genes, width=65) if genes else "—"
        pval = f"{row['pvalue']:.4f}" if pd.notna(row.get("pvalue")) else "N/A"
        fdr = f"{row['fdr']:.4f}" if pd.notna(row.get("fdr")) else "N/A"
        fe = f"{row['fold_enrichment']:.2f}" if pd.notna(row.get("fold_enrichment")) else "N/A"
        sig = "YES" if row.get("significant", False) else "no"

        rows.append([
            row["pathway"],
            f"{int(row.get('overlap_count', 0))}/{int(row.get('pathway_size', 0))}",
            fe, pval, fdr, sig, wrapped
        ])

    col_labels = ["Pathway", "Overlap", "Fold", "p-value", "FDR", "Sig?", "Genes"]

    ax = fig.add_axes([0.02, 0.03, 0.96, 0.88])
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)  # taller rows for wrapped text

    # Column widths: make Genes column much wider
    col_widths = [0.14, 0.06, 0.06, 0.07, 0.07, 0.05, 0.55]
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_widths[col])
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")

        # Highlight significant pathways
        if row > 0:
            try:
                sig_val = rows[row - 1][5]
                if sig_val == "YES":
                    cell.set_facecolor("#ffeaea" if row % 2 == 1 else "#ffe0e0")
            except (IndexError):
                pass

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_significant_sites(pdf, results):
    sig = results.get("significant", pd.DataFrame())
    if sig.empty: return

    cols = [c for c in ["gene_name", "reference_name", "ref_pos", "mod_type",
                         "stoichiometry_ctrl", "stoichiometry_treat",
                         "delta_stoichiometry", "fdr", "direction"] if c in sig.columns]
    top = sig.head(30)[cols].copy()

    for col in top.select_dtypes(include=[np.number]).columns:
        top[col] = top[col].apply(lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.0f}")

    labels = {"gene_name": "Gene", "reference_name": "Chr", "ref_pos": "Position",
              "mod_type": "Mod", "stoichiometry_ctrl": "Stoich Ctrl",
              "stoichiometry_treat": "Stoich Treat", "delta_stoichiometry": "Delta Stoich",
              "fdr": "FDR", "direction": "Direction"}
    top = top.rename(columns=labels)

    _draw_table(pdf, f"Top 30 Significant Modification Sites (of {len(sig):,} total)", top)


def page_significant_sites_extended(pdf, results):
    """Additional significant sites table: sites 31-60."""
    sig = results.get("significant", pd.DataFrame())
    if len(sig) <= 30: return

    cols = [c for c in ["gene_name", "reference_name", "ref_pos",
                         "stoichiometry_ctrl", "stoichiometry_treat",
                         "delta_stoichiometry", "fdr", "direction"] if c in sig.columns]
    batch = sig.iloc[30:60][cols].copy()

    for col in batch.select_dtypes(include=[np.number]).columns:
        batch[col] = batch[col].apply(lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.0f}")

    labels = {"gene_name": "Gene", "reference_name": "Chr", "ref_pos": "Position",
              "stoichiometry_ctrl": "Stoich Ctrl", "stoichiometry_treat": "Stoich Treat",
              "delta_stoichiometry": "Delta Stoich", "fdr": "FDR", "direction": "Direction"}
    batch = batch.rename(columns=labels)

    _draw_table(pdf, f"Significant Sites 31–60 (of {len(sig):,} total)", batch)


def page_stoich_summary_table(pdf, results):
    """Per-sample stoichiometry summary table."""
    df = results["stoichiometry"]
    if df.empty: return

    rows = []
    for sample in sorted(df["sample"].unique()):
        sub = df[df["sample"] == sample]
        for mt in sorted(sub["mod_type"].unique()) if "mod_type" in sub.columns else ["all"]:
            s = sub[sub["mod_type"] == mt] if "mod_type" in sub.columns else sub
            rows.append([
                sample, mt, f"{len(s):,}",
                f"{s['stoichiometry'].mean():.4f}",
                f"{s['stoichiometry'].median():.4f}",
                f"{s['total_reads'].median():.0f}",
                f"{s['total_reads'].mean():.0f}",
                f"{s['reference_name'].nunique():,}" if "reference_name" in s.columns else "—",
            ])

    tdf = pd.DataFrame(rows, columns=[
        "Sample", "Mod Type", "Sites", "Mean Stoich", "Median Stoich",
        "Median Cov", "Mean Cov", "Unique Refs"
    ])
    _draw_table(pdf, "Stoichiometry Summary per Sample", tdf)


def page_metatranscript_all(pdf, results):
    """Metatranscript distribution — all tested sites with stacked composition view."""
    df = results.get("metatranscript_all", pd.DataFrame())
    if df.empty: return
    df = df[df["region"].isin(["5UTR", "CDS", "3UTR"])].copy()
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=PAGE)
    fig.suptitle("Metatranscript Position — All Tested Sites", fontsize=16, fontweight="bold", y=0.98)

    # Left: density histogram (all sites, single colour, showing landscape)
    axes[0].hist(df["metatranscript_pos"].dropna(), bins=60,
                 alpha=0.85, density=True, color="#4f46e5", edgecolor="none")
    axes[0].axvline(1.0, color="black", ls="--", lw=1.0, label="5'UTR|CDS")
    axes[0].axvline(2.0, color="black", ls=":",  lw=1.0, label="CDS|3'UTR")
    axes[0].legend(fontsize=8)
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xticklabels(["5' end\n(0)", "5'UTR|\nCDS\n(1)", "CDS|\n3'UTR\n(2)", "3' end\n(3)"])
    axes[0].set_xlabel("Metatranscript position")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Position density — all tested sites")

    # Right: stacked bar showing direction composition per region
    regions = ["5UTR", "CDS", "3UTR"]
    dir_colors = {"increased": "#e74c3c", "decreased": "#3498db", "unchanged": "#cccccc"}
    directions = ["increased", "decreased", "unchanged"]
    if "direction" not in df.columns:
        df["direction"] = "unchanged"
    bottoms = np.zeros(len(regions))
    for direction in directions:
        vals = [
            len(df[(df["region"] == r) & (df["direction"] == direction)])
            for r in regions
        ]
        axes[1].bar(regions, vals, bottom=bottoms,
                           color=dir_colors[direction], label=direction, edgecolor="none")
        bottoms += np.array(vals, dtype=float)
    # total labels on top
    for i, total in enumerate(bottoms):
        if total > 0:
            axes[1].text(i, total + max(bottoms) * 0.01, f"{int(total):,}",
                         ha="center", va="bottom", fontsize=9)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_xlabel("Transcript region")
    axes[1].set_ylabel("Number of sites")
    axes[1].set_title("Region composition by direction")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_metatranscript_sig(pdf, results):
    """Metatranscript distribution — significant vs all sites overlay + enrichment bar."""
    df_sig = results.get("metatranscript_sig", pd.DataFrame())
    if df_sig.empty:
        return
    df_all = results.get("metatranscript_all", pd.DataFrame())

    regions = ["5UTR", "CDS", "3UTR"]
    sig_ann = df_sig[df_sig["region"].isin(regions)].copy()

    if not sig_ann.empty:
        fig, axes = plt.subplots(1, 2, figsize=PAGE)
        fig.suptitle("Metatranscript Position — Significant Sites vs All Sites",
                     fontsize=16, fontweight="bold", y=0.98)

        # Left: significant (red) overlaid on grey all-sites background
        if not df_all.empty:
            all_ann = df_all[df_all["region"].isin(regions)]
            axes[0].hist(all_ann["metatranscript_pos"].dropna(), bins=60,
                         alpha=0.45, density=True, color="#aaaaaa", label="All sites")
        dir_colors = {"increased": "#e74c3c", "decreased": "#3498db"}
        if "direction" in sig_ann.columns:
            for direction, grp in sig_ann.groupby("direction"):
                axes[0].hist(grp["metatranscript_pos"].dropna(), bins=40,
                             alpha=0.75, density=True, label=f"Sig ({direction})",
                             color=dir_colors.get(direction, "#e74c3c"))
        else:
            axes[0].hist(sig_ann["metatranscript_pos"].dropna(), bins=40,
                         alpha=0.75, density=True, color="#e74c3c", label="Significant")
        axes[0].axvline(1.0, color="black", ls="--", lw=1.0)
        axes[0].axvline(2.0, color="black", ls=":",  lw=1.0)
        axes[0].legend(fontsize=8)
        axes[0].set_xticks([0, 1, 2, 3])
        axes[0].set_xticklabels(["5' end\n(0)", "5'UTR|\nCDS\n(1)", "CDS|\n3'UTR\n(2)", "3' end\n(3)"])
        axes[0].set_xlabel("Metatranscript position")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Significant sites overlaid on all-sites background")

        # Right: % per region — all sites vs significant (enrichment view)
        all_pct, sig_pct = [], []
        all_ann2 = df_all[df_all["region"].isin(regions)] if not df_all.empty else sig_ann
        for reg in regions:
            n_all = len(all_ann2[all_ann2["region"] == reg]) if not df_all.empty else 0
            n_sig = len(sig_ann[sig_ann["region"] == reg])
            all_pct.append(100 * n_all / max(len(all_ann2), 1))
            sig_pct.append(100 * n_sig / max(len(sig_ann), 1))

        x = np.arange(len(regions))
        w = 0.35
        axes[1].bar(x - w/2, all_pct, w, color="#aaaaaa", label="All sites %", edgecolor="none")
        axes[1].bar(x + w/2, sig_pct, w, color="#e74c3c", label="Significant %", edgecolor="none")
        for i, (a, s) in enumerate(zip(all_pct, sig_pct)):
            axes[1].text(i - w/2, a + 0.5, f"{a:.1f}%", ha="center", fontsize=8)
            axes[1].text(i + w/2, s + 0.5, f"{s:.1f}%", ha="center", fontsize=8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regions)
        axes[1].set_xlabel("Transcript region")
        axes[1].set_ylabel("% of sites in region")
        axes[1].set_title("Region enrichment: significant vs all")
        axes[1].legend(fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Table of top 30 significant sites with region annotation
    cols = [c for c in ["gene_name", "reference_name", "ref_pos", "region",
                         "metatranscript_pos", "delta_stoichiometry", "fdr", "direction"]
            if c in df_sig.columns]
    top = df_sig.head(30)[cols].copy()
    for col in top.select_dtypes(include=[np.number]).columns:
        top[col] = top[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    labels = {"gene_name": "Gene", "reference_name": "Chr", "ref_pos": "Position",
              "region": "Region", "metatranscript_pos": "Meta Pos",
              "delta_stoichiometry": "Δ Stoich", "fdr": "FDR", "direction": "Direction"}
    top = top.rename(columns=labels)
    _draw_table(pdf, f"Top 30 Significant Sites with Transcript Region (of {len(df_sig):,} total)", top)


# ── Main ────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate PDF report")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None,
                        help="Path to results folder, e.g. results/results_A375")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: <results-dir>/report/epitranscriptome_report.pdf)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = config.get("output", {}).get("base_dir", "results")
    results_dir = os.path.abspath(results_dir)
    print(f"Loading results from: {results_dir}")

    results = load_results(results_dir)

    out_dir = os.path.join(results_dir, "report")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, "epitranscriptome_report.pdf")

    print(f"Generating report: {out_path}")

    page_num = 0
    with PdfPages(out_path) as pdf:
        # Title page needs both config (for thresholds/comparisons) and
        # results (for the actual sample names that ran).
        print(f"  Page 1: Title page")
        page_title(pdf, config, results)
        page_num = 1
        pages = [
            ("Summary statistics",         page_summary,                  results),
            ("Stoichiometry summary table", page_stoich_summary_table,    results),
            ("Stoichiometry distributions", page_stoichiometry,           results),
            ("Volcano plot",               page_volcano,                  results),
            ("SHAP importance chart",      page_shap,                     results),
            ("SHAP importance table",      page_shap_table,               results),
            ("Model performance",          page_model,                    results),
            ("Pathway enrichment chart",   page_pathway_chart,            results),
            ("Pathway enrichment table",   page_pathway_table,            results),
            ("Top 30 significant sites",      page_significant_sites,          results),
            ("Significant sites 31-60",       page_significant_sites_extended,  results),
            ("Metatranscript — all sites",    page_metatranscript_all,          results),
            ("Metatranscript — significant",  page_metatranscript_sig,          results),
        ]

        for name, func, data in pages:
            page_num += 1
            print(f"  Page {page_num}: {name}")
            try:
                func(pdf, data)
            except Exception as e:
                print(f"    WARNING: {e}")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nReport saved: {out_path}")
    print(f"  Pages: {page_num}")
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()