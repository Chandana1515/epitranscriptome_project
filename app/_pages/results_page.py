"""Results Dashboard — interactive visualization of pipeline outputs."""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from result_loader import load_all_results, get_results_overview

# All pipeline outputs live on the E: drive.
RESULTS_ROOT = r"E:\results" if os.name == "nt" else "/mnt/e/results"


def _has_pipeline_steps(path):
    """Check if a directory contains pipeline step subdirectories."""
    try:
        return any(e.startswith("0") and os.path.isdir(os.path.join(path, e))
                   for e in os.listdir(path))
    except Exception:
        return False


def _find_all_result_dirs(pipeline_dir):
    """
    Return list of (label, full_path) for every results directory found.
    Primary location: RESULTS_ROOT (E:/results). Also scans the project
    directory as a fallback for legacy runs.
    """
    found = []
    seen = set()

    def _scan(root):
        if not os.path.isdir(root):
            return
        try:
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if not os.path.isdir(full) or full in seen:
                    continue
                if name.startswith("results") and _has_pipeline_steps(full):
                    seen.add(full)
                    found.append((name, full))
        except Exception:
            pass

    _scan(RESULTS_ROOT)
    _scan(pipeline_dir)
    return found


def _get_project_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def render():
    st.header("Results dashboard")

    pipeline_dir = st.session_state.get("pipeline_dir", "") or _get_project_dir()
    st.session_state.pipeline_dir = pipeline_dir

    # ── Results folder selector ─────────────────────────────
    available = _find_all_result_dirs(pipeline_dir)

    if not available:
        st.info("No results found. Run the pipeline first.")
        return

    labels  = [label for label, _ in available]
    paths   = {label: path for label, path in available}

    selected_label = st.selectbox(
        "Select results to view",
        labels,
        index=0,
        key="results_dir_select",
    )
    results_dir = paths[selected_label]

    col_refresh, _ = st.columns([1, 5])
    if col_refresh.button("Refresh results"):
        load_all_results.clear()

    # Load all results from the selected folder
    results = load_all_results(results_dir)
    st.session_state.results = results

    # ── Overview metrics ────────────────────────────────────
    summary = results.get("summary", {})

    cols = st.columns(4)
    cols[0].metric("Tested sites", summary.get("total_tested_sites", "—"))
    cols[1].metric(
        "Significant",
        summary.get("significant_sites", "—"),
    )
    raw_auc = summary.get("model_roc_auc", "—")
    auc_display = f"{float(raw_auc):.4f}" if raw_auc != "—" else "—"
    cols[2].metric("Model AUC", auc_display)
    cols[3].metric("Enriched pathways", summary.get("enriched_pathways", "—"))

    # ── Data availability ───────────────────────────────────
    overview = get_results_overview(results)
    available_tabs = []

    if "extraction_summary" in results and not results["extraction_summary"].empty:
        available_tabs.append("Extraction summary")
    if "stoichiometry" in results and not results["stoichiometry"].empty:
        available_tabs.append("Stoichiometry")
    if "differential" in results and not results["differential"].empty:
        available_tabs.append("Volcano plot")
    if "shap_importance" in results and not results["shap_importance"].empty:
        available_tabs.append("SHAP importance")
    if "feature_importances" in results and not results["feature_importances"].empty:
        available_tabs.append("Model performance")
    if "pathway_enrichment" in results and not results["pathway_enrichment"].empty:
        available_tabs.append("Pathways")
    if "significant_sites" in results and not results["significant_sites"].empty:
        available_tabs.append("Significant sites")
    if "metatranscript" in results and not results["metatranscript"].empty:
        available_tabs.append("Metatranscript")

    if not available_tabs:
        st.info("No results available yet. Run the pipeline to generate output.")

        with st.expander("Data availability check"):
            for key, status in overview.items():
                st.markdown(f"- **{key}**: {status}")
        return

    # ── Tabbed results view ─────────────────────────────────
    tabs = st.tabs(available_tabs)
    tab_map = {name: tab for name, tab in zip(available_tabs, tabs)}

    # ── Extraction summary ──────────────────────────────────
    if "Extraction summary" in tab_map:
        with tab_map["Extraction summary"]:
            render_extraction_summary(results)

    # ── Stoichiometry ───────────────────────────────────────
    if "Stoichiometry" in tab_map:
        with tab_map["Stoichiometry"]:
            render_stoichiometry(results)

    # ── Volcano plot ────────────────────────────────────────
    if "Volcano plot" in tab_map:
        with tab_map["Volcano plot"]:
            render_volcano(results)

    # ── SHAP ────────────────────────────────────────────────
    if "SHAP importance" in tab_map:
        with tab_map["SHAP importance"]:
            render_shap(results)

    # ── Model performance ───────────────────────────────────
    if "Model performance" in tab_map:
        with tab_map["Model performance"]:
            render_model(results)

    # ── Pathways ────────────────────────────────────────────
    if "Pathways" in tab_map:
        with tab_map["Pathways"]:
            render_pathways(results)

    # ── Significant sites table ─────────────────────────────
    if "Significant sites" in tab_map:
        with tab_map["Significant sites"]:
            render_significant_sites(results)

    # ── Metatranscript ──────────────────────────────────────
    if "Metatranscript" in tab_map:
        with tab_map["Metatranscript"]:
            render_metatranscript(results)


def render_extraction_summary(results):
    df = results["extraction_summary"]
    st.subheader("Step 1 — Extraction summary")

    for _, row in df.iterrows():
        st.markdown(f"#### Sample: `{row.get('sample', '?')}`")
        cols = st.columns(4)
        cols[0].metric("Total mod calls", f"{int(row.get('total_mod_calls', 0)):,}")
        cols[1].metric("Unique reads", f"{int(row.get('unique_reads', 0)):,}")
        cols[2].metric("Unique sites", f"{int(row.get('unique_sites', 0)):,}")
        cols[3].metric("Chunks", f"{int(row.get('n_chunks', 0)):,}")

    with st.expander("Full table"):
        st.dataframe(df, use_container_width=True)


def render_stoichiometry(results):
    """Stoichiometry distribution plots."""
    df = results["stoichiometry"]

    st.subheader("Modification stoichiometry distribution")

    # Filter by mod type
    mod_types = sorted(df["mod_type"].unique()) if "mod_type" in df.columns else []
    if mod_types:
        selected_mod = st.selectbox(
            "Modification type", mod_types, key="stoich_mod"
        )
        df_plot = df[df["mod_type"] == selected_mod]
    else:
        df_plot = df

    if "sample" in df_plot.columns:
        fig = px.histogram(
            df_plot,
            x="stoichiometry",
            color="sample",
            nbins=60,
            barmode="overlay",
            opacity=0.6,
            labels={"stoichiometry": "Stoichiometry", "sample": "Sample"},
            title=f"Stoichiometry distribution — {selected_mod if mod_types else 'all'}",
        )
    else:
        fig = px.histogram(
            df_plot, x="stoichiometry", nbins=60,
            title="Stoichiometry distribution",
        )

    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Coverage distribution
    if "total_reads" in df_plot.columns:
        fig2 = px.histogram(
            df_plot,
            x="total_reads",
            color="sample" if "sample" in df_plot.columns else None,
            nbins=50,
            barmode="overlay",
            opacity=0.6,
            title="Read coverage per site",
            labels={"total_reads": "Coverage (reads)"},
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)


def render_volcano(results):
    """Volcano plot of differential modification results."""
    df = results.get("volcano_data", results.get("differential", pd.DataFrame()))
    if df.empty:
        st.info("No differential data available.")
        return

    st.subheader("Volcano plot — differential modification")

    # Compute -log10 values if not present
    if "-log10_fdr" not in df.columns and "fdr" in df.columns:
        df["-log10_fdr"] = -np.log10(df["fdr"].clip(lower=1e-300))
    if "-log10_pvalue" not in df.columns and "pvalue" in df.columns:
        df["-log10_pvalue"] = -np.log10(df["pvalue"].clip(lower=1e-300))

    y_col = "-log10_fdr" if "-log10_fdr" in df.columns else "-log10_pvalue"
    x_col = "delta_stoichiometry"

    if x_col not in df.columns or y_col not in df.columns:
        st.warning("Required columns not found for volcano plot.")
        return

    # Color by significance
    if "direction" in df.columns:
        color_col = "direction"
        color_map = {
            "unchanged": "#c0c0c0",
            "increased": "#e74c3c",
            "decreased": "#3498db",
        }
    elif "is_significant" in df.columns:
        df["_color"] = df["is_significant"].map({True: "significant", False: "not significant"})
        color_col = "_color"
        color_map = {"significant": "#e74c3c", "not significant": "#c0c0c0"}
    else:
        color_col = None
        color_map = None

    hover_cols = ["reference_name", "ref_pos", "mod_type"]
    hover_cols = [c for c in hover_cols if c in df.columns]

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_map=color_map,
        hover_data=hover_cols,
        opacity=0.5,
        title="Differential RNA modification",
        labels={
            x_col: "Δ Stoichiometry (treatment − control)",
            y_col: "-log₁₀(FDR)",
        },
    )

    # Add threshold lines
    fdr_thresh = 0.05
    delta_thresh = 0.10
    fig.add_hline(y=-np.log10(fdr_thresh), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=delta_thresh, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=-delta_thresh, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(height=550)
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)


def render_shap(results):
    """SHAP feature importance visualization."""
    df = results["shap_importance"]
    if df.empty:
        return

    st.subheader("SHAP feature importance")

    # Bar chart
    df_sorted = df.sort_values("mean_abs_shap", ascending=True).tail(20)

    fig = px.bar(
        df_sorted,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Global SHAP importance (top 20 features)",
        labels={
            "mean_abs_shap": "Mean |SHAP value|",
            "feature": "Feature",
        },
        color="mean_abs_shap",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Raw data table
    with st.expander("SHAP values table"):
        st.dataframe(df, use_container_width=True)


def render_model(results):
    """Model performance metrics and feature importances."""
    st.subheader("ML model performance")

    metrics = results.get("model_metrics", {})
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", metrics.get("model", "—"))
        col2.metric("ROC AUC", metrics.get("roc_auc", "—"))
        col3.metric("Precision", metrics.get("precision_1", "—"))
        col4.metric("Recall", metrics.get("recall_1", "—"))

    # ROC curve
    roc = results.get("roc_curve", pd.DataFrame())
    if not roc.empty and "fpr" in roc.columns and "tpr" in roc.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            mode="lines",
            name=f"Model (AUC={metrics.get('roc_auc', '?')})",
            line=dict(color="#4f46e5", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
        ))
        fig.update_layout(
            title="ROC curve",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance from model
    imp = results.get("feature_importances", pd.DataFrame())
    if not imp.empty:
        df_sorted = imp.sort_values("importance", ascending=True).tail(15)
        fig = px.bar(
            df_sorted,
            x="importance",
            y="feature",
            orientation="h",
            title="Built-in feature importance (top 15)",
            color="importance",
            color_continuous_scale="Purples",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_pathways(results):
    """Pathway enrichment results."""
    df = results["pathway_enrichment"]
    if df.empty:
        return

    st.subheader("Pathway enrichment analysis")

    if "pvalue" in df.columns:
        df_plot = df.copy()
        df_plot["-log10_pvalue"] = -np.log10(df_plot["pvalue"].clip(lower=1e-300))
        df_plot = df_plot.sort_values("pvalue")

        # Color by significance
        if "significant" in df_plot.columns:
            df_plot["_sig"] = df_plot["significant"].map(
                {True: "Significant", False: "Not significant"}
            )
            color_col = "_sig"
            color_map = {"Significant": "#e74c3c", "Not significant": "#95a5a6"}
        else:
            color_col = None
            color_map = None

        fig = px.bar(
            df_plot,
            x="-log10_pvalue",
            y="pathway",
            orientation="h",
            color=color_col,
            color_discrete_map=color_map,
            title="Curated pathway enrichment",
            labels={
                "-log10_pvalue": "-log₁₀(p-value)",
                "pathway": "Pathway",
            },
        )
        fig.add_vline(x=-np.log10(0.05), line_dash="dash", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Table with details
    display_cols = [
        c for c in [
            "pathway", "pathway_size", "overlap_count",
            "fold_enrichment", "pvalue", "fdr", "significant", "overlap_genes",
        ]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True)


def render_significant_sites(results):
    """Table of significantly altered modification sites."""
    df = results["significant_sites"]
    if df.empty:
        st.info("No significant sites found.")
        return

    st.subheader(f"Significant modification sites ({len(df)} sites)")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if "mod_type" in df.columns:
            mod_filter = st.multiselect(
                "Modification type",
                df["mod_type"].unique(),
                default=list(df["mod_type"].unique()),
            )
            df = df[df["mod_type"].isin(mod_filter)]
    with col2:
        if "direction" in df.columns:
            dir_filter = st.multiselect(
                "Direction",
                df["direction"].unique(),
                default=list(df["direction"].unique()),
            )
            df = df[df["direction"].isin(dir_filter)]

    # Show table
    display_cols = [
        c for c in [
            "reference_name", "ref_pos", "mod_type", "strand",
            "stoichiometry_ctrl", "stoichiometry_treat",
            "delta_stoichiometry", "pvalue", "fdr", "direction",
        ]
        if c in df.columns
    ]

    st.dataframe(
        df[display_cols].head(200),
        use_container_width=True,
        height=500,
    )

    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        "Download full table (CSV)",
        csv,
        "significant_sites.csv",
        "text/csv",
    )


def render_metatranscript(results):
    """Metatranscript position distribution of modification sites."""
    df = results["metatranscript"]
    if df.empty:
        st.info("No metatranscript data available.")
        return

    st.subheader("Metatranscript position analysis")
    st.caption(
        "Position of modifications relative to transcript structure. "
        "X-axis: 0–1 = 5'UTR, 1–2 = CDS, 2–3 = 3'UTR."
    )

    # Filter to annotated sites only
    df = df[df["region"].isin(["5UTR", "CDS", "3UTR"])].copy()
    if df.empty:
        st.warning("No sites were mapped to transcript regions.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        mod_types = sorted(df["mod_type"].unique()) if "mod_type" in df.columns else []
        selected_mod = st.selectbox("Modification type", ["All"] + mod_types, key="meta_mod")
    with col2:
        show = st.radio(
            "Show", ["All tested sites", "Significant only"], horizontal=True, key="meta_show"
        )

    plot_df = df.copy()
    if selected_mod != "All" and "mod_type" in plot_df.columns:
        plot_df = plot_df[plot_df["mod_type"] == selected_mod]
    if show == "Significant only" and "is_significant" in plot_df.columns:
        plot_df = plot_df[plot_df["is_significant"] == True]

    if plot_df.empty:
        st.warning("No sites match current filters.")
        return

    # Density histogram — colour by direction if available
    color_col = "direction" if "direction" in plot_df.columns else None
    color_map  = {"increased": "#e74c3c", "decreased": "#3498db", "unchanged": "#c0c0c0"}

    fig = go.Figure()

    groups = plot_df.groupby(color_col) if color_col else [("all", plot_df)]
    colors = {"increased": "#e74c3c", "decreased": "#3498db",
              "unchanged": "#c0c0c0", "all": "#6366f1"}

    for grp_name, grp in groups:
        positions = grp["metatranscript_pos"].dropna()
        fig.add_trace(go.Histogram(
            x=positions,
            name=str(grp_name),
            nbinsx=60,
            histnorm="probability density",
            opacity=0.65,
            marker_color=colors.get(str(grp_name), "#888"),
        ))

    # Zone shading
    fig.add_vrect(x0=0, x1=1, fillcolor="lightblue",  opacity=0.08, line_width=0)
    fig.add_vrect(x0=1, x1=2, fillcolor="lightyellow", opacity=0.12, line_width=0)
    fig.add_vrect(x0=2, x1=3, fillcolor="lightgreen",  opacity=0.08, line_width=0)

    # Zone boundary lines
    for x in [1.0, 2.0]:
        fig.add_vline(x=x, line_dash="dash", line_color="gray", opacity=0.5)

    # Zone labels
    for x, label in [(0.5, "5'UTR"), (1.5, "CDS"), (2.5, "3'UTR")]:
        fig.add_annotation(x=x, y=1.02, yref="paper", text=label,
                           showarrow=False, font=dict(size=13, color="gray"))

    fig.update_layout(
        barmode="overlay",
        xaxis=dict(title="Metatranscript position", range=[0, 3], tickvals=[0, 1, 2, 3]),
        yaxis_title="Density",
        height=450,
        legend_title="Direction",
        title=f"Modification distribution across transcript — {selected_mod}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Region summary bar chart
    if "metatranscript_summary" in results and not results["metatranscript_summary"].empty:
        st.subheader("Sites per region")
        sum_df = results["metatranscript_summary"].copy()
        if selected_mod != "All" and "mod_type" in sum_df.columns:
            sum_df = sum_df[sum_df["mod_type"] == selected_mod]

        sum_df = sum_df[sum_df["region"].isin(["5UTR", "CDS", "3UTR"])]
        fig2 = px.bar(
            sum_df,
            x="region", y="significant",
            color="mod_type" if "mod_type" in sum_df.columns else None,
            barmode="group",
            labels={"significant": "Significant sites", "region": "Region"},
            title="Significant sites per transcript region",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Download
    st.download_button(
        "Download annotated sites (CSV)",
        plot_df.to_csv(index=False),
        "metatranscript_positions.csv",
        "text/csv",
    )
