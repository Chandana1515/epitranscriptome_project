"""
Ollama LLM Integration
======================
Handles communication with a local Ollama instance for
AI-powered interpretation of pipeline results.

Context is kept compact — only summary stats and top 10 items
from each result table. No RAG needed.
"""

import json
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a bioinformatics expert helping analyze RNA modification results from A375 melanoma skin cancer cells sequenced with Oxford Nanopore.

Experimental setup:
- Cell line: A375 (human melanoma)
- Modification detected: m6A (N6-methyladenosine) via Dorado basecaller
- Comparison: normoxia (control) vs hypoxia (treatment)
- Pipeline: extraction → stoichiometry → differential analysis → XGBoost/SHAP → pathway enrichment

Key biology:
- m6A is the most common internal mRNA modification, regulated by writers (METTL3/14), erasers (FTO, ALKBH5), readers (YTHDF1/2/3)
- m6A occurs preferentially in DRACH motifs near stop codons and 3'UTRs
- Hypoxia activates HIF1A and can upregulate m6A writers
- Cold atmospheric plasma generates ROS causing oxidative stress

Answer concisely using the provided data. Cite specific numbers when available."""


def check_ollama_status():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []


def get_available_models():
    ok, models = check_ollama_status()
    return models if ok else []


def build_context_from_results(results: dict) -> str:
    """
    Build a COMPACT context string from pipeline results.
    Only sends summary statistics and top 10 items — not full tables.
    Must stay under ~2000 tokens to work with small local models.
    """
    parts = []

    # Summary stats (very compact)
    if "summary" in results:
        parts.append("SUMMARY:")
        for key, val in results["summary"].items():
            parts.append(f"  {key}: {val}")

    # Stoichiometry summary (just the numbers)
    if "stoichiometry_summary" in results:
        parts.append("\nSTOICHIOMETRY:")
        for key, val in results["stoichiometry_summary"].items():
            parts.append(f"  {key}: {val}")

    # Top 10 significant sites only
    if "significant_sites" in results and len(results["significant_sites"]) > 0:
        df = results["significant_sites"]
        parts.append(f"\nTOP 10 SIGNIFICANT SITES (of {len(df)} total):")

        cols = ["reference_name", "ref_pos", "delta_stoichiometry", "fdr"]
        # Add gene_name if available
        if "gene_name" in df.columns:
            cols = ["gene_name", "reference_name", "ref_pos", "delta_stoichiometry", "fdr"]

        available = [c for c in cols if c in df.columns]
        if available:
            top = df.head(10)[available]
            for _, row in top.iterrows():
                line = ", ".join(f"{c}={row[c]}" for c in available)
                parts.append(f"  {line}")

    # SHAP importance — top 10 features
    if "shap_importance" in results and len(results["shap_importance"]) > 0:
        df = results["shap_importance"]
        parts.append(f"\nSHAP FEATURE IMPORTANCE (top 10):")
        for _, row in df.head(10).iterrows():
            parts.append(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Model metrics (one line)
    if "model_metrics" in results:
        m = results["model_metrics"]
        parts.append(
            f"\nMODEL: {m.get('model','?')}, "
            f"AUC={m.get('roc_auc','?')}, "
            f"precision={m.get('precision_1','?')}, "
            f"recall={m.get('recall_1','?')}"
        )

    # Pathway enrichment — just pathway names and p-values
    if "pathway_enrichment" in results and len(results["pathway_enrichment"]) > 0:
        df = results["pathway_enrichment"]
        parts.append(f"\nPATHWAY ENRICHMENT:")
        for _, row in df.iterrows():
            sig = "*" if row.get("significant", False) else ""
            parts.append(
                f"  {sig}{row['pathway']}: "
                f"overlap={row.get('overlap_count','?')}/{row.get('pathway_size','?')}, "
                f"p={row.get('pvalue','?'):.4e}"
            )

    context = "\n".join(parts)

    # Safety check: truncate if still too long
    if len(context) > 3000:
        context = context[:3000] + "\n... (truncated)"

    return context


def chat_with_ollama(user_message, results, chat_history, model=DEFAULT_MODEL):
    context = build_context_from_results(results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context.strip():
        messages.append({
            "role": "user",
            "content": f"Pipeline results:\n{context}"
        })
        messages.append({
            "role": "assistant",
            "content": "I have the results. What would you like to know?"
        })

    # Only last 6 messages from history (keep context small)
    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096,
                },
            },
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"Error (HTTP {response.status_code}): {response.text[:200]}"
    except requests.ConnectionError:
        return "Cannot connect to Ollama. Make sure it's running."
    except requests.Timeout:
        return "Request timed out. Try a simpler question or a smaller model."
    except Exception as e:
        return f"Error: {str(e)}"


def stream_chat_with_ollama(user_message, results, chat_history, model=DEFAULT_MODEL):
    context = build_context_from_results(results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context.strip():
        messages.append({
            "role": "user",
            "content": f"Pipeline results:\n{context}"
        })
        messages.append({
            "role": "assistant",
            "content": "I have the results. What would you like to know?"
        })

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096,
                },
            },
            stream=True,
            timeout=300,
        )
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
        else:
            yield f"Error (HTTP {response.status_code})"
    except requests.ConnectionError:
        yield "Cannot connect to Ollama. Run `ollama serve`."
    except requests.Timeout:
        yield "Timed out. Try a simpler question."
    except Exception as e:
        yield f"Error: {str(e)}"