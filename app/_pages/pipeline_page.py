"""Pipeline Runner page — detached background process, survives browser sleep/refresh."""

import os
import sys
import json
import time
import subprocess
import yaml
import streamlit as st

RUNTIME_CONFIG = "runtime_config.yaml"
STATE_FILE = "pipeline_state.json"
DONE_FILE = "pipeline_done.json"
LOG_FILE = "pipeline_run.log"

STEP_NAMES = [
    "Extract modifications from BAM",
    "Compute site-level stoichiometry",
    "Differential modification analysis",
    "Construct biological features",
    "Train ML models",
    "SHAP analysis",
    "Pathway enrichment",
    "Metatranscript position analysis",
]

RESULT_DIRS = {
    1: "01_extraction", 2: "02_stoichiometry", 3: "03_differential",
    4: "04_features",   5: "05_model",         6: "06_shap",
    7: "07_pathway",    8: "08_metatranscript",
}


def _get_project_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_config_path(pipeline_dir):
    runtime = os.path.join(pipeline_dir, RUNTIME_CONFIG)
    return runtime if os.path.exists(runtime) else None


def _get_base_dir(pipeline_dir):
    config_path = _get_config_path(pipeline_dir)
    if not config_path:
        return os.path.join(pipeline_dir, "results")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("output", {}).get("base_dir", "./results")
    if not os.path.isabs(base):
        base = os.path.join(pipeline_dir, base)
    return base


def _is_pid_running(pid):
    """Check if a process is still alive (cross-platform)."""
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.wintypes.DWORD()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    else:
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _launch_background(pipeline_dir, from_step, to_step, method):
    """Launch run_pipeline.py as a detached process independent of the browser."""
    log_path = os.path.join(pipeline_dir, LOG_FILE)
    state_path = os.path.join(pipeline_dir, STATE_FILE)
    done_path = os.path.join(pipeline_dir, DONE_FILE)

    if os.path.exists(done_path):
        os.remove(done_path)

    run_script = os.path.join(pipeline_dir, "run_pipeline.py")
    cmd = [
        sys.executable, "-u", run_script,
        "--from-step", str(from_step),
        "--to-step",   str(to_step),
        "--method",    method,
    ]

    log_f = open(log_path, "w", buffering=1)

    if sys.platform == "win32":
        proc = subprocess.Popen(
            cmd, cwd=pipeline_dir,
            stdout=log_f, stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            cmd, cwd=pipeline_dir,
            stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    state = {
        "pid":        proc.pid,
        "from_step":  from_step,
        "to_step":    to_step,
        "method":     method,
        "started":    time.time(),
        "log":        log_path,
    }
    with open(state_path, "w") as f:
        json.dump(state, f)

    return state


def _get_state(pipeline_dir):
    path = os.path.join(pipeline_dir, STATE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_done(pipeline_dir):
    path = os.path.join(pipeline_dir, DONE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_log_tail(log_path, n=50):
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return "(log not yet available)"


def _kill_pipeline(pid):
    try:
        import psutil
        psutil.Process(int(pid)).terminate()
        return
    except Exception:
        pass
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(int(pid), 15)
    except Exception:
        pass


def _step_status_icons(base_dir, running_state, is_running):
    """Return list of (icon, name) for all 8 steps."""
    icons = []
    for i, name in enumerate(STEP_NAMES):
        step_num = i + 1
        rdir = os.path.join(base_dir, RESULT_DIRS[step_num])
        has_results = (
            os.path.isdir(rdir)
            and len([f for f in os.listdir(rdir) if not f.endswith(".log")]) > 1
        )
        if has_results:
            icon = "✅"
        elif is_running and running_state and (
            running_state["from_step"] <= step_num <= running_state["to_step"]
        ):
            icon = "⏳"
        else:
            icon = "⬜"
        icons.append((icon, name))
    return icons


def _render_step_list(base_dir, state, is_running):
    for step_num, (icon, name) in enumerate(_step_status_icons(base_dir, state, is_running), 1):
        st.markdown(f"{icon} **Step {step_num}:** {name}")


def _render_run_controls(pipeline_dir):
    st.markdown("Select which steps to run.")
    col1, col2 = st.columns([3, 1])
    with col1:
        run_from = st.selectbox(
            "Start from step",
            options=range(1, 9),
            format_func=lambda x: f"Step {x}: {STEP_NAMES[x - 1]}",
            index=0,
            key="run_from",
        )
        run_to = st.selectbox(
            "Run through step",
            options=range(run_from, 9),
            format_func=lambda x: f"Step {x}: {STEP_NAMES[x - 1]}",
            index=min(7, 8 - run_from),
            key="run_to",
        )
    with col2:
        st.markdown("")
        st.markdown("")
        method = st.selectbox(
            "Statistical method",
            ["fisher", "logistic"],
            help="For Step 3",
            key="method",
        )

    if st.button(
        f"▶️  Run steps {run_from}–{run_to}",
        type="primary",
        use_container_width=True,
    ):
        state = _launch_background(pipeline_dir, run_from, run_to, method)
        st.success(
            f"Pipeline launched (PID {state['pid']}). "
            "It will keep running even if you close or refresh this page."
        )
        time.sleep(1)
        st.rerun()


def render():
    st.header("Run pipeline")

    pipeline_dir = st.session_state.get("pipeline_dir", "") or _get_project_dir()
    st.session_state.pipeline_dir = pipeline_dir

    config_path = _get_config_path(pipeline_dir)
    if not config_path:
        st.error(
            "No saved configuration found. "
            "Go to **Upload & Configure**, fill in your samples, and click **Save configuration** first."
        )
        return

    base_dir = _get_base_dir(pipeline_dir)
    st.caption(
        f"Using config: `{os.path.basename(config_path)}` · "
        f"Output: `{os.path.basename(base_dir)}/`"
    )

    state = _get_state(pipeline_dir)
    is_running = state is not None and _is_pid_running(state["pid"])
    done = _get_done(pipeline_dir)

    # ── Step status ─────────────────────────────────────────
    st.divider()
    _render_step_list(base_dir, state, is_running)
    st.divider()

    # ── Running state ────────────────────────────────────────
    if is_running:
        elapsed = int(time.time() - state.get("started", time.time()))
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        st.info(
            f"Pipeline running — steps {state['from_step']}–{state['to_step']} · "
            f"Elapsed: {h:02d}:{m:02d}:{s:02d} · PID {state['pid']}"
        )

        log_path = state.get("log", os.path.join(pipeline_dir, LOG_FILE))
        st.code(_read_log_tail(log_path, 40), language="text")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh log", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("⏹ Stop pipeline", use_container_width=True, type="secondary"):
                _kill_pipeline(state["pid"])
                st.warning("Stop signal sent.")
                time.sleep(1)
                st.rerun()

        # Auto-refresh every 5 s while running
        time.sleep(5)
        st.rerun()
        return

    # ── Finished state ───────────────────────────────────────
    if done is not None:
        if done.get("success"):
            st.success("Pipeline complete! Go to **Results** to view output.")
        else:
            failed = done.get("failed_step", "?")
            st.error(f"Pipeline failed at step {failed}. Check the log below.")

        log_path = (state or {}).get("log", os.path.join(pipeline_dir, LOG_FILE))
        with st.expander("View full run log"):
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    st.code(f.read(), language="text")
            except Exception:
                st.write("Log not available.")

        st.divider()
        st.subheader("Run again")

    # ── Run controls (always shown when not running) ─────────
    _render_run_controls(pipeline_dir)
