"""
AutoForecast Pro — Streamlit Application
Run: streamlit run app.py
"""
import sys, os, json, io, warnings, time
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoForecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] { background: #0f0d1c; }
[data-testid="stSidebar"] { background: #161428 !important; border-right: 1px solid #2a2750; }
[data-testid="stHeader"] { background: transparent; }

/* Typography */
body, .stMarkdown, p, label { color: #e0daf5 !important; font-family: 'Segoe UI', system-ui, sans-serif; }
h1, h2, h3 { color: #f0ecff !important; }

/* Cards */
.af-card {
    background: #1c1a32; border: 1px solid #2a2750;
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
}
.af-card-highlight {
    background: linear-gradient(135deg, #2a1950 0%, #1c1a32 100%);
    border: 1px solid #6d28d9;
}

/* Metrics */
.af-metric { background: #1c1a32; border: 1px solid #2a2750; border-radius: 10px; padding: 14px 18px; }
.af-metric .label { font-size: 11px; color: #7c6fad; text-transform: uppercase; letter-spacing: .8px; }
.af-metric .value { font-size: 26px; font-weight: 700; color: #f0ecff; line-height: 1.1; margin-top: 4px; }
.af-metric .sub   { font-size: 11px; color: #7c6fad; margin-top: 3px; }

/* Badges */
.badge-best  { background: #6d28d9; color: #fff; font-size: 10px; padding: 2px 9px; border-radius: 20px; font-weight: 600; }
.badge-rank  { background: #1c1a32; color: #7c6fad; font-size: 10px; padding: 2px 8px; border-radius: 5px; border: 1px solid #2a2750; }

/* Sidebar section label */
.sb-section { font-size: 10px; font-weight: 600; color: #7c6fad; text-transform: uppercase;
              letter-spacing: 1px; margin: 16px 0 6px; border-bottom: 1px solid #2a2750; padding-bottom: 4px; }

/* Upload prompt hero */
.upload-hero { text-align: center; padding: 60px 40px; color: #7c6fad; }
.upload-hero h2 { font-size: 22px; color: #b8b0e0 !important; margin-bottom: 10px; }
.upload-hero p  { font-size: 14px; color: #7c6fad; line-height: 1.7; }

/* Status log */
.status-log { background: #0a0914; border: 1px solid #2a2750; border-radius: 8px;
              padding: 12px; font-family: 'Courier New', monospace; font-size: 12px;
              color: #7c6fad; height: 180px; overflow-y: auto; }
.log-ok  { color: #10b981; }
.log-err { color: #ef4444; }
.log-run { color: #a78bfa; }

/* Streamlit overrides */
.stButton > button { background: #6d28d9 !important; color: white !important;
                     border: none !important; border-radius: 8px !important;
                     font-weight: 600 !important; transition: all .15s !important; }
.stButton > button:hover { background: #7c3aed !important; transform: translateY(-1px); }
div[data-testid="stSelectbox"] > label,
div[data-testid="stMultiSelect"] > label,
div[data-testid="stNumberInput"] > label { color: #b8b0e0 !important; font-size: 13px !important; }
.stTabs [data-baseweb="tab-list"] { background: #1c1a32; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"]      { background: transparent; color: #7c6fad; border-radius: 6px; }
.stTabs [aria-selected="true"]    { background: #6d28d9 !important; color: white !important; }
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
.stProgress > div > div { background: #6d28d9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Imports from local modules ────────────────────────────────────────────────
from data.loader import (
    get_sheet_names, load_uploaded_file, auto_detect_columns,
    validate_and_prepare, split_by_sku
)
from data.cleaner import CLEANING_LABELS
from evaluation.metrics import all_metrics, METRICS
from evaluation.pipeline import run_sku_pipeline

# ─── Session state init ────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "raw_df": None, "skus": {}, "results": {},
        "run_complete": False, "log_lines": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─── Plotly theme ─────────────────────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0daf5", family="Segoe UI"),
    xaxis=dict(gridcolor="#2a2750", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#2a2750", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2750"),
)
BRAND = "#6d28d9"
ACCENT = "#f59e0b"
SUCCESS = "#10b981"
PALETTE = ["#6d28d9", "#f59e0b", "#10b981", "#ef4444", "#3b82f6", "#ec4899"]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    cfg = {}

    with st.sidebar:
        st.markdown(
            '<div style="font-size:20px;font-weight:800;color:#f0ecff;'
            'padding-bottom:14px;border-bottom:1px solid #2a2750;'
            'display:flex;align-items:center;gap:8px">'
            '📈 AutoForecast Pro</div>',
            unsafe_allow_html=True
        )

        # ── File upload ────────────────────────────────────────────────────
        st.markdown('<div class="sb-section">📂 Excel / CSV Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload your data file",
            type=["xlsx", "xls", "csv"],
            help="Upload an Excel (.xlsx/.xls) or CSV file. Must contain a date column and a numeric value column.",
            label_visibility="collapsed",
        )

        if uploaded is None:
            st.info("⬆️ Please upload an Excel (.xlsx / .xls) or CSV file to begin.")
            cfg["uploaded"] = None
            return cfg

        cfg["uploaded"] = uploaded

        # Sheet selection for Excel
        sheet_name = None
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            sheets = get_sheet_names(uploaded)
            uploaded.seek(0)
            if len(sheets) > 1:
                sheet_name = st.selectbox("Sheet", sheets)
        cfg["sheet_name"] = sheet_name

        # Load and parse
        try:
            df_raw = load_uploaded_file(uploaded, sheet_name=sheet_name)
            uploaded.seek(0)
            st.session_state["raw_df"] = df_raw
            st.success(f"✓ {len(df_raw):,} rows · {df_raw.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ {e}")
            cfg["uploaded"] = None
            return cfg

        # Column mapping
        st.markdown('<div class="sb-section">🗂 Column Mapping</div>', unsafe_allow_html=True)
        cols = df_raw.columns.tolist()
        auto_date, auto_val, auto_sku = auto_detect_columns(df_raw)

        date_col = st.selectbox("Date column (ds)", cols,
            index=cols.index(auto_date) if auto_date in cols else 0)
        val_col = st.selectbox("Value column (y)", cols,
            index=cols.index(auto_val) if auto_val in cols else min(1, len(cols)-1))
        sku_options = ["— none (single series) —"] + [c for c in cols if c not in (date_col, val_col)]
        sku_sel = st.selectbox("SKU / Group column (optional)", sku_options)
        sku_col = sku_sel if sku_sel != "— none (single series) —" else None

        cfg.update({"date_col": date_col, "val_col": val_col, "sku_col": sku_col})

        # Validate
        try:
            prepared = validate_and_prepare(df_raw, date_col, val_col, sku_col)
            skus = split_by_sku(prepared)
            st.session_state["skus"] = skus
            n_sku = len(skus)
            total = sum(len(v) for v in skus.values())
            st.caption(f"{n_sku} SKU{'s' if n_sku>1 else ''} · {total:,} total rows")
        except Exception as e:
            st.error(f"❌ Column error: {e}")
            cfg["uploaded"] = None
            return cfg

        # ── Forecast settings ──────────────────────────────────────────────
        st.markdown('<div class="sb-section">🔮 Forecast Settings</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        cfg["horizon"]   = c1.number_input("Horizon", 1, 365, 12, help="Number of future periods to forecast")
        cfg["n_test"]    = c2.number_input("Test set size", 3, 200, 6, help="Periods held out for validation")
        cfg["metric"]    = st.selectbox("Validation metric", ["SMAPE", "MAE", "RMSE", "MAPE"])
        cfg["seed"]      = st.number_input("Random seed", 0, 9999, 42)

        # ── Cleaning methods ───────────────────────────────────────────────
        st.markdown('<div class="sb-section">🧹 Cleaning Methods</div>', unsafe_allow_html=True)
        cleaning_sel = []
        if st.checkbox("No cleaning (raw data)", True): cleaning_sel.append("none")
        if st.checkbox("Hampel filter (outlier removal)", True): cleaning_sel.append("hampel")
        if st.checkbox("DWT denoising (wavelet)", False): cleaning_sel.append("dwt")
        if st.checkbox("SSA denoising", False): cleaning_sel.append("ssa")
        cfg["cleaning_methods"] = cleaning_sel or ["none"]

        # ── Models ────────────────────────────────────────────────────────
        st.markdown('<div class="sb-section">🤖 Models</div>', unsafe_allow_html=True)
        model_avail = {
            "ARIMA":          _check("statsmodels"),
            "Exp. Smoothing": _check("statsmodels"),
            "XGBoost":        _check("xgboost"),
            "Prophet":        _check("prophet"),
            "Theta":          True,
        }
        model_sel = []
        for m, avail in model_avail.items():
            lbl = m if avail else f"{m} *(not installed)*"
            default = avail and m in ("ARIMA", "Exp. Smoothing", "XGBoost", "Theta")
            if st.checkbox(lbl, value=default, disabled=not avail):
                model_sel.append(m)
        cfg["models"] = model_sel or ["Theta"]

        # ── Optimization ──────────────────────────────────────────────────
        st.markdown('<div class="sb-section">🎯 Optimization</div>', unsafe_allow_html=True)
        cfg["optim_method"] = st.selectbox("Method", ["random", "optuna", "grid"],
            format_func=lambda x: {"random": "Random search", "optuna": "Optuna (Bayesian)", "grid": "Grid search"}[x])
        cfg["n_trials"] = st.number_input("Trials per pipeline", 3, 100, 8)

        st.divider()
        cfg["can_run"] = bool(st.session_state.get("skus")) and bool(model_sel)

    return cfg


@st.cache_data(show_spinner=False)
def _check(lib):
    try: __import__(lib); return True
    except ImportError: return False


# ─── Run engine ───────────────────────────────────────────────────────────────
def run_all(cfg):
    skus = st.session_state["skus"]
    log_lines = []
    results = {}

    status_box = st.empty()
    overall_prog = st.progress(0.0, text="Starting…")
    log_area = st.empty()

    def log(msg, kind="run"):
        log_lines.append(f'<div class="log-{kind}">{msg}</div>')
        log_area.markdown(
            f'<div class="status-log">{"".join(log_lines[-20:])}</div>',
            unsafe_allow_html=True
        )

    n_skus = len(skus)
    for i, (sku_id, sku_df) in enumerate(skus.items()):
        status_box.markdown(
            f'<div class="af-card" style="padding:12px 18px">'
            f'<span style="color:#a78bfa;font-size:13px">Processing SKU:</span> '
            f'<strong style="color:#f0ecff">{sku_id}</strong> '
            f'<span style="color:#7c6fad">({i+1}/{n_skus})</span></div>',
            unsafe_allow_html=True
        )

        def status_fn(msg):
            log(msg, "run")

        try:
            sku_result = run_sku_pipeline(
                sku_id=sku_id,
                sku_df=sku_df,
                cleaning_methods=cfg["cleaning_methods"],
                model_names=cfg["models"],
                metric=cfg["metric"],
                split_method="last_n",
                n_test=cfg["n_test"],
                optim_method=cfg["optim_method"],
                n_trials=cfg["n_trials"],
                horizon=cfg["horizon"],
                seed=cfg["seed"],
                status_fn=status_fn,
            )
            results[sku_id] = sku_result
            best = sku_result.get("best")
            if best:
                log(f"✓ {sku_id}: BEST = {best['cleaning']} + {best['model']} ({cfg['metric']}={best['score']:.3f})", "ok")
            else:
                log(f"⚠ {sku_id}: No successful pipelines", "err")
        except Exception as e:
            log(f"✗ {sku_id}: {e}", "err")

        overall_prog.progress((i + 1) / n_skus, text=f"SKU {i+1}/{n_skus} complete")

    overall_prog.progress(1.0, text="✅ All complete!")
    status_box.empty()
    return results


# ─── Charts ───────────────────────────────────────────────────────────────────
def forecast_chart(train_df, test_df, fc_df, model_name, cleaning, sku_id, metric_val, metric_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_df["ds"], y=train_df["y"], name="Train",
        line=dict(color="#5b5296", width=1.5), mode="lines"
    ))
    if test_df is not None and len(test_df):
        fig.add_trace(go.Scatter(
            x=test_df["ds"], y=test_df["y"], name="Actuals (test)",
            line=dict(color=ACCENT, width=2), mode="lines+markers", marker=dict(size=4)
        ))
    if fc_df is not None and len(fc_df):
        if "yhat_upper" in fc_df and "yhat_lower" in fc_df:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_df["ds"], fc_df["ds"][::-1]]),
                y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(109,40,217,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="95% CI"
            ))
        fig.add_trace(go.Scatter(
            x=fc_df["ds"], y=fc_df["yhat"], name=f"{model_name} forecast",
            line=dict(color=BRAND, width=2.5), mode="lines"
        ))
    fig.update_layout(
        title=f"{sku_id} — {cleaning} + {model_name} ({metric_name}={metric_val:.3f})",
        hovermode="x unified", height=420, **THEME
    )
    return fig


def comparison_bar(ranked, sku_id, metric_name):
    if not ranked: return go.Figure()
    labels = [f"{r['cleaning']} + {r['model']}" for r in ranked]
    scores = [r["score"] for r in ranked]
    colors = [BRAND if i == 0 else "#2a2750" for i in range(len(scores))]
    fig = go.Figure(go.Bar(
        x=labels, y=scores, marker_color=colors,
        text=[f"{s:.3f}" for s in scores], textposition="outside",
        marker_line_color=[BRAND if i == 0 else "#4a4275" for i in range(len(scores))],
        marker_line_width=1,
    ))
    fig.update_layout(
        title=f"{sku_id} — Pipeline Comparison ({metric_name})",
        yaxis_title=metric_name, height=380, **THEME
    )
    return fig


def heatmap_chart(results_by_sku, metric_name):
    rows = []
    for sku_id, data in results_by_sku.items():
        for r in data.get("results", []):
            if r["error"] is None:
                rows.append({"SKU": sku_id, "Pipeline": f"{r['cleaning']}+{r['model']}", "Score": round(r["score"], 3)})
    if not rows: return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="SKU", columns="Pipeline", values="Score", aggfunc="min")
    fig = px.imshow(pivot, color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
                    aspect="auto", title=f"Validation {metric_name} Heatmap (lower = better)")
    fig.update_layout(height=max(300, 80*len(pivot)), **THEME)
    return fig


def feature_importance_fig(model_obj):
    if not hasattr(model_obj, "feature_importance_") or model_obj.feature_importance_ is None:
        return None
    fi = model_obj.feature_importance_.head(15)
    fig = px.bar(fi, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#2a2750", BRAND])
    fig.update_layout(title="Feature Importance (XGBoost)", height=400,
                      coloraxis_showscale=False, **THEME)
    return fig


# ─── Results rendering ────────────────────────────────────────────────────────
def render_results(results, metric_name):
    if not results:
        st.warning("No results available.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    tab_summary, tab_forecast, tab_details, tab_repro = st.tabs([
        "📊 Summary", "📈 Forecast", "🔬 Model Details", "🔁 Reproducibility"
    ])

    summary_rows = []
    for sku_id, data in results.items():
        ranked = data.get("ranked", [])
        for i, r in enumerate(ranked):
            m = r.get("metrics", {})
            summary_rows.append({
                "Rank": i + 1, "SKU": sku_id,
                "Cleaning": CLEANING_LABELS.get(r["cleaning"], r["cleaning"]),
                "Model": r["model"],
                metric_name: round(r["score"], 4),
                "MAE":   round(m.get("MAE", 0), 2),
                "RMSE":  round(m.get("RMSE", 0), 2),
                "Best?": "⭐" if i == 0 else "",
            })
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    with tab_summary:
        if not summary_df.empty:
            best_df = summary_df[summary_df["Rank"] == 1].drop(columns=["Rank"])
            st.markdown("#### Best pipeline per SKU")

            # Metric cards
            n_skus = len(results)
            successful = sum(1 for d in results.values() if d.get("best"))
            avg_score = np.mean([d["best"]["score"] for d in results.values() if d.get("best")])
            total_pipes = sum(len(d.get("results", [])) for d in results.values())

            c1, c2, c3, c4 = st.columns(4)
            for col, label, val, sub in [
                (c1, "SKUs processed", str(n_skus), "total"),
                (c2, "Successful", str(successful), "with results"),
                (c3, f"Avg {metric_name}", f"{avg_score:.2f}", "best per SKU"),
                (c4, "Pipelines tested", str(total_pipes), "across all SKUs"),
            ]:
                col.markdown(
                    f'<div class="af-metric"><div class="label">{label}</div>'
                    f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.dataframe(best_df, use_container_width=True, hide_index=True)

            hm = heatmap_chart(results, metric_name)
            if hm: st.plotly_chart(hm, use_container_width=True)

            # Downloads
            st.markdown("#### ⬇️ Downloads")
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                csv_buf = io.StringIO()
                summary_df.to_csv(csv_buf, index=False)
                st.download_button("📋 Pipeline results (CSV)", csv_buf.getvalue(),
                                   "pipeline_results.csv", "text/csv")
            with dl2:
                all_fc = []
                for sku_id, data in results.items():
                    best = data.get("best")
                    if best and best.get("forecast_df") is not None:
                        fc = best["forecast_df"].copy()
                        fc["sku_id"] = sku_id
                        fc["model"] = best["model"]
                        fc["cleaning"] = best["cleaning"]
                        all_fc.append(fc)
                if all_fc:
                    all_fc_df = pd.concat(all_fc, ignore_index=True)
                    csv2 = io.StringIO()
                    all_fc_df.to_csv(csv2, index=False)
                    st.download_button("📈 All forecasts (CSV)", csv2.getvalue(),
                                       "all_forecasts.csv", "text/csv")
            with dl3:
                repro = build_repro_package(results)
                st.download_button("🔁 Reproducibility (JSON)", json.dumps(repro, indent=2, default=str),
                                   "reproducibility.json", "application/json")

    # ── Forecast tab ──────────────────────────────────────────────────────
    with tab_forecast:
        sku_keys = list(results.keys())
        sel_sku = st.selectbox("Select SKU", sku_keys, key="fc_sku_sel")
        if sel_sku:
            data = results[sel_sku]
            best = data.get("best")
            ranked = data.get("ranked", [])
            if best:
                fig = forecast_chart(
                    best["train_df"], best["test_df"], best["forecast_df"],
                    best["model"], CLEANING_LABELS.get(best["cleaning"], best["cleaning"]),
                    sel_sku, best["score"], metric_name
                )
                st.plotly_chart(fig, use_container_width=True)

                # Overlay all pipelines
                if len(ranked) > 1:
                    with st.expander("🔀 Overlay all pipeline forecasts"):
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=best["train_df"]["ds"], y=best["train_df"]["y"],
                            name="Historical", line=dict(color="#5b5296", width=1)
                        ))
                        for i, r in enumerate(ranked[:5]):
                            if r.get("forecast_df") is not None:
                                fig2.add_trace(go.Scatter(
                                    x=r["forecast_df"]["ds"], y=r["forecast_df"]["yhat"],
                                    name=f"{'★ ' if i==0 else ''}{r['cleaning']}+{r['model']}",
                                    line=dict(color=PALETTE[i % len(PALETTE)],
                                              width=2.5 if i==0 else 1.5,
                                              dash="solid" if i==0 else "dot")
                                ))
                        fig2.update_layout(hovermode="x unified", height=400, **THEME)
                        st.plotly_chart(fig2, use_container_width=True)

                # Forecast table
                if best.get("forecast_df") is not None:
                    fc = best["forecast_df"].copy()
                    for col in ["yhat", "yhat_lower", "yhat_upper"]:
                        if col in fc.columns:
                            fc[col] = fc[col].round(3)
                    fc["ds"] = fc["ds"].astype(str)
                    st.markdown("**Forecast values**")
                    st.dataframe(fc, use_container_width=True, hide_index=True)

                    # Per-SKU download
                    buf = io.StringIO()
                    fc.to_csv(buf, index=False)
                    st.download_button(f"⬇️ Download {sel_sku} forecast",
                                       buf.getvalue(), f"forecast_{sel_sku}.csv", "text/csv")
            else:
                st.warning(f"No successful forecast for {sel_sku}")

    # ── Model details tab ─────────────────────────────────────────────────
    with tab_details:
        sel_sku2 = st.selectbox("Select SKU", list(results.keys()), key="det_sku_sel")
        if sel_sku2:
            data = results[sel_sku2]
            ranked = data.get("ranked", [])
            best = data.get("best")

            if best:
                st.markdown(f"#### {sel_sku2} — All pipeline results")
                pipe_rows = []
                for r in data.get("results", []):
                    m = r.get("metrics", {})
                    pipe_rows.append({
                        "Cleaning": CLEANING_LABELS.get(r["cleaning"], r["cleaning"]),
                        "Model": r["model"],
                        metric_name: round(r["score"], 4) if r["score"] < 1e8 else "Failed",
                        "Error": r.get("error", "") or "",
                    })
                if pipe_rows:
                    st.dataframe(pd.DataFrame(pipe_rows), use_container_width=True, hide_index=True)

                st.plotly_chart(comparison_bar(ranked, sel_sku2, metric_name), use_container_width=True)

                # Model-specific insights
                model_obj = best.get("model_obj")
                if model_obj:
                    if best["model"] == "XGBoost":
                        fi_fig = feature_importance_fig(model_obj)
                        if fi_fig: st.plotly_chart(fi_fig, use_container_width=True)
                    elif best["model"] == "ARIMA" and hasattr(model_obj, "aic"):
                        st.markdown(f"**AIC:** {model_obj.aic:.2f}  |  **BIC:** {model_obj.bic:.2f}")
                        st.markdown(f"**Best params:** `{best['params']}`")

                # Hyperparameters
                with st.expander("Best hyperparameters"):
                    st.json(best.get("params", {}))

    # ── Reproducibility tab ───────────────────────────────────────────────
    with tab_repro:
        repro = build_repro_package(results)
        st.markdown("#### Reproducibility package")
        st.markdown(
            "All seeds, split indices, and hyperparameters are recorded here. "
            "Re-running with the same file and these settings will produce identical results."
        )
        st.json(repro)
        st.download_button(
            "⬇️ Download reproducibility.json",
            json.dumps(repro, indent=2, default=str),
            "reproducibility.json", "application/json"
        )


def build_repro_package(results):
    return {
        "generated_at": datetime.now().isoformat(),
        "skus": {
            sku_id: {
                "best_pipeline": {
                    "cleaning": d["best"]["cleaning"],
                    "model":    d["best"]["model"],
                    "params":   d["best"]["params"],
                    "score":    d["best"]["score"],
                } if d.get("best") else None,
                "split": d.get("split", {}),
                "all_results": [
                    {"cleaning": r["cleaning"], "model": r["model"],
                     "params": r["params"], "score": r["score"]}
                    for r in d.get("results", [])
                ],
            }
            for sku_id, d in results.items()
        }
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg = render_sidebar()

    # Header
    st.markdown(
        '<div style="background:linear-gradient(135deg,#2d1b69 0%,#1c1a32 70%,#0f0d1c 100%);'
        'border-radius:14px;padding:28px 32px;margin-bottom:24px;border:1px solid #2a2750">'
        '<div style="font-size:28px;font-weight:800;color:#f0ecff">📈 AutoForecast Pro</div>'
        '<div style="color:#9d94c8;font-size:14px;margin-top:6px">'
        'Multi-SKU · 5 models · 4 cleaning methods · automated pipeline selection</div></div>',
        unsafe_allow_html=True
    )

    # ── No file uploaded ───────────────────────────────────────────────────
    if cfg.get("uploaded") is None:
        st.markdown(
            '<div class="upload-hero">'
            '<div style="font-size:56px;margin-bottom:16px">📊</div>'
            '<h2>No data loaded</h2>'
            '<p>Please upload an Excel file (.xlsx or .xls) or CSV using the sidebar.<br>'
            'The app will auto-detect your date, value, and SKU columns.<br><br>'
            '<strong style="color:#b8b0e0">Expected format:</strong><br>'
            '<code style="background:#1c1a32;padding:2px 8px;border-radius:4px;color:#a78bfa">'
            'ds (date) &nbsp;|&nbsp; y (value) &nbsp;|&nbsp; sku_id (optional)</code>'
            '</p></div>',
            unsafe_allow_html=True
        )
        return

    # ── Run button ─────────────────────────────────────────────────────────
    skus = st.session_state.get("skus", {})
    if not skus:
        st.warning("❌ No valid data found after column mapping. Check your column selections.")
        return

    n_pipes = len(cfg.get("cleaning_methods", [])) * len(cfg.get("models", []))
    n_skus  = len(skus)

    col_run, col_info = st.columns([2, 3])
    with col_run:
        run_clicked = st.button(
            f"🚀 Run Full Comparison",
            use_container_width=True,
            disabled=not cfg.get("can_run", False),
        )
    with col_info:
        st.markdown(
            f'<div style="padding:10px;color:#9d94c8;font-size:13px">'
            f'{n_skus} SKU{"s" if n_skus>1 else ""} · '
            f'{len(cfg.get("cleaning_methods",[]))} cleaning × '
            f'{len(cfg.get("models",[]))} models = '
            f'<strong style="color:#a78bfa">{n_pipes} pipelines/SKU</strong></div>',
            unsafe_allow_html=True
        )

    # ── Execute ────────────────────────────────────────────────────────────
    if run_clicked:
        with st.spinner(""):
            with st.expander("▶ Progress log", expanded=True):
                results = run_all(cfg)
        st.session_state["results"]      = results
        st.session_state["run_complete"] = True
        st.success(f"✅ Complete! {n_skus} SKU(s) processed.")
        st.rerun()

    # ── Show results ───────────────────────────────────────────────────────
    if st.session_state.get("run_complete") and st.session_state.get("results"):
        render_results(st.session_state["results"], cfg.get("metric", "SMAPE"))


if __name__ == "__main__":
    main()
