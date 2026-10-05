"""
MT4 Forex Predictor — Phase 4
MT4 右进左出 K 线惯例。100根/页，时间起止翻页，点击 K 线居中。
启动: cd D:\dapanAllThings\codes\forex_predictor && .\venv\Scripts\python.exe -m streamlit run dashboard/streamlit_app.py --server.port 8501
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

API = "http://localhost:8000"
PAGE_SIZE = 100
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
RUN_COLORS_CYCLE = ["#4ade80", "#e8d44d", "#ffffff", "#ffcc00", "#8b5cf6"]
TFS = ["1m", "5m", "15m", "30m", "1h"]

st.set_page_config(page_title="MT4 Forex Predictor", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# SESSION STATE
# ============================================================
KEYS = {
    "mode": "training",      # training | live
    "symbol": "EURUSD",
    "dimension": 212,
    "model_tf": "all",       # filter for model list
    "selected_runs": [],
    "view_tf": "15m",        # active chart TF
    "start_ts": "",          # pagination start time (empty = latest)
}
for k, v in KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# API
# ============================================================
@st.cache_data(ttl=10, show_spinner=False)
def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params, timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=5, show_spinner="加载蜡烛...")
def load_candles(symbol, tf, limit=100, before=None, from_ts=None):
    params = {"symbol": symbol, "tf": tf, "limit": limit}
    if before:
        params["before"] = before
    elif from_ts:
        params["from"] = from_ts
    return api_get("/api/candles", params)


@st.cache_data(ttl=10, show_spinner=False)
def load_feature_versions(status="active"):
    return api_get("/api/feature-versions", {"status": status, "limit": 50})


@st.cache_data(ttl=10, show_spinner=False)
def load_runs(symbol, dimension=None, target_tf=None, limit=20):
    p = {"symbol": symbol, "limit": limit, "sort_by": "auc", "order": "desc"}
    if dimension:
        p["dimension"] = dimension
    if target_tf and target_tf != "all":
        p["target_tf"] = target_tf
    return api_get("/api/runs", p)


@st.cache_data(ttl=15, show_spinner=False)
def load_predictions(symbol, run_id, target_tf, mode, from_ts=None, to_ts=None, limit=500):
    p = {"symbol": symbol, "run_id": run_id, "target_tf": target_tf, "mode": mode, "limit": limit}
    if from_ts:
        p["from"] = from_ts
    if to_ts:
        p["to"] = to_ts
    return api_get("/api/predictions", p)


@st.cache_data(ttl=60, show_spinner=False)
def load_shap(run_id, limit=10):
    return api_get("/api/shap", {"run_id": run_id, "limit": limit})


@st.cache_data(ttl=30, show_spinner=False)
def load_assessment(symbol, run_ids_str):
    return api_get("/api/assessment", {"symbol": symbol, "run_ids": run_ids_str})


# ============================================================
# INDICATORS
# ============================================================
def ma(arr, p):
    return pd.Series(arr).rolling(window=p, min_periods=p).mean().values


def macd(arr):
    s = pd.Series(arr)
    ef = s.ewm(span=12, adjust=False).mean()
    es = s.ewm(span=26, adjust=False).mean()
    ml = (ef - es).values
    sl = pd.Series(ml).ewm(span=9, adjust=False).mean().values
    return ml, sl, ml - sl


def rsi(arr, p=14):
    s = pd.Series(arr)
    d = s.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ag = g.ewm(span=p, adjust=False).mean()
    al = l.ewm(span=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def run_colors(runs):
    m = {}
    for i, r in enumerate(runs):
        m[r["run_id"]] = RUN_COLORS_CYCLE[i % len(RUN_COLORS_CYCLE)]
    return m


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("### MT4 Predictor")

        # Mode — radio directly controls state
        mode_idx = 0 if st.session_state.mode == "training" else 1
        mode = st.radio("模式", ["训练回测", "实盘预测"], index=mode_idx, horizontal=True,
                         help="训练回测=看holdout预测 | 实盘预测=看实时natural预测",
                         key="_mode_radio")
        st.session_state.mode = "training" if mode == "训练回测" else "live"

        # Symbol
        syms = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY"]
        st.session_state.symbol = st.selectbox(
            "货币对", syms, index=syms.index(st.session_state.symbol) if st.session_state.symbol in syms else 0,
            key="_symbol_sel")

        # Dimension
        fv = load_feature_versions("active")
        dims = sorted(set(v["dimension"] for v in fv["versions"])) if (fv and fv.get("versions")) else [212, 113]
        if st.session_state.dimension not in dims:
            st.session_state.dimension = dims[0]
        st.session_state.dimension = st.selectbox("维度", dims, index=dims.index(st.session_state.dimension),
                                                   key="_dim_sel")

        # Model TF filter
        tf_opts = ["all", "15m", "5m", "30m", "1h"]
        st.session_state.model_tf = st.selectbox("模型 TF", tf_opts,
                                                  index=tf_opts.index(st.session_state.model_tf) if st.session_state.model_tf in tf_opts else 0,
                                                  key="_tf_sel")

        # Runs
        st.markdown("**Run** (按 AUC)")
        mtf = None if st.session_state.model_tf == "all" else st.session_state.model_tf
        runs_data = load_runs(st.session_state.symbol, st.session_state.dimension, mtf)
        runs = runs_data.get("runs", []) if runs_data else []
        st.session_state._runs = runs

        for r in runs:
            rid = r["run_id"]
            auc_str = f" AUC:{r['auc']:.4f}" if r.get("auc") else ""
            label = f"{rid[:12]}...{auc_str}"
            checked = rid in st.session_state.selected_runs
            cb = st.checkbox(label, value=checked, key=f"run_{rid}")
            if cb and rid not in st.session_state.selected_runs:
                st.session_state.selected_runs.append(rid)
            elif not cb and rid in st.session_state.selected_runs:
                st.session_state.selected_runs.remove(rid)

        if not runs:
            st.caption("(训练后出现)")

        st.divider()
        st.caption(f"更新 {datetime.now().strftime('%H:%M:%S')}")
        st.caption(API)


# ============================================================
# ASSESSMENT
# ============================================================
def render_assessment():
    if not st.session_state.selected_runs:
        return
    data = load_assessment(st.session_state.symbol, ",".join(st.session_state.selected_runs[:10]))
    if not data or not data.get("tfs"):
        return
    with st.expander(" 研判依据", expanded=True):
        cols = st.columns(4)
        for i, tf in enumerate(["5m", "15m", "30m", "1h"]):
            with cols[i]:
                st.markdown(f"**{tf}**")
                items = data["tfs"].get(tf, [])
                if not items:
                    st.caption("—")
                    continue
                up = sum(1 for it in items if it["signal"] == "涨")
                down = len(items) - up
                for it in items:
                    st.caption(f"{'' if it['signal']=='涨' else ''}{it['run_id'][:8]} {it['signal']}"
                              f"{' *' if it.get('is_adopted') else ''}")
                st.caption(f"_{up}涨{down}跌_")


# ============================================================
# CHART BUILDING
# ============================================================
def build_figure(df):
    n = len(df)
    if n < 2:
        return None

    closes = df["close"].values
    times = df["timestamp"].values

    # Use integer index as x-axis → no gaps (MT4 style: adjacent bars regardless of time gaps)
    x_idx = list(range(n))
    # Labels for x-axis ticks (show ~10 labels)
    step = max(1, n // 10)
    tick_idx = list(range(0, n, step))
    tick_labels = [str(t)[:16] for t in times[::step]]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.55, 0.22, 0.23],
                        subplot_titles=(f"{st.session_state.symbol} {st.session_state.view_tf}",
                                        "MACD (12,26,9)", "RSI (14)"))

    # Candlestick — x as index
    fig.add_trace(go.Candlestick(
        x=x_idx, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="", showlegend=False,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # MA
    for label, period, color in [("5",5,"#f9a825"),("10",10,"#ce93d8"),("20",20,"#42a5f5"),
                                  ("60",60,"#ef5350"),("100",100,"#66bb6a")]:
        ma_vals = ma(closes, period)
        fig.add_trace(go.Scatter(x=x_idx, y=ma_vals, mode="lines", line=dict(width=1, color=color),
                                  name=f"MA{label}"), row=1, col=1)

    # Prediction markers
    api_mode = "training" if st.session_state.mode == "training" else "natural"
    ts_to_idx = {str(t): i for i, t in enumerate(times)}
    from_ts = str(times[0]).replace(" ", "T")
    to_ts = str(times[-1]).replace(" ", "T")
    all_runs = st.session_state.get("_runs", [])
    color_map = run_colors(all_runs)

    for rid in st.session_state.selected_runs:
        pd_data = load_predictions(st.session_state.symbol, rid, st.session_state.view_tf,
                                   api_mode, from_ts=from_ts, to_ts=to_ts)
        if not pd_data or not pd_data.get("predictions"):
            continue
        clr = color_map.get(rid, "#707080")
        for p in pd_data["predictions"]:
            ts = p["timestamp"]
            idx = ts_to_idx.get(str(ts))
            if idx is None:
                continue
            sig = p.get("signal")
            adopted = p.get("is_adopted", 0)
            prob = p.get("prob_up", 0.5)
            y = float(df.iloc[idx]["high"]) * 1.0003 if sig == 1 else float(df.iloc[idx]["low"]) * 0.9997
            marker_color = "#707080" if api_mode == "training" else clr
            symbol = "diamond" if adopted else "circle"
            fig.add_trace(go.Scatter(
                x=[idx], y=[y], mode="markers",
                marker=dict(color=marker_color, symbol=symbol, size=10 if adopted else 7,
                            line=dict(width=1, color=marker_color)),
                name=f"{rid[:8]}",
                text=[f"Run:{rid[:8]} Prob:{prob:.2%} {'ADOPTED' if adopted else ''}"],
                hoverinfo="text", showlegend=False,
            ), row=1, col=1)

    # MACD
    ml, ms, mh = macd(closes)
    fig.add_trace(go.Bar(x=x_idx, y=mh, marker_color=["#26a69a" if v>=0 else "#ef5350" for v in mh],
                          name="Hist", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_idx, y=ml, mode="lines", line=dict(width=1, color="#f9a825"), name="DIF"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_idx, y=ms, mode="lines", line=dict(width=1, color="#ef5350"), name="DEA"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#484f58", width=1), row=2, col=1)

    # RSI
    rsi_vals = rsi(closes)
    fig.add_trace(go.Scatter(x=x_idx, y=rsi_vals, mode="lines", line=dict(width=1.5, color="#ce93d8"), name="RSI"), row=3, col=1)
    for lvl, c in [(70,"#f85149"),(50,"#484f58"),(30,"#3fb950")]:
        fig.add_hline(y=lvl, line=dict(color=c, width=1, dash="dot" if lvl==50 else "solid"), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=700,
        margin=dict(l=10, r=10, t=35, b=10),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        dragmode="pan",  # no selection box
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        paper_bgcolor="#131722", plot_bgcolor="#131722",
    )
    fig.update_xaxes(gridcolor="#1e2433", tickvals=tick_idx, ticktext=tick_labels, tickangle=45)
    fig.update_yaxes(gridcolor="#1e2433")
    for ann in fig.layout.annotations:
        ann.font = dict(size=10, color="#8b949e")
    return fig


# ============================================================
# MAIN PAGE
# ============================================================
st.title("MT4 Forex Predictor")
st.caption(f"{st.session_state.symbol} | {st.session_state.view_tf} | "
           f"{'训练回测' if st.session_state.mode=='training' else '实盘预测'}")

render_sidebar()

# --- TF radio ---
tf_idx = TFS.index(st.session_state.view_tf) if st.session_state.view_tf in TFS else 2
st.session_state.view_tf = st.radio("时间框架", TFS, index=tf_idx, horizontal=True)

# --- Assessment ---
render_assessment()

# --- Load candles ---
start_ts = st.session_state.start_ts if st.session_state.start_ts else None
candle_data = load_candles(st.session_state.symbol, st.session_state.view_tf,
                           PAGE_SIZE, from_ts=start_ts)

if candle_data and candle_data.get("candles") and len(candle_data["candles"]) > 0:
    candles = candle_data["candles"]
    total = candle_data.get("total_count", 0)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Current page info
    first_ts = candles[0]["timestamp"]
    last_ts = candles[-1]["timestamp"]
    tf_min = TF_MINUTES.get(st.session_state.view_tf, 15)

    # --- Chart ---
    fig = build_figure(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="main_chart")

    # --- Pagination (MT4: 前=older/left, 后=newer/right) ---
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 2, 1, 1])

    # "最新" → reset to latest
    is_latest = (not st.session_state.start_ts)
    with c1:
        if st.button("最新", use_container_width=True, disabled=is_latest,
                     help="回到最新 100 根蜡烛"):
            st.session_state.start_ts = ""
            st.rerun()

    # "前" → older (subtract 100 * TF minutes from current start)
    with c2:
        if st.button("前", use_container_width=True, help=f"更早 {PAGE_SIZE*tf_min} 分钟"):
            dt = pd.to_datetime(first_ts)
            new_start = dt - timedelta(minutes=PAGE_SIZE * tf_min)
            st.session_state.start_ts = new_start.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    # "后" → newer (add 100 * TF minutes)
    with c3:
        can_go_next = bool(st.session_state.start_ts)
        if st.button("后", use_container_width=True, disabled=not can_go_next,
                     help=f"更新 {PAGE_SIZE*tf_min} 分钟"):
            if st.session_state.start_ts:
                dt = pd.to_datetime(st.session_state.start_ts)
                new_start = dt + timedelta(minutes=PAGE_SIZE * tf_min)
                st.session_state.start_ts = new_start.strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

    # Info
    with c4:
        st.markdown(
            f"<div style='text-align:center;color:#8b949e;font-size:13px;padding-top:5px'>"
            f"{first_ts[:16]} → {last_ts[:16]} | 共{total:,}根 | 约{total_pages}页</div>",
            unsafe_allow_html=True)

    # "最早" → first page
    with c5:
        if st.button("最早", use_container_width=True, disabled=total == 0,
                     help="跳到最早蜡烛"):
            st.session_state.start_ts = "2000-01-01 00:00:00"
            st.rerun()

    # Time input for direct jump
    with c6:
        jump_to = st.text_input("跳转", key="_jump_ts", placeholder="YYYY-MM-DD HH:MM",
                                label_visibility="collapsed")
        if jump_to and len(jump_to) >= 16:
            st.session_state.start_ts = jump_to[:16]
            st.rerun()

else:
    st.warning("蜡烛数据为空。请确认 MySQL 数据已同步。")

# --- SHAP in sidebar ---
with st.sidebar:
    st.divider()
    st.markdown("### SHAP")
    if st.session_state.selected_runs:
        run_sel = st.selectbox("Run", st.session_state.selected_runs, format_func=lambda r: r[:16], key="_shap_sel")
        if run_sel and st.button("查询 SHAP", key="_shap_btn"):
            sd = load_shap(run_sel, 10)
            if sd and sd.get("features"):
                df_s = pd.DataFrame(sd["features"]).sort_values("importance", ascending=True)
                st.bar_chart(df_s.set_index("feature_name")["importance"], horizontal=True)
            else:
                st.caption("暂无 SHAP 数据（训练后灌入）")
    else:
        st.caption("勾选 Run 后查询")


# ============================================================
# PAGE-DATA 四页签（训练概览 / 最新信号 / SHAP Top / 数据健康度）
# 数据源: dashboard/data/page-data.json（辅助C schema）
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def load_page_data():
    try:
        p = Path(__file__).parent / "data" / "page-data.json"
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


st.divider()
pd_data = load_page_data()
if pd_data:
    st.caption(f"数据快照 {pd_data.get('generated_at','')} | {pd_data.get('source_note','')}")
else:
    st.caption("page-data.json 未就位（四页签显示暂无数据）")

tab_overview, tab_signals, tab_shap, tab_health = st.tabs(
    ["训练概览", "最新信号", "SHAP Top", "数据健康度"])

with tab_overview:
    mm = (pd_data or {}).get("model_meta")
    if mm:
        st.caption(f"模型: {mm.get('name','')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{mm['auc']:.4f}" if isinstance(mm.get('auc'), (int, float)) else "—")
        c2.metric("特征数", mm.get('features_n', '—'))
        c3.metric("训练时间", str(mm.get('trained_at', '—'))[:16])
        c4.metric("货币对", ",".join(mm.get('currency_pairs', [])) or "—")
    else:
        st.caption("暂无数据")

with tab_signals:
    signals = (pd_data or {}).get("latest_signals")
    if signals:
        df_sig = pd.DataFrame(signals)
        st.dataframe(df_sig, use_container_width=True, hide_index=True)
    else:
        st.caption("暂无数据")

with tab_shap:
    shap_top = (pd_data or {}).get("shap_top")
    if shap_top:
        df_shap = pd.DataFrame(shap_top).sort_values("importance", ascending=True)
        st.bar_chart(df_shap.set_index("feature")["importance"], horizontal=True)
    else:
        st.caption("暂无数据")

with tab_health:
    dh = (pd_data or {}).get("data_health")
    if dh:
        c1, c2, c3 = st.columns(3)
        c1.metric("M1 最新时间", str(dh.get('m1_latest_ts', '—'))[:16])
        c2.metric("覆盖天数", f"{dh['coverage_days']:,.0f}" if isinstance(dh.get('coverage_days'), (int, float)) else "—")
        c3.metric("缺口小时数", f"{dh['gap_hours']:,.1f}" if isinstance(dh.get('gap_hours'), (int, float)) else "—")
        st.caption(dh.get("source", ""))
    else:
        st.caption("暂无数据")
