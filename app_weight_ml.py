# app_weight_ml.py
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from strategy_weight_ml_wf import (
    load_price_panel_6, build_signals_base, regime_series_5,
    walk_forward, ann_stats
)

st.set_page_config(page_title="ML Regime Portfolio", layout="wide")
load_dotenv()

st.title("ML Regime Portfolio — Interactive")

# Sidebar controls
st.sidebar.header("Controls")
freq = st.sidebar.selectbox("Rebalance Frequency", ["ME","2W-FRI","W-FRI"], index=2)
init = st.sidebar.number_input("Initial Capital", min_value=100000.0, value=1_000_000.0, step=100000.0)
contrib = st.sidebar.number_input("Monthly Contribution", min_value=0.0, value=300_000.0, step=100000.0)
tc = st.sidebar.number_input("Trading Cost (%)", min_value=0.0, value=0.2, step=0.05, format="%.2f")
history_years = st.sidebar.slider("History Years", 5, 15, 10)
train_years   = st.sidebar.slider("Train Window (years)", 3, 8, 5)
min_win       = st.sidebar.slider("Min samples", 12, 60, 24, step=6)
fast          = st.sidebar.checkbox("Fast mode (lower tuning)", value=True)

st.sidebar.markdown("---")
mu_bonus = st.sidebar.slider("μ Bonus (return tilt)", 0.0, 0.6, 0.35, 0.05)
max_lev  = st.sidebar.slider("Max Leverage", 1.0, 2.5, 2.0, 0.1)

st.sidebar.markdown("### σ Targets (per regime)")
sigma_vro = st.sidebar.slider("very_risk_off σ", 0.04, 0.20, 0.08, 0.01)
sigma_ro  = st.sidebar.slider("risk_off σ",      0.06, 0.25, 0.11, 0.01)
sigma_neu = st.sidebar.slider("neutral σ",       0.08, 0.30, 0.17, 0.01)
sigma_ron = st.sidebar.slider("risk_on σ",       0.10, 0.35, 0.22, 0.01)
sigma_von = st.sidebar.slider("very_risk_on σ",  0.12, 0.40, 0.30, 0.01)

st.sidebar.markdown("### Constraints")
min_equity_floor = st.sidebar.slider("Min Equity Floor", 0.0, 0.7, 0.30, 0.05)
bil_cap          = st.sidebar.slider("BIL Cap", 0.0, 0.8, 0.25, 0.05)
lambda_cash_scale= st.sidebar.slider("Cash Penalty Scale", 0.2, 2.0, 1.0, 0.1)
caps_scale       = st.sidebar.slider("Equity Caps Scale", 0.7, 1.5, 1.15, 0.05)

run = st.sidebar.button("Run Backtest")

@st.cache_data(show_spinner=False)
def _load_inputs(history_years: int):
    px = load_price_panel_6(years=history_years)
    sig = build_signals_base(years=history_years)
    return px, sig

if run:
    with st.spinner("Downloading & computing..."):
        px_daily, signals = _load_inputs(history_years)
        reg5 = regime_series_5(signals, freq=freq)

        sigma_targets = {
            "very_risk_off": sigma_vro,
            "risk_off":      sigma_ro,
            "neutral":       sigma_neu,
            "risk_on":       sigma_ron,
            "very_risk_on":  sigma_von,
        }
        outdir = Path("data_pro")
        nav, eq, twr, xirr, avg_df = walk_forward(
            px_daily, reg5, freq=freq,
            train_years=train_years, min_win=min_win,
            tc_percent=tc, initial_capital=init, monthly_contribution=contrib,
            max_lev=max_lev, fast=fast, mu_bonus=mu_bonus,
            min_equity_floor=min_equity_floor, lambda_cash_scale=lambda_cash_scale,
            caps_scale=caps_scale, bil_cap=bil_cap, sigma_targets=sigma_targets,
            export_weights_path=outdir / f"wf_weights_{freq}.csv",
            export_positions_path=outdir / f"wf_positions_{freq}.csv",
            redistribute_excess=True
        )

    # NAV chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="NAV", mode="lines"))
    fig.update_layout(height=420, template="plotly_white",
                      title="Time-Weighted NAV", hovermode="x unified",
                      yaxis_title="NAV (start=1.0)")
    st.plotly_chart(fig, use_container_width=True)

    # Performance
    ppy_map = {"ME":12,"2W-FRI":26,"W-FRI":52}
    perf = ann_stats(nav, ppy=ppy_map[freq])
    st.subheader("Performance")
    st.write(pd.DataFrame([{
        "Freq":freq, "TWR_CAGR":perf["CAGR"], "TWR_VOL":perf["VOL"],
        "TWR_MDD":perf["MDD"], "TWR_Sharpe":perf["Sharpe"], "MWR_XIRR":xirr,
        "Final_Equity": eq.iloc[-1], "Total_Contrib": float(init + contrib * max(0, len(set(eq.index.to_period('M')))-1))
    }]).round(6))

    # Avg weights by regime
    st.subheader("Learned Weights — Average by Regime")
    st.dataframe((avg_df.set_index("Asset")*100).round(2))

    # Latest positions (from exported CSV)
    wf_pos = pd.read_csv(f"data_pro/wf_positions_{freq}.csv", parse_dates=["date"])
    last_date = wf_pos["date"].max()
    st.subheader(f"Latest Positions @ {last_date.date()}")
    st.write(wf_pos[wf_pos["date"]==last_date].drop(columns=["date"]).reset_index(drop=True))

    # Downloads
    c1,c2,c3,c4 = st.columns(4)
    c1.download_button("Download NAV CSV", nav.to_csv().encode("utf-8"), file_name=f"wf_nav_{freq}.csv")
    c2.download_button("Download Equity CSV", eq.to_csv().encode("utf-8"), file_name=f"wf_equity_{freq}.csv")
    c3.download_button("Download Weights by Regime CSV", (avg_df).to_csv(index=False).encode("utf-8"),
                       file_name=f"wf_learned_weights_by_regime_{freq}.csv")
    c4.download_button("Download Positions CSV", wf_pos.to_csv(index=False).encode("utf-8"),
                       file_name=f"wf_positions_{freq}.csv")

else:
    st.info("좌측에서 옵션을 고르고 **Run Backtest**를 눌러주세요.")
