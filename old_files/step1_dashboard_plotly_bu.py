# step1_dashboard_plotly.py
# -----------------------------------------------------------
# Interactive Macro Dashboard (Plotly) with per-chart subtitles
# + Global Equities (10) & Commodities (5) section
# + USD terms (FX-adjusted) for global indices
# - 3y data from FRED + yfinance (with fallbacks)
# - Monthly EOM resample, derived (breakeven, curve), Δ3M/Δ1Y
# - Single HTML report with interactive charts + range slider
# -----------------------------------------------------------
import os, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# ============== Config ==============
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing. Put it in .env like: FRED_API_KEY=YOUR_KEY")

OUTDIR = Path(os.getenv("OUTDIR", "dash_pro"))
OUTDIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUTDIR / "index.html"

today = dt.date.today()
start_date = (today - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# ============== Helpers ==============
def fred_series(code, start=None):
    s = fred.get_series_latest_release(code)
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s.astype(float)

def yf_series_multi(candidates, period="3y", interval="1d"):
    for tkr in candidates:
        try:
            df = yf.download(tkr, period=period, interval=interval,
                             progress=False, auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                return df, tkr
        except Exception:
            pass
    return None, None

def eom(x):
    if x is None:
        return None
    if hasattr(x, "empty") and x.empty:
        return None
    if isinstance(x, pd.DataFrame):
        x = x["Close"] if "Close" in x.columns else x.iloc[:, 0]
        x = x.squeeze()
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    x = x.dropna()
    x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[~x.index.isna()]
    if x.empty:
        return None
    return x.resample("ME").last().dropna()

def delta_nm(s, n):
    s = s.dropna()
    if len(s) < (n+1): return None
    v = s.iloc[-1] - s.iloc[-(n+1)]
    return float(v.item() if hasattr(v, "item") else v)

def last_label(ts, unit=""):
    if ts is None or ts.empty: return "—"
    v = ts.iloc[-1]
    if unit == "bps": return f"{float(v):,.0f} bps"
    if unit == "pct": return f"{float(v):,.2f}%"
    if unit == "idx": return f"{float(v):,.2f}"
    if unit == "fx":  return f"{float(v):,.2f}"
    return f"{float(v):,.2f}"

# ---------- Figure builders (with subtitles) ----------
def fig_line(ts, name, yaxis_title, unit=None, show_range_slider=True, subtitle=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name=name, mode="lines"))
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=130, b=50),
        yaxis_title=yaxis_title,
        title=dict(text=name, x=0.01, xanchor="left", font=dict(size=18, color="#222")),
        template="plotly_white",
        hovermode="x unified",
    )
    if subtitle:
        fig.add_annotation(
            text=f"<b style='color:#666;font-size:13px'>{subtitle}</b>",
            xref="paper", yref="paper",
            x=0.0, y=1.30,
            showarrow=False
        )

    if show_range_slider:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

    if ts is not None and len(ts):
        fig.add_annotation(
            x=ts.index[-1], y=ts.values[-1],
            text=last_label(ts, unit),
            showarrow=True, arrowhead=2, ax=20, ay=-20
        )
    return fig

def fig_two_lines(ts1, ts2, name1, name2, yaxis_title, unit=None,
                  show_range_slider=True, subtitle=None, legend_pos="top-right", subtitle_y=1.32):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts1.index, y=ts1.values, name=name1, mode="lines"))
    fig.add_trace(go.Scatter(x=ts2.index, y=ts2.values, name=name2, mode="lines"))

    top_margin = 150 if subtitle_y > 1.30 else 130
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=top_margin, b=60),
        yaxis_title=yaxis_title,
        title=dict(text=f"{name1} & {name2}", x=0.01, xanchor="left", font=dict(size=18, color="#222")),
        template="plotly_white",
        hovermode="x unified",
    )

    if legend_pos == "bottom":
        legend_cfg = dict(orientation="h", y=0.02, yanchor="bottom", x=0.98, xanchor="right")
    elif legend_pos == "top":
        legend_cfg = dict(orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center")
    elif legend_pos == "top-right":
        legend_cfg = dict(orientation="h", y=1.15, yanchor="top", x=1.0, xanchor="right")
    else:
        legend_cfg = dict(orientation="h", y=0.02, yanchor="bottom", x=0.5, xanchor="center")
    fig.update_layout(legend=legend_cfg)

    if subtitle:
        fig.add_annotation(
            text=f"<b style='color:#666;font-size:13px'>{subtitle}</b>",
            xref="paper", yref="paper",
            x=0.0, y=subtitle_y,
            showarrow=False
        )

    if show_range_slider:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

    if ts1 is not None and len(ts1):
        fig.add_annotation(x=ts1.index[-1], y=ts1.values[-1], text=last_label(ts1, unit),
                           showarrow=True, arrowhead=2, ax=20, ay=-20)
    if ts2 is not None and len(ts2):
        fig.add_annotation(x=ts2.index[-1], y=ts2.values[-1], text=last_label(ts2, unit),
                           showarrow=True, arrowhead=2, ax=20, ay=20)
    return fig

def fig_bar(series, title, yaxis_title):
    s = series.dropna()
    fig = go.Figure(go.Bar(
        x=s.index.astype(str), y=s.values,
        text=[f"{v:,.2f}" for v in s.values], textposition="auto"
    ))
    fig.update_layout(
        height=420, margin=dict(l=60, r=30, t=50, b=80),
        yaxis_title=yaxis_title, template="plotly_white",
        xaxis=dict(tickangle=15)
    )
    return fig

# ===================== FX mapping for USD rebasing =====================
# quote_type: 'usd_per_local'  (e.g., EURUSD=X, GBPUSD=X)  -> USD series = local_index * FX
#              'local_per_usd' (e.g., KRW=X, JPY=X, CNY=X) -> USD series = local_index / FX
FX_MAP = {
    "S&P 500 (^GSPC)":      {"fx": None,          "quote_type": None},            # already USD
    "Euro Stoxx 50 (^STOXX50E)": {"fx": "EURUSD=X", "quote_type": "usd_per_local"},
    "Nikkei 225 (^N225)":   {"fx": "JPY=X",       "quote_type": "local_per_usd"},
    "Shanghai Composite (000001.SS)": {"fx": "CNY=X","quote_type": "local_per_usd"},
    "Hang Seng (^HSI)":     {"fx": "HKD=X",       "quote_type": "local_per_usd"},
    "KOSPI (^KS11)":        {"fx": "KRW=X",       "quote_type": "local_per_usd"},
    "TAIEX (^TWII)":        {"fx": "TWD=X",       "quote_type": "local_per_usd"},
    "Nifty 50 (^NSEI)":     {"fx": "INR=X",       "quote_type": "local_per_usd"},
    "DAX (^GDAXI)":         {"fx": "EURUSD=X",    "quote_type": "usd_per_local"},
    "FTSE 100 (^FTSE)":     {"fx": "GBPUSD=X",    "quote_type": "usd_per_local"},
}

def to_usd_terms(series_local, fx_series, quote_type):
    if fx_series is None or series_local is None:
        return None
    s_loc = series_local.dropna()
    s_fx  = fx_series.dropna()
    if s_loc.empty or s_fx.empty:
        return None
    # 월말로 이미 맞춘 후 공통 교집합
    idx = s_loc.index.intersection(s_fx.index)
    if len(idx) == 0:
        return None
    s_loc = s_loc.loc[idx]
    s_fx  = s_fx.loc[idx]
    if quote_type == "usd_per_local":
        usd = s_loc * s_fx
    elif quote_type == "local_per_usd":
        usd = s_loc / s_fx
    else:
        usd = s_loc
    return usd.dropna()

# ============== Main ==============
def main():
    # ---------- Macro core ----------
    dgs10 = fred_series("DGS10", start=start_date)
    dgs2  = fred_series("DGS2", start=start_date)
    tips10= fred_series("DFII10", start=start_date)
    walcl = fred_series("WALCL", start=start_date)  # mn USD
    hyoas = fred_series("BAMLH0A0HYM2", start=start_date)

    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
    usdk_df, _       = yf_series_multi(["KRW=X"])
    vix_df, _        = yf_series_multi(["^VIX"])
    btc_df, _        = yf_series_multi(["BTC-USD", "XBT-USD", "BTCUSD=X"])

    # Monthly macro
    dgs10_m = eom(dgs10); dgs2_m = eom(dgs2); tips_m = eom(tips10)
    dxy_m = eom(dxy_df); usdk_m = eom(usdk_df); vix_m = eom(vix_df); hyoas_m = eom(hyoas)
    walcl_m = eom(walcl)  # mn USD

    be_m = (dgs10_m - tips_m) * 100.0 if dgs10_m is not None and tips_m is not None else None
    curve_m = (dgs10_m - dgs2_m) * 100.0 if dgs10_m is not None and dgs2_m is not None else None

    # ---------- Δ3M / Δ1Y summary ----------
    parts = []
    if tips_m is not None:
        d3 = delta_nm(tips_m, 3); d12 = delta_nm(tips_m, 12)
        parts += [pd.Series({"Real Δ3M (bp)": d3*100.0 if d3 is not None else np.nan,
                             "Real Δ1Y (bp)": d12*100.0 if d12 is not None else np.nan})]
    if curve_m is not None:
        d3 = delta_nm(curve_m, 3); d12 = delta_nm(curve_m, 12)
        parts += [pd.Series({"Curve Δ3M (bp)": d3, "Curve Δ1Y (bp)": d12})]
    if dxy_m is not None:
        d3 = delta_nm(dxy_m, 3); d12 = delta_nm(dxy_m, 12)
        parts += [pd.Series({"DXY Δ3M": d3, "DXY Δ1Y": d12})]
    if usdk_m is not None:
        d3 = delta_nm(usdk_m, 3); d12 = delta_nm(usdk_m, 12)
        parts += [pd.Series({"USDKRW Δ3M": d3, "USDKRW Δ1Y": d12})]
    if walcl_m is not None:
        walcl_trn_m = walcl_m / 1_000_000.0
        d3 = delta_nm(walcl_trn_m, 3); d12 = delta_nm(walcl_trn_m, 12)
        parts += [pd.Series({"Liquidity Δ3M (Trn)": d3, "Liquidity Δ1Y (Trn)": d12})]
    bar_ser = pd.concat(parts).dropna() if parts else None

    # ---------- Global assets ----------
    indices = {
        "S&P 500 (^GSPC)": "^GSPC",
        "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
        "Nikkei 225 (^N225)": "^N225",
        "Shanghai Composite (000001.SS)": "000001.SS",
        "Hang Seng (^HSI)": "^HSI",
        "KOSPI (^KS11)": "^KS11",
        "TAIEX (^TWII)": "^TWII",
        "Nifty 50 (^NSEI)": "^NSEI",
        "DAX (^GDAXI)": "^GDAXI",
        "FTSE 100 (^FTSE)": "^FTSE"
    }
    commodities = {
        "Gold (GC=F)": "GC=F",
        "Silver (SI=F)": "SI=F",
        "Crude Oil (CL=F)": "CL=F",
        "Copper (HG=F)": "HG=F",
        "Natural Gas (NG=F)": "NG=F"
    }

    # Local terms EOM
    local_eom = {}
    for name, tkr in indices.items():
        df, _ = yf_series_multi([tkr])
        if df is not None and not df.empty:
            s = eom(df)
            if s is not None and not s.empty:
                local_eom[name] = s

    # FX EOM
    fx_eom = {}
    for name, meta in FX_MAP.items():
        fx_tkr = meta["fx"]
        if fx_tkr:
            df_fx, _ = yf_series_multi([fx_tkr])
            if df_fx is not None and not df_fx.empty:
                fx_eom[name] = eom(df_fx)

    # USD-terms EOM
    usd_eom = {}
    for name, s_local in local_eom.items():
        meta = FX_MAP.get(name, {"fx": None, "quote_type": None})
        fx_s = fx_eom.get(name, None) if meta["fx"] else None
        usd_series = to_usd_terms(s_local, fx_s, meta["quote_type"])
        if usd_series is None and meta["fx"] is None:
            # already USD (e.g., S&P 500)
            usd_series = s_local.copy()
        if usd_series is not None and not usd_series.empty:
            usd_eom[name + " [USD]"] = usd_series

    # Commodities (already USD quoted)
    comm_eom = {}
    for name, tkr in commodities.items():
        df, _ = yf_series_multi([tkr])
        if df is not None and not df.empty:
            s = eom(df)
            if s is not None and not s.empty:
                comm_eom[name] = s

    # ===== Figures =====
    figs = []

    if dgs10_m is not None and dgs2_m is not None and len(dgs10_m) and len(dgs2_m):
        figs.append(("Rates — US10Y & US2Y",
                     fig_two_lines(
                         dgs10_m, dgs2_m, "US10Y", "US2Y",
                         "Yield (%)", unit="pct",
                         subtitle="장·단기 금리 비교 — 경기 싸이클 및 정책금리 기대 반영",
                         legend_pos="top-right", subtitle_y=1.34
                     )))
    if tips_m is not None and len(tips_m):
        figs.append(("10Y TIPS Real Yield",
                     fig_line(tips_m, "10Y TIPS Real Yield", "Yield (%)",
                              unit="pct",
                              subtitle="실질금리 — 금융 여건 및 할인율 지표 (상승 시 위험자산 압박)")))
    if be_m is not None and len(be_m):
        figs.append(("10Y Breakeven",
                     fig_line(be_m, "10Y Breakeven", "bps", unit="bps",
                              subtitle="시장 기대 인플레이션 — 향후 10년 평균 물가 기대")))
    if curve_m is not None and len(curve_m):
        figs.append(("10Y–2Y Curve",
                     fig_line(curve_m, "10Y–2Y Curve", "bps", unit="bps",
                              subtitle="수익률 곡선 — 장단기 금리차, 경기 침체 선행 신호")))
    if dxy_m is not None and len(dxy_m):
        dxy_title = "DXY (Dollar Index)" + (f" — used {dxy_used}" if dxy_used else "")
        figs.append(("DXY (Dollar Index)",
                     fig_line(dxy_m, dxy_title, "Index", unit="idx",
                              subtitle="달러 강세/약세 — 글로벌 자금 흐름 및 위험선호 지표")))
    if usdk_m is not None and len(usdk_m):
        figs.append(("USDKRW",
                     fig_line(usdk_m, "USDKRW", "KRW per USD", unit="fx",
                              subtitle="원화 환율 — 외국인 자금 유출입과 국내 Risk-On/Off 체감")))
    if vix_m is not None and len(vix_m):
        figs.append(("VIX",
                     fig_line(vix_m, "VIX", "Index", unit="idx",
                              subtitle="S&P500 변동성 (공포지수) — 시장 불확실성 척도")))
    if hyoas_m is not None and len(hyoas_m):
        figs.append(("HY OAS",
                     fig_line(hyoas_m, "HY OAS", "bps", unit="bps",
                              subtitle="하이일드 채권 스프레드 — 신용위험/유동성 스트레스")))

    bar_fig = fig_bar(bar_ser, "Latest Δ3M / Δ1Y", "Δ (unit per label)") if (bar_ser is not None and not bar_ser.empty) else None

    # ===== HTML assemble =====
    from plotly.offline import plot as plot_offline
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Macro Dashboard — Plotly</title>",
        "<style>body{font-family:-apple-system,Roboto,Segoe UI,Helvetica,Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 16px}",
        ".card{box-shadow:0 8px 28px rgba(0,0,0,.08);border-radius:14px;padding:16px;margin:20px 0}",
        "h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}p{line-height:1.55;color:#333}",
        ".grid{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:1100px){.grid{grid-template-columns:1fr 1fr}}",
        ".muted{color:#666;font-size:13px}</style></head><body>",
        f"<h1>Macro Dashboard — Plotly <span class='muted'>(Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')})</span></h1>"
    ]

    # Group: Rates
    group1_titles = {"Rates — US10Y & US2Y", "10Y TIPS Real Yield", "10Y Breakeven", "10Y–2Y Curve"}
    html_parts.append("<div class='card'><h2>Rates</h2><div class='grid'>")
    first = True
    for title, fig in figs:
        if title in group1_titles:
            html_parts.append(plot_offline(fig, include_plotlyjs='cdn' if first else False, output_type='div'))
            first = False
    html_parts.append("</div></div>")

    # Group: Dollar/FX & Credit
    html_parts.append("<div class='card'><h2>Dollar / FX & Credit</h2><div class='grid'>")
    for title, fig in figs:
        if title not in group1_titles:
            html_parts.append(plot_offline(fig, include_plotlyjs=False, output_type='div'))
    html_parts.append("</div></div>")

    if bar_fig is not None:
        html_parts.append("<div class='card'><h2>Latest Δ3M / Δ1Y</h2>")
        html_parts.append(plot_offline(bar_fig, include_plotlyjs=False, output_type='div'))
        html_parts.append("</div>")

    # Group: Global Equities (Local)
    local_figs = []
    for name, series in local_eom.items():
        local_figs.append((name, fig_line(series, name, "Index", unit="idx",
                                          subtitle=f"{name} — Local currency terms")))
    if local_figs:
        html_parts.append("<div class='card'><h2>Global Equities — Local</h2><div class='grid'>")
        for title, fig in local_figs:
            html_parts.append(plot_offline(fig, include_plotlyjs=False, output_type='div'))
        html_parts.append("</div></div>")

    # Group: Global Equities (USD Terms)
    usd_figs = []
    for name, series in usd_eom.items():
        usd_figs.append((name, fig_line(series, name, "Index (USD terms)", unit="idx",
                                        subtitle=f"{name} — USD terms (FX-adjusted)")))
    if usd_figs:
        html_parts.append("<div class='card'><h2>Global Equities — USD Terms</h2><div class='grid'>")
        for title, fig in usd_figs:
            html_parts.append(plot_offline(fig, include_plotlyjs=False, output_type='div'))
        html_parts.append("</div></div>")

    # Group: Commodities (USD quoted)
    if comm_eom:
        html_parts.append("<div class='card'><h2>Commodities</h2><div class='grid'>")
        for name, series in comm_eom.items():
            html_parts.append(plot_offline(fig_line(series, name, "Price (USD)", unit="idx",
                                                    subtitle=f"{name} — 주요 원자재 가격 추이"),
                                           include_plotlyjs=False, output_type='div'))
        html_parts.append("</div></div>")

    html_parts.append("</body></html>")
    HTML_PATH.write_text("\n".join(html_parts), encoding="utf-8")
    print("✅ Open:", HTML_PATH)

if __name__ == "__main__":
    main()
