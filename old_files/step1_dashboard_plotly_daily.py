# step1_dashboard_plotly_daily.py
# -----------------------------------------------------------
# Interactive Macro Dashboard (Plotly) — DAILY version
# Tabs UI + Global Equities (Local & USD terms) + Commodities + Crypto (Majors/Alts)
# - ~3y data from FRED + yfinance (with fallbacks)
# - DAILY alignment with robust numeric coercion & ffill for mixed frequencies
# - Δ changes using trading-day approximations: 63d(~3M), 252d(~1Y)
# - Responsive Plotly + resize on tab switch
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

OUTDIR = Path(os.getenv("OUTDIR_DAILY", "dash_pro_daily"))
OUTDIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUTDIR / "index.html"

today = dt.date.today()
start_date = (today - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# ========= Helpers: numeric coercion & daily alignment =========
def to_dt_index(s):
    s.index = pd.to_datetime(s.index, errors="coerce")
    return s[~s.index.isna()]

def to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce").dropna()

def fred_series_daily(code, start=None) -> pd.Series:
    """FRED latest release -> numeric -> DAILY (business day) with ffill."""
    s = fred.get_series_latest_release(code)
    s = pd.Series(s)
    s = to_dt_index(s).dropna()
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    s = to_numeric_series(s).astype(float)
    # align to business days, forward fill (handles weekly/monthly series too)
    bdays = pd.date_range(s.index.min(), today, freq="B")
    s = s.reindex(bdays).ffill()
    s.index.name = None
    return s

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

def close_daily(df_or_series) -> pd.Series | None:
    """Get daily Close series -> numeric, DatetimeIndex, business day reindex (no forward fill for gaps across weekends)."""
    if df_or_series is None:
        return None
    if hasattr(df_or_series, "empty") and df_or_series.empty:
        return None
    if isinstance(df_or_series, pd.DataFrame):
        s = df_or_series["Close"] if "Close" in df_or_series.columns else df_or_series.iloc[:, 0]
        s = s.squeeze()
    else:
        s = pd.Series(df_or_series)
    s = to_dt_index(s)
    s = to_numeric_series(s)
    if s.empty:
        return None
    # keep trading days as-is (different markets have different calendars)
    return s

def num_series(s: pd.Series) -> pd.Series | None:
    if s is None:
        return None
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    s = to_dt_index(s)
    s = to_numeric_series(s)
    if s.empty:
        return None
    return s

def delta_ndays(s, n):
    """Δ over n trading days (approx)"""
    s = num_series(s)
    if s is None or len(s) < (n+1):
        return None
    v = s.iloc[-1] - s.iloc[-(n+1)]
    return float(v.item() if hasattr(v, "item") else v)

def last_label(ts, unit=""):
    ts = num_series(ts)
    if ts is None or ts.empty: return "—"
    v = ts.iloc[-1]
    if unit == "bps": return f"{float(v):,.0f} bps"
    if unit == "pct": return f"{float(v):,.2f}%"
    if unit == "idx": return f"{float(v):,.2f}"
    if unit == "fx":  return f"{float(v):,.2f}"
    return f"{float(v):,.2f}"

# ===================== FX mapping for USD rebasing =====================
# quote_type: 'usd_per_local'  (e.g., EURUSD=X, GBPUSD=X)  -> USD terms = local_index * FX
#              'local_per_usd' (e.g., KRW=X, JPY=X, CNY=X) -> USD terms = local_index / FX
FX_MAP = {
    "S&P 500 (^GSPC)":      {"fx": None,          "quote_type": None},
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
    series_local = num_series(series_local)
    if series_local is None:
        return None
    if fx_series is None and quote_type is None:
        return series_local.copy()  # already USD (e.g., S&P 500)
    fx_series = num_series(fx_series)
    if fx_series is None or fx_series.empty:
        return None
    idx = series_local.index.intersection(fx_series.index)
    if len(idx) == 0:
        return None
    s_loc = series_local.loc[idx]
    s_fx  = fx_series.loc[idx]
    if quote_type == "usd_per_local":
        usd = s_loc * s_fx
    elif quote_type == "local_per_usd":
        usd = s_loc / s_fx
    else:
        usd = s_loc
    usd = pd.to_numeric(usd, errors="coerce").dropna()
    return usd

# ============== Figure builders (with subtitles) ==============
def fig_line(ts, name, yaxis_title, unit=None, show_range_slider=True, subtitle=None, log_y=False):
    ts = num_series(ts)
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
    fig.update_yaxes(type="log" if log_y else "linear")

    if subtitle:
        fig.add_annotation(
            text=f"<b style='color:#666;font-size:13px'>{subtitle}</b>",
            xref="paper", yref="paper", x=0.0, y=1.30, showarrow=False
        )

    if show_range_slider:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

    if ts is not None and len(ts):
        fig.add_annotation(x=ts.index[-1], y=ts.values[-1], text=last_label(ts, unit),
                           showarrow=True, arrowhead=2, ax=20, ay=-20)
    return fig

def fig_two_lines(ts1, ts2, name1, name2, yaxis_title, unit=None,
                  show_range_slider=True, subtitle=None, legend_pos="top-right", subtitle_y=1.32):
    ts1 = num_series(ts1); ts2 = num_series(ts2)
    idx = ts1.index.intersection(ts2.index)
    ts1 = ts1.loc[idx]; ts2 = ts2.loc[idx]

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
    fig.update_yaxes(type="linear")

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
        fig.add_annotation(text=f"<b style='color:#666;font-size:13px'>{subtitle}</b>",
                           xref="paper", yref="paper", x=0.0, y=subtitle_y, showarrow=False)

    if show_range_slider:
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

    if len(ts1):
        fig.add_annotation(x=ts1.index[-1], y=ts1.values[-1], text=last_label(ts1, unit),
                           showarrow=True, arrowhead=2, ax=20, ay=-20)
    if len(ts2):
        fig.add_annotation(x=ts2.index[-1], y=ts2.values[-1], text=last_label(ts2, unit),
                           showarrow=True, arrowhead=2, ax=20, ay=20)
    return fig

def fig_bar(series, title, yaxis_title):
    s = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    fig = go.Figure(go.Bar(
        x=pd.Index(s.index).astype(str), y=s.values,
        text=[f"{v:,.2f}" for v in s.values], textposition="auto"
    ))
    fig.update_layout(
        height=420, margin=dict(l=60, r=30, t=50, b=80),
        yaxis_title=yaxis_title, template="plotly_white",
        xaxis=dict(tickangle=15)
    )
    fig.update_yaxes(type="linear")
    return fig

# ============== Main ==============
def main():
    # ---------- Macro (DAILY) ----------
    dgs10 = fred_series_daily("DGS10", start=start_date)
    dgs2  = fred_series_daily("DGS2",  start=start_date)
    tips10= fred_series_daily("DFII10", start=start_date)
    walcl = fred_series_daily("WALCL",  start=start_date)  # mn USD (weekly -> business-day ffilled)
    hyoas = fred_series_daily("BAMLH0A0HYM2", start=start_date)  # OAS daily/weekly -> ffilled

    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"], period="3y", interval="1d")
    usdk_df, _       = yf_series_multi(["KRW=X"], period="3y", interval="1d")
    vix_df, _        = yf_series_multi(["^VIX"], period="3y", interval="1d")
    btc_df, _        = yf_series_multi(["BTC-USD", "XBT-USD", "BTCUSD=X"], period="3y", interval="1d")

    dxy_d  = close_daily(dxy_df); usdk_d = close_daily(usdk_df)
    vix_d  = close_daily(vix_df); btc_d  = close_daily(btc_df)

    # Derived (DAILY)
    be_d    = (dgs10 - tips10) * 100.0 if (dgs10 is not None and tips10 is not None) else None  # bps
    curve_d = (dgs10 - dgs2)   * 100.0 if (dgs10 is not None and dgs2  is not None) else None  # bps
    walcl_trn_d = walcl / 1_000_000.0 if walcl is not None else None

    # Δ blocks (63d ~ 3M, 252d ~ 1Y)
    parts = []
    if tips10 is not None:
        d63 = delta_ndays(tips10, 63); d252 = delta_ndays(tips10, 252)
        parts += [pd.Series({"Real Δ3M (bp, ~63d)": (d63*100.0) if d63 is not None else np.nan,
                             "Real Δ1Y (bp, ~252d)": (d252*100.0) if d252 is not None else np.nan})]
    if curve_d is not None:
        d63 = delta_ndays(curve_d, 63); d252 = delta_ndays(curve_d, 252)
        parts += [pd.Series({"Curve Δ3M (bp, ~63d)": d63, "Curve Δ1Y (bp, ~252d)": d252})]
    if dxy_d is not None:
        d63 = delta_ndays(dxy_d, 63); d252 = delta_ndays(dxy_d, 252)
        parts += [pd.Series({"DXY Δ3M (~63d)": d63, "DXY Δ1Y (~252d)": d252})]
    if usdk_d is not None:
        d63 = delta_ndays(usdk_d, 63); d252 = delta_ndays(usdk_d, 252)
        parts += [pd.Series({"USDKRW Δ3M (~63d)": d63, "USDKRW Δ1Y (~252d)": d252})]
    if walcl_trn_d is not None:
        d63 = delta_ndays(walcl_trn_d, 63); d252 = delta_ndays(walcl_trn_d, 252)
        parts += [pd.Series({"Liquidity Δ3M (Trn, ~63d)": d63, "Liquidity Δ1Y (Trn, ~252d)": d252})]
    bar_ser = pd.concat(parts).dropna() if parts else None

    # ---------- Global assets (DAILY) ----------
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

    local_daily = {}
    for name, tkr in indices.items():
        df, _ = yf_series_multi([tkr], period="3y", interval="1d")
        s = close_daily(df)
        if s is not None: local_daily[name] = s

    fx_daily = {}
    for name, meta in FX_MAP.items():
        if meta["fx"]:
            df_fx, _ = yf_series_multi([meta["fx"]], period="3y", interval="1d")
            sfx = close_daily(df_fx)
            if sfx is not None: fx_daily[name] = sfx

    usd_daily = {}
    for name, s_local in local_daily.items():
        meta = FX_MAP.get(name, {"fx": None, "quote_type": None})
        fx_s = fx_daily.get(name) if meta["fx"] else None
        s_usd = to_usd_terms(s_local, fx_s, meta["quote_type"])
        if s_usd is None and meta["fx"] is None:
            s_usd = s_local.copy()
        if s_usd is not None and not s_usd.empty:
            usd_daily[name + " [USD]"] = s_usd

    comm_daily = {}
    for name, tkr in commodities.items():
        df, _ = yf_series_multi([tkr], period="3y", interval="1d")
        s = close_daily(df)
        if s is not None: comm_daily[name] = s

    # ---------- Crypto (DAILY, linear y) ----------
    crypto_majors = {
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD",
        "BNB-USD": "BNB-USD",
        "SOL-USD": "SOL-USD",
        "XRP-USD": "XRP-USD",
    }
    crypto_alts = {
        "ADA-USD": "ADA-USD",
        "DOGE-USD": "DOGE-USD",
        "AVAX-USD": "AVAX-USD",
        "LINK-USD": "LINK-USD",
        "LTC-USD": "LTC-USD",
    }

    def crypto_map_daily(symbols):
        out = {}
        for nm, tkr in symbols.items():
            df, _ = yf_series_multi([tkr], period="3y", interval="1d")
            s = close_daily(df)
            if s is not None: out[nm] = s
        return out

    majors_d = crypto_map_daily(crypto_majors)
    alts_d   = crypto_map_daily(crypto_alts)

    # ===== Figures (Overview Daily) =====
    figs_overview = []
    if dgs10 is not None and dgs2 is not None and len(dgs10) and len(dgs2):
        figs_overview.append(("Rates — US10Y & US2Y (Daily)",
                              fig_two_lines(dgs10, dgs2, "US10Y", "US2Y",
                                            "Yield (%)", unit="pct",
                                            subtitle="일별 장단기 금리 — 정책/경기 기대 반영",
                                            legend_pos="top-right", subtitle_y=1.34)))
    if tips10 is not None and len(tips10):
        figs_overview.append(("10Y TIPS Real Yield (Daily)",
                              fig_line(tips10, "10Y TIPS Real Yield (Daily)", "Yield (%)",
                                       unit="pct", subtitle="실질금리(일별) — 금융여건/할인율")))
    if be_d is not None and len(be_d):
        figs_overview.append(("10Y Breakeven (Daily)",
                              fig_line(be_d, "10Y Breakeven (Daily)", "bps", unit="bps",
                                       subtitle="시장 기대 인플레이션(일별)")))
    if curve_d is not None and len(curve_d):
        figs_overview.append(("10Y–2Y Curve (Daily)",
                              fig_line(curve_d, "10Y–2Y Curve (Daily)", "bps", unit="bps",
                                       subtitle="수익률 곡선(일별)")))
    if dxy_d is not None and len(dxy_d):
        dxy_title = "DXY (Dollar Index, Daily)" + (f" — used {dxy_used}" if dxy_used else "")
        figs_overview.append(("DXY (Daily)",
                              fig_line(dxy_d, dxy_title, "Index", unit="idx",
                                       subtitle="달러 인덱스(일별)")))
    if usdk_d is not None and len(usdk_d):
        figs_overview.append(("USDKRW (Daily)",
                              fig_line(usdk_d, "USDKRW (Daily)", "KRW per USD", unit="fx",
                                       subtitle="원/달러 환율(일별)")))
    if vix_d is not None and len(vix_d):
        figs_overview.append(("VIX (Daily)",
                              fig_line(vix_d, "VIX (Daily)", "Index", unit="idx",
                                       subtitle="S&P500 변동성(일별)")))
    if hyoas is not None and len(hyoas):
        figs_overview.append(("HY OAS (Daily, ffilled)",
                              fig_line(hyoas, "HY OAS (Daily, ffilled)", "bps", unit="bps",
                                       subtitle="하이일드 스프레드(주/월→일 보간)")))
    if walcl_trn_d is not None and len(walcl_trn_d):
        figs_overview.append(("Fed Balance Sheet (Trn, Daily, ffilled)",
                              fig_line(walcl_trn_d, "Fed Balance Sheet (Trn USD, Daily)", "Trn USD", unit="idx",
                                       subtitle="연준 대차대조표(주간→일 보간)")))

    bar_fig = fig_bar(bar_ser, "Latest Δ(63d/252d) — Daily", "Δ (unit per label)") if (bar_ser is not None and not bar_ser.empty) else None

    # ===== Global Equities — Local (Daily) =====
    figs_local = [(nm, fig_line(s, nm + " (Daily, Local)", "Index", unit="idx",
                                subtitle=f"{nm} — Local currency (Daily)"))
                  for nm, s in local_daily.items()]

    # ===== Global Equities — USD Terms (Daily) =====
    figs_usd = [(nm, fig_line(s, nm + " (Daily, USD terms)", "Index (USD terms)", unit="idx",
                              subtitle=f"{nm} — USD terms (Daily, FX-adjusted)"))
                for nm, s in usd_daily.items()]

    # ===== Commodities (Daily) =====
    figs_comm = [(nm, fig_line(s, nm + " (Daily)", "Price (USD)", unit="idx",
                               subtitle=f"{nm} — 일별 가격 추이"))
                 for nm, s in comm_daily.items()]

    # ===== Crypto (Daily, linear y) =====
    figs_cmaj = []
    for name, series in majors_d.items():
        title = name.replace("-USD", "")
        figs_cmaj.append((title, fig_line(series, title + " (Daily)", "Price (USD)", unit="idx",
                                          subtitle=f"{title} — Major crypto (Daily)", log_y=False)))
    figs_calt = []
    for name, series in alts_d.items():
        title = name.replace("-USD", "")
        figs_calt.append((title, fig_line(series, title + " (Daily)", "Price (USD)", unit="idx",
                                          subtitle=f"{title} — Altcoin (Daily)", log_y=False)))

    # ===== HTML assemble with TABS =====
    from plotly.offline import plot as plot_offline

    def plot_div(fig, include_js=False):
        return plot_offline(fig, include_plotlyjs=('cdn' if include_js else False),
                            output_type='div', config={'responsive': True})

    def section_html(section_id, title, figs):
        parts = [f"<div class='tab-section' id='{section_id}' style='display:none'>"]
        parts.append(f"<div class='card'><h2>{title}</h2><div class='grid'>")
        for t, f in figs:
            parts.append(plot_div(f, include_js=False))
        parts.append("</div></div></div>")
        return "\n".join(parts)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Macro Dashboard — Plotly (Daily)</title>",
        # ---------- Styles ----------
        "<style>",
        "body{font-family:-apple-system,Roboto,Segoe UI,Helvetica,Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 16px}",
        ".card{box-shadow:0 8px 28px rgba(0,0,0,.08);border-radius:14px;padding:16px;margin:20px 0}",
        "h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}p{line-height:1.55;color:#333}",
        ".grid{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:1100px){.grid{grid-template-columns:1fr 1fr}}",
        ".grid > div { min-width: 0; }",
        ".muted{color:#666;font-size:13px}",
        ".tabs{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 20px}",
        ".tabbtn{padding:8px 12px;border:1px solid #ddd;border-radius:10px;background:#f8f9fb;cursor:pointer}",
        ".tabbtn.active{background:#2b6cb0;color:#fff;border-color:#2b6cb0}",
        "</style>",
        # ---------- Tabs + Resize Script ----------
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<script>",
        "function resizeSection(id){",
        "  const sec = document.getElementById(id);",
        "  if(!sec) return;",
        "  const plots = sec.querySelectorAll('.plotly-graph-div');",
        "  plots.forEach(gd => {",
        "    try { Plotly.Plots.resize(gd); Plotly.relayout(gd, {autosize: true}); } catch(e){}",
        "  });",
        "}",
        "function showTab(id){",
        "  const sections=document.querySelectorAll('.tab-section');",
        "  sections.forEach(s=>s.style.display='none');",
        "  const tabbtns=document.querySelectorAll('.tabbtn');",
        "  tabbtns.forEach(b=>b.classList.remove('active'));",
        "  const sec=document.getElementById(id); if(sec){ sec.style.display='block'; }",
        "  const btn=document.querySelector(`[data-target='${id}']`); if(btn) btn.classList.add('active');",
        "  setTimeout(()=>resizeSection(id), 30);",
        "  window.scrollTo({top:0,behavior:'smooth'});",
        "}",
        "window.addEventListener('resize', ()=>{",
        "  const visible = Array.from(document.querySelectorAll('.tab-section')).find(el => el.style.display !== 'none');",
        "  if(visible) resizeSection(visible.id);",
        "});",
        "document.addEventListener('DOMContentLoaded',()=>{ showTab('tab-overview'); });",
        "</script>",
        "</head><body>",
        f"<h1>Macro Dashboard — Plotly (Daily) <span class='muted'>(Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')})</span></h1>",
        # ---------- Tabs ----------
        "<div class='tabs'>",
        "<button class='tabbtn active' data-target='tab-overview' onclick=\"showTab('tab-overview')\">Overview (Macro, Daily)</button>",
        "<button class='tabbtn' data-target='tab-global-local' onclick=\"showTab('tab-global-local')\">Global Equities (Local, Daily)</button>",
        "<button class='tabbtn' data-target='tab-global-usd' onclick=\"showTab('tab-global-usd')\">Global Equities (USD, Daily)</button>",
        "<button class='tabbtn' data-target='tab-commodities' onclick=\"showTab('tab-commodities')\">Commodities (Daily)</button>",
        "<button class='tabbtn' data-target='tab-crypto-majors' onclick=\"showTab('tab-crypto-majors')\">Crypto (Majors, Daily)</button>",
        "<button class='tabbtn' data-target='tab-crypto-alts' onclick=\"showTab('tab-crypto-alts')\">Crypto (Alts, Daily)</button>",
        "</div>",
    ]

    # ---- Overview (Macro, Daily) ----
    from plotly.offline import plot as plot_offline
    def plot_div_first(fig):
        return plot_offline(fig, include_plotlyjs='cdn', output_type='div', config={'responsive': True})

    ov_html = ["<div class='tab-section' id='tab-overview' style='display:none'>"]
    # Rates
    ov_html.append("<div class='card'><h2>Rates</h2><div class='grid'>")
    first = True
    for t, f in figs_overview:
        if "Rates" in t or "TIPS" in t or "Breakeven" in t or "Curve" in t:
            ov_html.append(plot_div_first(f) if first else plot_div(f))
            first = False
    ov_html.append("</div></div>")
    # Dollar/FX & Credit & Fed BS
    ov_html.append("<div class='card'><h2>Dollar / FX & Credit / Liquidity</h2><div class='grid'>")
    for t, f in figs_overview:
        if not ("Rates" in t or "TIPS" in t or "Breakeven" in t or "Curve" in t):
            ov_html.append(plot_div(f))
    ov_html.append("</div></div>")
    # Δ panel
    if bar_fig is not None:
        ov_html.append("<div class='card'><h2>Latest Δ (63d / 252d)</h2>")
        ov_html.append(plot_div(bar_fig))
        ov_html.append("</div>")
    ov_html.append("</div>")
    html_parts.append("\n".join(ov_html))

    # ---- Other tabs ----
    html_parts.append(section_html("tab-global-local", "Global Equities — Local (Daily)", figs_local))
    html_parts.append(section_html("tab-global-usd", "Global Equities — USD Terms (Daily)", figs_usd))
    html_parts.append(section_html("tab-commodities", "Commodities (Daily)", figs_comm))
    html_parts.append(section_html("tab-crypto-majors", "Crypto (Majors, Daily)", figs_cmaj))
    html_parts.append(section_html("tab-crypto-alts", "Crypto (Alts, Daily)", figs_calt))

    html_parts.append("</body></html>")
    HTML_PATH.write_text("\n".join(html_parts), encoding="utf-8")
    print("✅ Open:", HTML_PATH)

if __name__ == "__main__":
    main()
