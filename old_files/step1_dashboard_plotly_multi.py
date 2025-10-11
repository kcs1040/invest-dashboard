# step1_dashboard_plotly_multi.py
# -----------------------------------------------------------
# One script, three modes: --freq daily | weekly | monthly
# Tabs UI + Global Equities (Local & USD) + Commodities + Crypto
# - FRED + yfinance
# - Robust numeric coercion, frequency alignment
# - Δ window adapted per freq (3M/1Y equivalents)
# - Responsive Plotly + resize on tab switch
# -----------------------------------------------------------
import os, argparse, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# ===================== Config & CLI =====================
def parse_args():
    ap = argparse.ArgumentParser(description="Macro Dashboard (daily/weekly/monthly)")
    ap.add_argument("--freq", choices=["daily", "weekly", "monthly"], default="monthly",
                    help="frequency of charts and calculations")
    return ap.parse_args()

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing. Put it in .env like: FRED_API_KEY=YOUR_KEY")

args = parse_args()
FREQ = args.freq  # daily | weekly | monthly

OUTDIR = Path(f"dash_pro_{FREQ}")
OUTDIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUTDIR / "index.html"

today = dt.date.today()
start_date = (today - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# Δ windows by freq
DELTA_WINDOWS = {
    "daily":   (63, 252),
    "weekly":  (13, 52),
    "monthly": (3, 12),
}
DELTA_LBL = {
    "daily":   "(~63d / ~252d)",
    "weekly":  "(~13w / ~52w)",
    "monthly": "(3M / 12M)",
}
# Range selector buttons by freq
RANGE_BTNS = {
    "daily": [
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(step="all")
    ],
    "weekly": [
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=2, label="2Y", step="year", stepmode="backward"),
        dict(step="all")
    ],
    "monthly": [
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=2, label="2Y", step="year", stepmode="backward"),
        dict(step="all")
    ],
}

# ===================== Helpers (data) =====================
def to_dt_index(s):
    s.index = pd.to_datetime(s.index, errors="coerce")
    return s[~s.index.isna()]

def to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce").dropna()

def align_series_freq(s: pd.Series, freq: str) -> pd.Series | None:
    """Return numeric series aligned to desired frequency with sensible sampling."""
    if s is None: return None
    s = to_dt_index(pd.Series(s))
    s = to_numeric_series(s)
    if s.empty: return None

    if freq == "daily":
        # Align to business days with ffill (for FRED weekly/monthly); for equities we keep trading days as-is.
        # Here we do generic business day alignment + ffill.
        bdays = pd.date_range(s.index.min(), today, freq="B")
        s = s.reindex(bdays).ffill()
    elif freq == "weekly":
        # Week-ending Friday
        s = s.resample("W-FRI").last().dropna()
    elif freq == "monthly":
        # End-of-month
        s = s.resample("ME").last().dropna()
    return s

def fred_series(code, start=None):
    s = fred.get_series_latest_release(code)
    s = pd.Series(s)
    s = to_dt_index(s)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    s = to_numeric_series(s).astype(float)
    return s

def fred_series_aligned(code, start=None, freq="monthly"):
    s = fred_series(code, start=start)
    return align_series_freq(s, freq)

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

def close_series_aligned(df_or_series, freq="monthly") -> pd.Series | None:
    """Take yfinance df (Close) or Series; align to freq."""
    if df_or_series is None: return None
    if hasattr(df_or_series, "empty") and df_or_series.empty: return None
    if isinstance(df_or_series, pd.DataFrame):
        s = df_or_series["Close"] if "Close" in df_or_series.columns else df_or_series.iloc[:, 0]
        s = s.squeeze()
    else:
        s = pd.Series(df_or_series)
    s = to_dt_index(s)
    s = to_numeric_series(s)
    if s.empty: return None
    return align_series_freq(s, freq)

def num_series(s: pd.Series) -> pd.Series | None:
    if s is None:
        return None
    s = to_dt_index(pd.Series(s))
    s = to_numeric_series(s)
    return s if not s.empty else None

def delta_n(s, n):
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
# quote_type: 'usd_per_local'  (EURUSD=X, GBPUSD=X)  -> USD terms = local_index * FX
#              'local_per_usd' (KRW=X, JPY=X, CNY=X) -> USD terms = local_index / FX
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
        return series_local.copy()  # already USD
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

# ===================== Figures =====================
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
                rangeselector=dict(buttons=RANGE_BTNS[FREQ]),
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
                rangeselector=dict(buttons=RANGE_BTNS[FREQ]),
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

# ===================== Main =====================
def main():
    # ---------- Macro core ----------
    dgs10 = fred_series_aligned("DGS10",  start=start_date, freq=FREQ)
    dgs2  = fred_series_aligned("DGS2",   start=start_date, freq=FREQ)
    tips10= fred_series_aligned("DFII10", start=start_date, freq=FREQ)
    walcl = fred_series_aligned("WALCL",  start=start_date, freq=FREQ)  # mn USD
    hyoas = fred_series_aligned("BAMLH0A0HYM2", start=start_date, freq=FREQ)

    # Market data
    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
    usdk_df, _       = yf_series_multi(["KRW=X"])
    vix_df, _        = yf_series_multi(["^VIX"])

    dxy_s  = close_series_aligned(dxy_df,  freq=FREQ)
    usdk_s = close_series_aligned(usdk_df, freq=FREQ)
    vix_s  = close_series_aligned(vix_df,  freq=FREQ)

    # Derived
    be_s    = (dgs10 - tips10) * 100.0 if (dgs10 is not None and tips10 is not None) else None  # bps
    curve_s = (dgs10 - dgs2)   * 100.0 if (dgs10 is not None and dgs2  is not None) else None  # bps
    walcl_trn = walcl / 1_000_000.0 if walcl is not None else None  # Trillion USD

    # Δ summary
    n3, n12 = DELTA_WINDOWS[FREQ]
    parts = []
    if tips10 is not None:
        d3 = delta_n(tips10, n3); d12 = delta_n(tips10, n12)
        parts += [pd.Series({"Real Δ(3M)": (d3*100.0) if d3 is not None else np.nan,
                             "Real Δ(1Y)": (d12*100.0) if d12 is not None else np.nan})]
    if curve_s is not None:
        d3 = delta_n(curve_s, n3); d12 = delta_n(curve_s, n12)
        parts += [pd.Series({"Curve Δ(3M)": d3, "Curve Δ(1Y)": d12})]
    if dxy_s is not None:
        d3 = delta_n(dxy_s, n3); d12 = delta_n(dxy_s, n12)
        parts += [pd.Series({"DXY Δ(3M)": d3, "DXY Δ(1Y)": d12})]
    if usdk_s is not None:
        d3 = delta_n(usdk_s, n3); d12 = delta_n(usdk_s, n12)
        parts += [pd.Series({"USDKRW Δ(3M)": d3, "USDKRW Δ(1Y)": d12})]
    if walcl_trn is not None:
        d3 = delta_n(walcl_trn, n3); d12 = delta_n(walcl_trn, n12)
        parts += [pd.Series({"Liquidity Δ(3M) [Trn]": d3, "Liquidity Δ(1Y) [Trn]": d12})]
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

    local_map = {}
    for name, tkr in indices.items():
        df, _ = yf_series_multi([tkr])
        s = close_series_aligned(df, freq=FREQ)
        if s is not None: local_map[name] = s

    # FX for USD rebasing
    fx_map = {}
    for name, meta in FX_MAP.items():
        if meta["fx"]:
            df_fx, _ = yf_series_multi([meta["fx"]])
            sfx = close_series_aligned(df_fx, freq=FREQ)
            if sfx is not None: fx_map[name] = sfx

    usd_map = {}
    for name, s_local in local_map.items():
        meta = FX_MAP.get(name, {"fx": None, "quote_type": None})
        fx_s = fx_map.get(name) if meta["fx"] else None
        s_usd = to_usd_terms(s_local, fx_s, meta["quote_type"])
        if s_usd is None and meta["fx"] is None:
            s_usd = s_local.copy()
        if s_usd is not None and not s_usd.empty:
            usd_map[name + " [USD]"] = s_usd

    comm_map = {}
    for name, tkr in commodities.items():
        df, _ = yf_series_multi([tkr])
        s = close_series_aligned(df, freq=FREQ)
        if s is not None: comm_map[name] = s

    # ---------- Crypto (linear y) ----------
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

    def crypto_map(symbols):
        out = {}
        for nm, tkr in symbols.items():
            df, _ = yf_series_multi([tkr])
            s = close_series_aligned(df, freq=FREQ)
            if s is not None: out[nm] = s
        return out

    majors_map = crypto_map(crypto_majors)
    alts_map   = crypto_map(crypto_alts)

    # ===== Figures (Overview) =====
    freq_tag = {"daily":"(Daily)", "weekly":"(Weekly)", "monthly":"(Monthly)"}[FREQ]
    figs_overview = []
    if dgs10 is not None and dgs2 is not None and len(dgs10) and len(dgs2):
        figs_overview.append((f"Rates — US10Y & US2Y {freq_tag}",
                              fig_two_lines(dgs10, dgs2, "US10Y", "US2Y",
                                            "Yield (%)", unit="pct",
                                            subtitle=f"장·단기 금리 — {FREQ} 빈도",
                                            legend_pos="top-right", subtitle_y=1.34)))
    if tips10 is not None and len(tips10):
        figs_overview.append((f"10Y TIPS Real Yield {freq_tag}",
                              fig_line(tips10, f"10Y TIPS Real Yield {freq_tag}", "Yield (%)",
                                       unit="pct", subtitle=f"실질금리 — {FREQ} 빈도")))
    if be_s is not None and len(be_s):
        figs_overview.append((f"10Y Breakeven {freq_tag}",
                              fig_line(be_s, f"10Y Breakeven {freq_tag}", "bps", unit="bps",
                                       subtitle=f"시장 기대 인플레이션 — {FREQ} 빈도")))
    if curve_s is not None and len(curve_s):
        figs_overview.append((f"10Y–2Y Curve {freq_tag}",
                              fig_line(curve_s, f"10Y–2Y Curve {freq_tag}", "bps", unit="bps",
                                       subtitle=f"수익률 곡선 — {FREQ} 빈도")))
    if dxy_s is not None and len(dxy_s):
        dxy_title = f"DXY (Dollar Index) {freq_tag}" + (f" — used {dxy_used}" if dxy_used else "")
        figs_overview.append((f"DXY {freq_tag}",
                              fig_line(dxy_s, dxy_title, "Index", unit="idx",
                                       subtitle=f"달러 인덱스 — {FREQ} 빈도")))
    if usdk_s is not None and len(usdk_s):
        figs_overview.append((f"USDKRW {freq_tag}",
                              fig_line(usdk_s, f"USDKRW {freq_tag}", "KRW per USD", unit="fx",
                                       subtitle=f"원/달러 환율 — {FREQ} 빈도")))
    if vix_s is not None and len(vix_s):
        figs_overview.append((f"VIX {freq_tag}",
                              fig_line(vix_s, f"VIX {freq_tag}", "Index", unit="idx",
                                       subtitle=f"S&P500 변동성 — {FREQ} 빈도")))
    if hyoas is not None and len(hyoas):
        figs_overview.append((f"HY OAS {freq_tag}",
                              fig_line(hyoas, f"HY OAS {freq_tag}", "bps", unit="bps",
                                       subtitle=f"하이일드 스프레드 — {FREQ} 빈도")))
    if walcl_trn is not None and len(walcl_trn):
        figs_overview.append((f"Fed Balance Sheet (Trn) {freq_tag}",
                              fig_line(walcl_trn, f"Fed Balance Sheet (Trn USD) {freq_tag}", "Trn USD", unit="idx",
                                       subtitle=f"연준 대차대조표 — {FREQ} 빈도")))

    bar_fig = None
    if bar_ser is not None and not bar_ser.empty:
        bar_fig = fig_bar(bar_ser, f"Latest Δ {DELTA_LBL[FREQ]}", "Δ (unit per label)")

    # ===== Global Equities — Local =====
    figs_local = [(nm, fig_line(s, nm + f" {freq_tag} (Local)", "Index", unit="idx",
                                subtitle=f"{nm} — Local currency {freq_tag}"))
                  for nm, s in local_map.items()]

    # ===== Global Equities — USD Terms =====
    figs_usd = [(nm, fig_line(s, nm + f" {freq_tag} (USD terms)", "Index (USD terms)", unit="idx",
                              subtitle=f"{nm} — USD terms {freq_tag} (FX-adjusted)"))
                for nm, s in usd_map.items()]

    # ===== Commodities =====
    figs_comm = [(nm, fig_line(s, nm + f" {freq_tag}", "Price (USD)", unit="idx",
                               subtitle=f"{nm} — {FREQ} 빈도"))
                 for nm, s in comm_map.items()]

    # ===== Crypto (linear y) =====
    figs_cmaj = []
    for name, series in majors_map.items():
        title = name.replace("-USD", "")
        figs_cmaj.append((title, fig_line(series, title + f" {freq_tag}", "Price (USD)", unit="idx",
                                          subtitle=f"{title} — Major crypto {freq_tag}", log_y=False)))
    figs_calt = []
    for name, series in alts_map.items():
        title = name.replace("-USD", "")
        figs_calt.append((title, fig_line(series, title + f" {freq_tag}", "Price (USD)", unit="idx",
                                          subtitle=f"{title} — Altcoin {freq_tag}", log_y=False)))

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
        f"<title>Macro Dashboard — Plotly ({FREQ.capitalize()})</title>",
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
        f"<h1>Macro Dashboard — Plotly <span class='muted'>({FREQ.capitalize()} • Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')})</span></h1>",
        # ---------- Tabs ----------
        "<div class='tabs'>",
        f"<button class='tabbtn active' data-target='tab-overview' onclick=\"showTab('tab-overview')\">Overview (Macro, {FREQ.capitalize()})</button>",
        f"<button class='tabbtn' data-target='tab-global-local' onclick=\"showTab('tab-global-local')\">Global Equities (Local, {FREQ.capitalize()})</button>",
        f"<button class='tabbtn' data-target='tab-global-usd' onclick=\"showTab('tab-global-usd')\">Global Equities (USD, {FREQ.capitalize()})</button>",
        f"<button class='tabbtn' data-target='tab-commodities' onclick=\"showTab('tab-commodities')\">Commodities ({FREQ.capitalize()})</button>",
        f"<button class='tabbtn' data-target='tab-crypto-majors' onclick=\"showTab('tab-crypto-majors')\">Crypto (Majors, {FREQ.capitalize()})</button>",
        f"<button class='tabbtn' data-target='tab-crypto-alts' onclick=\"showTab('tab-crypto-alts')\">Crypto (Alts, {FREQ.capitalize()})</button>",
        "</div>",
    ]

    # ---- Overview ----
    from plotly.offline import plot as plot_offline
    def plot_div_first(fig):
        return plot_offline(fig, include_plotlyjs='cdn', output_type='div', config={'responsive': True})

    ov_html = ["<div class='tab-section' id='tab-overview' style='display:none'>"]
    # Rates
    ov_html.append("<div class='card'><h2>Rates</h2><div class='grid'>")
    first = True
    for t, f in figs_overview:
        if ("Rates" in t) or ("TIPS" in t) or ("Breakeven" in t) or ("Curve" in t):
            ov_html.append(plot_div_first(f) if first else plot_div(f))
            first = False
    ov_html.append("</div></div>")
    # Dollar/FX & Credit & Liquidity
    ov_html.append("<div class='card'><h2>Dollar / FX & Credit / Liquidity</h2><div class='grid'>")
    for t, f in figs_overview:
        if not (("Rates" in t) or ("TIPS" in t) or ("Breakeven" in t) or ("Curve" in t)):
            ov_html.append(plot_div(f))
    ov_html.append("</div></div>")
    # Δ panel
    if bar_fig is not None:
        ov_html.append(f"<div class='card'><h2>Latest Δ {DELTA_LBL[FREQ]}</h2>")
        ov_html.append(plot_div(bar_fig))
        ov_html.append("</div>")
    ov_html.append("</div>")
    html_parts.append("\n".join(ov_html))

    # ---- Other tabs ----
    html_parts.append(section_html("tab-global-local", f"Global Equities — Local {freq_tag}", figs_local))
    html_parts.append(section_html("tab-global-usd",  f"Global Equities — USD Terms {freq_tag}", figs_usd))
    html_parts.append(section_html("tab-commodities", f"Commodities {freq_tag}", figs_comm))
    html_parts.append(section_html("tab-crypto-majors", f"Crypto (Majors) {freq_tag}", figs_cmaj))
    html_parts.append(section_html("tab-crypto-alts",   f"Crypto (Alts) {freq_tag}", figs_calt))

    html_parts.append("</body></html>")
    HTML_PATH.write_text("\n".join(html_parts), encoding="utf-8")
    print("✅ Open:", HTML_PATH)

if __name__ == "__main__":
    main()
