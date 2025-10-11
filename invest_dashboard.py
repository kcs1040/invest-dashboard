# step1_dashboard_plotly_per_chart_freq.py
# -----------------------------------------------------------
# Per-chart frequency toggles (Daily / Weekly / Monthly)
# Tabs: Overview, Global Equities (Local/USD), Commodities, Crypto (Majors/Alts)
# UI: Freq dropdown (left-top, outside y=1.33), Range selector (right-top, outside y=1.33),
#     Subtitle y=1.50, Legend inside top-right (two-line), consistent layout (height=500, t=200)
# Data: FRED + yfinance (원본 주파수 그대로 사용 — 빈 날짜 채우기 제거)
# JS: 안정적인 x축 '오늘로 클램프' (무한 루프 방지)
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

OUTDIR = Path(os.getenv("OUTDIR_PERCHART", "dash_pro_perchart"))
OUTDIR.mkdir(parents=True, exist_ok=True)
HTML_PATH = OUTDIR / "index.html"

today = dt.date.today()
start_date = (today - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# ============== Helpers (data) ==============
def to_dt_index(s):
    s.index = pd.to_datetime(s.index, errors="coerce")
    return s[~s.index.isna()]

def to_numeric_series(s):
    return pd.to_numeric(s, errors="coerce").dropna()

def num_series(s: pd.Series) -> pd.Series | None:
    if s is None:
        return None
    if isinstance(s, pd.DataFrame):
        if "Close" in s.columns:
            s = s["Close"]
        else:
            s = s.iloc[:, 0]
        s = s.squeeze("columns")
    if isinstance(s, (np.ndarray, list, tuple)):
        arr = np.asarray(s)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        s = pd.Series(arr)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    s = s.copy()
    try:
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()]
    except Exception:
        pass
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s if not s.empty else None

def fred_series(code, start=None):
    s = fred.get_series_latest_release(code)
    s = pd.Series(s)
    s = to_dt_index(s)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    s = to_numeric_series(s).astype(float)
    return s

def fred_series_all_freq(code, start=None):
    """
    {'daily': 원시 일별(가능하면 영업일로 채움 없이 원본), 
     'weekly': W-FRI last, 
     'monthly': ME last}
    """
    s = fred_series(code, start=start)
    if s is None or s.empty:
        return {'daily': None, 'weekly': None, 'monthly': None}

    # FRED는 일별 관측치가 비는 날이 있음 → 'daily'는 원본 그대로(보간/ffill 하지 않음)
    d = s.copy()

    # 주/월: 원본을 기준으로 다운샘플
    w = s.resample("W-FRI").last().dropna()
    m = s.resample("ME").last().dropna()
    return {'daily': d, 'weekly': w, 'monthly': m}

def yf_series_multi(candidates, period="3y", interval="1d"):
    for tkr in candidates:
        try:
            df = yf.download(
                tkr, period=period, interval=interval,
                progress=False, auto_adjust=False,
                group_by="column"  # prevent MultiIndex columns
            )
            if df is not None and not df.empty and "Close" in df.columns:
                return df, tkr
        except Exception:
            pass
    return None, None

def market_series_all_freq(df_or_symbol):
    """yfinance df or symbol -> dict of {'daily','weekly','monthly'} Close series. (보간/ffill 없음)"""
    if isinstance(df_or_symbol, str):
        df, _ = yf_series_multi([df_or_symbol])
        if df is None or df.empty:
            return {'daily': None, 'weekly': None, 'monthly': None}
    else:
        df = df_or_symbol

    s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze("columns")
    s = num_series(s)
    if s is None:
        return {'daily': None, 'weekly': None, 'monthly': None}

    d = s.copy()                         # 원본 일별
    w = s.resample("W-FRI").last().dropna()
    m = s.resample("ME").last().dropna()
    return {'daily': d, 'weekly': w, 'monthly': m}

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
# quote_type: 'usd_per_local' (EURUSD=X, GBPUSD=X)  -> USD terms = local_index * FX
#             'local_per_usd'(KRW=X, JPY=X, CNY=X) -> USD terms = local_index / FX
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
    return pd.to_numeric(usd, errors="coerce").dropna()

# ============== Annotations & updatemenus helpers ==============
def subtitle_annot(text, y=1.50):
    return dict(
        text=f"<b style='color:#666;font-size:13px'>{text}</b>",
        xref="paper", yref="paper", x=0.0, y=y, showarrow=False
    )

def last_value_annot(ts, unit="", ax=20, ay=-20):
    ts = num_series(ts)
    if ts is None or ts.empty:
        return None
    return dict(
        x=ts.index[-1],
        y=ts.values[-1],
        text=last_label(ts, unit),
        showarrow=True, arrowhead=2, ax=ax, ay=ay
    )

def add_freq_menu(fig: go.Figure, vis_daily, vis_weekly, vis_monthly,
                  subtitle_base, ann_daily=None, ann_weekly=None, ann_monthly=None):
    if ann_monthly is None:
        ann_monthly = fig.layout.annotations
    if ann_weekly is None:
        ann_weekly = fig.layout.annotations
    if ann_daily is None:
        ann_daily = fig.layout.annotations

    buttons = [
        dict(label="Monthly", method="update",
             args=[{"visible": vis_monthly}, {"annotations": ann_monthly}]),
        dict(label="Weekly", method="update",
             args=[{"visible": vis_weekly}, {"annotations": ann_weekly}]),
        dict(label="Daily", method="update",
             args=[{"visible": vis_daily}, {"annotations": ann_daily}]),
    ]
    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="right",
            x=0.01, xanchor="left",
            y=1.33, yanchor="top",
            bgcolor="white", bordercolor="#ddd", borderwidth=1,
            pad={"t": 2, "b": 2, "l": 2, "r": 2}
        )]
    )

# ============== Figure builders (no densify, pure lines) ==============
RANGE_BTNS = [
    dict(count=1, label="1M", step="month", stepmode="backward"),
    dict(count=3, label="3M", step="month", stepmode="backward"),
    dict(count=1, label="1Y", step="year", stepmode="backward"),
    dict(step="all"),
]

def _x_range_to_today(series_list):
    """주어진 시리즈들 중 최소 시작일 ~ 오늘(자정)까지로 범위 설정."""
    today_dt = pd.Timestamp.today().normalize()
    candidates = [s for s in series_list if s is not None and len(s)]
    min_dt = min((s.index.min() for s in candidates), default=today_dt - pd.Timedelta(days=365))
    return [min_dt, today_dt]

def fig_line_multi(series_map, name, yaxis_title, unit=None, subtitle_base=""):
    d = num_series(series_map.get("daily"))
    w = num_series(series_map.get("weekly"))
    m = num_series(series_map.get("monthly"))

    x_range = _x_range_to_today([d, w, m])

    fig = go.Figure()
    common_trace = dict(mode="lines", connectgaps=True)
    fig.add_trace(go.Scatter(x=d.index if d is not None else [],
                             y=d.values if d is not None else [],
                             name=name, visible=False, **common_trace))
    fig.add_trace(go.Scatter(x=w.index if w is not None else [],
                             y=w.values if w is not None else [],
                             name=name, visible=False, **common_trace))
    fig.add_trace(go.Scatter(x=m.index if m is not None else [],
                             y=m.values if m is not None else [],
                             name=name, visible=True,  **common_trace))

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=30, t=200, b=45),
        yaxis_title=yaxis_title,
        title=dict(text=name, x=0.01, xanchor="left", font=dict(size=18, color="#222")),
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            range=x_range,    # 초기엔 오늘까지
            rangeselector=dict(
                buttons=RANGE_BTNS,
                x=0.99, xanchor="right",
                y=1.33, yanchor="top"
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        showlegend=False
    )
    fig.update_yaxes(type="linear")

    ann_m = [subtitle_annot(f"{subtitle_base} — <u>Monthly</u>"), last_value_annot(m, unit)]
    ann_w = [subtitle_annot(f"{subtitle_base} — <u>Weekly</u>"),  last_value_annot(w, unit)]
    ann_d = [subtitle_annot(f"{subtitle_base} — <u>Daily</u>"),   last_value_annot(d, unit)]
    fig.update_layout(annotations=[a for a in ann_m if a is not None])

    vis_daily   = [True,  False, False]
    vis_weekly  = [False, True,  False]
    vis_monthly = [False, False, True ]
    add_freq_menu(fig, vis_daily, vis_weekly, vis_monthly, subtitle_base,
                  ann_daily=[a for a in ann_d if a is not None],
                  ann_weekly=[a for a in ann_w if a is not None],
                  ann_monthly=[a for a in ann_m if a is not None])
    return fig

def fig_two_lines_multi(series_map1, series_map2, name1, name2, yaxis_title, unit=None, subtitle_base="",
                        legend_pos="top-right"):
    d1 = num_series(series_map1.get("daily"));  d2 = num_series(series_map2.get("daily"))
    w1 = num_series(series_map1.get("weekly")); w2 = num_series(series_map2.get("weekly"))
    m1 = num_series(series_map1.get("monthly"));m2 = num_series(series_map2.get("monthly"))

    def align_pair(a, b):
        if a is None or b is None: return pd.Series([], dtype=float), pd.Series([], dtype=float)
        idx = a.index.intersection(b.index)
        return a.loc[idx], b.loc[idx]

    d1, d2 = align_pair(d1, d2)
    w1, w2 = align_pair(w1, w2)
    m1, m2 = align_pair(m1, m2)

    x_range = _x_range_to_today([d1, d2, w1, w2, m1, m2])

    fig = go.Figure()
    common_trace = dict(mode="lines", connectgaps=True)
    fig.add_trace(go.Scatter(x=d1.index, y=d1.values, name=name1, visible=False, **common_trace))
    fig.add_trace(go.Scatter(x=d2.index, y=d2.values, name=name2, visible=False, **common_trace))
    fig.add_trace(go.Scatter(x=w1.index, y=w1.values, name=name1, visible=False, **common_trace))
    fig.add_trace(go.Scatter(x=w2.index, y=w2.values, name=name2, visible=False, **common_trace))
    fig.add_trace(go.Scatter(x=m1.index, y=m1.values, name=name1, visible=True,  **common_trace))
    fig.add_trace(go.Scatter(x=m2.index, y=m2.values, name=name2, visible=True,  **common_trace))

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=30, t=200, b=45),
        yaxis_title=yaxis_title,
        title=dict(text=f"{name1} & {name2}", x=0.01, xanchor="left", font=dict(size=18, color="#222")),
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            range=x_range,    # 초기엔 오늘까지
            rangeselector=dict(
                buttons=RANGE_BTNS,
                x=0.99, xanchor="right",
                y=1.33, yanchor="top"
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        legend=dict(
            x=0.99, xanchor="right",
            y=0.98, yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.1)", borderwidth=1
        )
    )
    fig.update_yaxes(type="linear")

    ann_m = [subtitle_annot(f"{subtitle_base} — <u>Monthly</u>"),
             last_value_annot(m1, unit), last_value_annot(m2, unit)]
    ann_w = [subtitle_annot(f"{subtitle_base} — <u>Weekly</u>"),
             last_value_annot(w1, unit), last_value_annot(w2, unit)]
    ann_d = [subtitle_annot(f"{subtitle_base} — <u>Daily</u>"),
             last_value_annot(d1, unit), last_value_annot(d2, unit)]
    fig.update_layout(annotations=[a for a in ann_m if a is not None])

    vis_mo = [False, False, False, False, True,  True ]
    vis_we = [False, False, True,  True,  False, False]
    vis_da = [True,  True,  False, False, False, False]
    add_freq_menu(fig, vis_da, vis_we, vis_mo, subtitle_base,
                  ann_daily=[a for a in ann_d if a is not None],
                  ann_weekly=[a for a in ann_w if a is not None],
                  ann_monthly=[a for a in ann_m if a is not None])
    return fig

def fig_bar_multi(values_daily: pd.Series|None, values_weekly: pd.Series|None, values_monthly: pd.Series|None,
                  title, yaxis_title, subtitle_base=""):
    def clean(s):
        if s is None:
            return pd.Series(dtype=float)
        s = pd.Series(s)
        s = pd.to_numeric(s.dropna(), errors="coerce").dropna()
        return s

    d = clean(values_daily)
    w = clean(values_weekly)
    m = clean(values_monthly)

    x_all = pd.Index(sorted(set(d.index.astype(str)).union(w.index.astype(str)).union(m.index.astype(str))))
    def y_for(xi, s):
        if len(s) == 0:
            return [np.nan] * len(xi)
        s2 = pd.Series(s.values, index=s.index.astype(str))
        out = []
        for x in xi:
            val = s2.get(x, np.nan)
            if isinstance(val, (pd.Series, list, np.ndarray)):
                val = val[0] if len(val) else np.nan
            try: out.append(float(val))
            except: out.append(np.nan)
        return out

    y_d = y_for(x_all, d); y_w = y_for(x_all, w); y_m = y_for(x_all, m)
    txt_d = [f"{v:,.2f}" if np.isfinite(v) else "" for v in y_d]
    txt_w = [f"{v:,.2f}" if np.isfinite(v) else "" for v in y_w]
    txt_m = [f"{v:,.2f}" if np.isfinite(v) else "" for v in y_m]

    fig = go.Figure()
    fig.add_bar(x=x_all, y=y_d, name="Daily",  visible=False, text=txt_d, textposition="auto")
    fig.add_bar(x=x_all, y=y_w, name="Weekly", visible=False, text=txt_w, textposition="auto")
    fig.add_bar(x=x_all, y=y_m, name="Monthly",visible=True,  text=txt_m, textposition="auto")

    fig.update_layout(
        height=440, margin=dict(l=60, r=30, t=160, b=70),
        yaxis_title=yaxis_title, template="plotly_white",
        xaxis=dict(tickangle=15)
    )
    fig.update_yaxes(type="linear")

    fig.add_annotation(text=f"<b style='color:#666;font-size:13px'>{subtitle_base} — <u>Monthly</u></b>",
                       xref="paper", yref="paper", x=0.0, y=1.50, showarrow=False)

    vis_mo = [False, False, True]
    vis_we = [False, True,  False]
    vis_da = [True,  False, False]
    add_freq_menu(fig, vis_da, vis_we, vis_mo, subtitle_base)
    return fig

# ============== Main ==============
def main():
    # Macro core
    dgs10 = fred_series_all_freq("DGS10",  start=start_date)
    dgs2  = fred_series_all_freq("DGS2",   start=start_date)
    tips10= fred_series_all_freq("DFII10", start=start_date)
    walcl = fred_series_all_freq("WALCL",  start=start_date)   # mn USD
    hyoas = fred_series_all_freq("BAMLH0A0HYM2", start=start_date)

    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
    usdk_df, _       = yf_series_multi(["KRW=X"])
    vix_df, _        = yf_series_multi(["^VIX"])

    dxy  = market_series_all_freq(dxy_df) if dxy_df is not None else {'daily':None,'weekly':None,'monthly':None}
    usdk = market_series_all_freq(usdk_df) if usdk_df is not None else {'daily':None,'weekly':None,'monthly':None}
    vix  = market_series_all_freq(vix_df) if vix_df is not None else {'daily':None,'weekly':None,'monthly':None}

    # Derived per freq
    def map_apply(m1, m2, fn):
        return {k: (fn(m1[k], m2[k]) if (m1.get(k) is not None and m2.get(k) is not None) else None)
                for k in ["daily","weekly","monthly"]}
    be    = map_apply(dgs10, tips10, lambda a,b: (a-b)*100.0)
    curve = map_apply(dgs10, dgs2,   lambda a,b: (a-b)*100.0)
    walcl_trn = {k: (walcl[k]/1_000_000.0 if walcl.get(k) is not None else None)
                 for k in ["daily","weekly","monthly"]}

    # Δ panels
    DELTA_WINDOWS = {"daily": (63,252), "weekly": (13,52), "monthly": (3,12)}
    def deltas_for(s_map, unit_scale=1.0):
        out = {}
        for f in ["daily","weekly","monthly"]:
            s = s_map.get(f)
            if s is None:
                out[f] = None
                continue
            n3, n12 = DELTA_WINDOWS[f]
            d3 = delta_n(s, n3); d12 = delta_n(s, n12)
            if d3 is None and d12 is None:
                out[f] = None
            else:
                out[f] = pd.Series({f"Δ3M": (d3*unit_scale if d3 is not None else np.nan),
                                    f"Δ1Y": (d12*unit_scale if d12 is not None else np.nan)})
        return out

    real_delta  = deltas_for(tips10, unit_scale=100.0)
    curve_delta = deltas_for(curve,  unit_scale=1.0)
    dxy_delta   = deltas_for(dxy)
    usdk_delta  = deltas_for(usdk)
    liq_delta   = deltas_for(walcl_trn)

    def bar_series_for(freq):
        parts = []
        if real_delta.get(freq)  is not None:
            s = real_delta[freq].rename_axis("Δ").rename("Real")
            s.index = [f"Real — {idx}" for idx in s.index]
            parts.append(s)
        if curve_delta.get(freq) is not None:
            s = curve_delta[freq].rename_axis("Δ").rename("Curve")
            s.index = [f"Curve — {idx}" for idx in s.index]
            parts.append(s)
        if dxy_delta.get(freq)   is not None:
            s = dxy_delta[freq].rename_axis("Δ").rename("DXY")
            s.index = [f"DXY — {idx}" for idx in s.index]
            parts.append(s)
        if usdk_delta.get(freq)  is not None:
            s = usdk_delta[freq].rename_axis("Δ").rename("USDKRW")
            s.index = [f"USDKRW — {idx}" for idx in s.index]
            parts.append(s)
        if liq_delta.get(freq)   is not None:
            s = liq_delta[freq].rename_axis("Δ").rename("Liquidity (Trn)")
            s.index = [f"Liquidity (Trn) — {idx}" for idx in s.index]
            parts.append(s)
        if parts:
            ser = pd.concat(parts)
            ser.index = ser.index.astype(str)
            return ser
        return None

    bar_daily   = bar_series_for("daily")
    bar_weekly  = bar_series_for("weekly")
    bar_monthly = bar_series_for("monthly")

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

    local_maps = {}
    for nm, tkr in indices.items():
        df, _ = yf_series_multi([tkr])
        local_maps[nm] = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}

    fx_maps = {}
    for name, meta in FX_MAP.items():
        if meta["fx"]:
            df_fx, _ = yf_series_multi([meta["fx"]])
            fx_maps[name] = market_series_all_freq(df_fx) if df_fx is not None else {'daily':None,'weekly':None,'monthly':None}

    usd_maps = {}
    for name, s_map in local_maps.items():
        meta = FX_MAP.get(name, {"fx": None, "quote_type": None})
        fx_map = fx_maps.get(name)
        out = {}
        for f in ["daily","weekly","monthly"]:
            s_loc = s_map.get(f)
            fx_s  = fx_map.get(f) if (fx_map and meta["fx"]) else None
            out[f] = to_usd_terms(s_loc, fx_s, meta["quote_type"]) if (s_loc is not None) else None
            if out[f] is None and meta["fx"] is None:
                out[f] = s_loc
        usd_maps[name + " [USD]"] = out

    comm_maps = {}
    for nm, tkr in commodities.items():
        df, _ = yf_series_multi([tkr])
        comm_maps[nm] = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}

    # ---------- Build Figures ----------
    figs_overview = []
    figs_overview.append((
        "Rates — US10Y & US2Y",
        fig_two_lines_multi(dgs10, dgs2, "US10Y", "US2Y", "Yield (%)", unit="pct",
                            subtitle_base="장·단기 금리 비교 — 경기/정책 기대")
    ))
    figs_overview.append((
        "10Y TIPS Real Yield",
        fig_line_multi(tips10, "10Y TIPS Real Yield", "Yield (%)", unit="pct",
                       subtitle_base="실질금리 — 금융여건/할인율 지표")
    ))
    figs_overview.append((
        "10Y Breakeven",
        fig_line_multi(be, "10Y Breakeven", "bps", unit="bps",
                       subtitle_base="기대 인플레이션 — 향후 10년 평균 물가 기대")
    ))
    figs_overview.append((
        "10Y–2Y Curve",
        fig_line_multi(curve, "10Y–2Y Curve", "bps", unit="bps",
                       subtitle_base="수익률 곡선 — 장단기 금리차")
    ))
    dxy_title = "DXY (Dollar Index)" + (f" — used {dxy_used}" if dxy_used else "")
    figs_overview.append((
        "DXY (Dollar Index)",
        fig_line_multi(dxy, dxy_title, "Index", unit="idx",
                       subtitle_base="달러 강세/약세 — 글로벌 자금 흐름")
    ))
    figs_overview.append((
        "USDKRW",
        fig_line_multi(usdk, "USDKRW", "KRW per USD", unit="fx",
                       subtitle_base="원/달러 환율 — 자금 유출입/리스크 온오프")
    ))
    figs_overview.append((
        "VIX",
        fig_line_multi(vix, "VIX", "Index", unit="idx",
                       subtitle_base="S&P500 변동성(공포지수)")
    ))
    figs_overview.append((
        "HY OAS",
        fig_line_multi(hyoas, "HY OAS", "bps", unit="bps",
                       subtitle_base="하이일드 스프레드 — 신용 리스크")
    ))
    figs_overview.append((
        "Fed Balance Sheet (Trn USD)",
        fig_line_multi(walcl_trn, "Fed Balance Sheet (Trn USD)", "Trn USD", unit="idx",
                       subtitle_base="연준 대차대조표 규모")
    ))

    bar_fig = fig_bar_multi(bar_daily, bar_weekly, bar_monthly,
                            "Latest Δ (per freq)", "Δ (unit per label)",
                            subtitle_base="변화율 패널(3M/1Y 근사치)")

    figs_local = [(nm, fig_line_multi(series_map, nm, "Index", unit="idx",
                                      subtitle_base=f"{nm} — Local currency"))
                  for nm, series_map in local_maps.items()]
    figs_usd   = [(nm, fig_line_multi(series_map, nm, "Index (USD terms)", unit="idx",
                                      subtitle_base=f"{nm} — USD terms (FX-adjusted)"))
                  for nm, series_map in usd_maps.items()]
    figs_comm  = [(nm, fig_line_multi(series_map, nm, "Price (USD)", unit="idx",
                                      subtitle_base=f"{nm} — 원자재 가격"))
                  for nm, series_map in comm_maps.items()]

    # Crypto
    crypto_majors = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD"]
    crypto_alts   = ["ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","LTC-USD"]
    def crypto_map(symbols):
        out=[]
        for t in symbols:
            df, _ = yf_series_multi([t])
            m = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}
            ttl = t.replace("-USD","")
            out.append((ttl, fig_line_multi(m, ttl, "Price (USD)", unit="idx",
                                            subtitle_base=f"{ttl} — Crypto")))
        return out
    figs_cmaj = crypto_map(crypto_majors)
    figs_calt = crypto_map(crypto_alts)

    # ===== HTML assemble with TABS + responsive + safe clamp-to-today =====
    from plotly.offline import plot as plot_offline
    def plot_div(fig, include_js=False):
        return plot_offline(fig, include_plotlyjs=('cdn' if include_js else False),
                            output_type='div', config={'responsive': True})

    def section_html(section_id, title, figs):
        parts = [f"<div class='tab-section' id='{section_id}' style='display:none'>"]
        parts.append(f"<div class='card'><h2>{title}</h2><div class='grid'>")
        for _, f in figs:
            parts.append(plot_div(f, include_js=False))
        parts.append("</div></div></div>")
        return "\n".join(parts)

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Macro Dashboard — Per-Chart Frequency</title>",
        "<style>",
        "body{font-family:-apple-system,Roboto,Segoe UI,Helvetica,Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 16px}",
        ".card{box-shadow:0 8px 28px rgba(0,0,0,.08);border-radius:14px;padding:14px;margin:14px 0}",
        "h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 10px}p{line-height:1.55;color:#333}",
        ".grid{display:grid;grid-template-columns:1fr;gap:12px}@media(min-width:1100px){.grid{grid-template-columns:1fr 1fr}}",
        ".grid > div { min-width: 0; }",
        ".muted{color:#666;font-size:13px}",
        ".tabs{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 14px}",
        ".tabbtn{padding:8px 12px;border:1px solid #ddd;border-radius:10px;background:#f8f9fb;cursor:pointer}",
        ".tabbtn.active{background:#2b6cb0;color:#fff;border-color:#2b6cb0}",
        "</style>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<script>",
        "function resizeSection(id){",
        "  const sec = document.getElementById(id);",
        "  if(!sec) return;",
        "  const plots = sec.querySelectorAll('.plotly-graph-div');",
        "  plots.forEach(gd => { try { Plotly.Plots.resize(gd); Plotly.relayout(gd, {autosize: true}); } catch(e){} });",
        "}",
        "// === 안전한 오늘 클램프 (무한루프 방지) ===",
        "function clampAxisToToday(gd){",
        "  if (gd.__clampAttached) return;",
        "  gd.__clampAttached = true;",
        "  gd.__isClamping = false;",
        "  const today = new Date(); today.setHours(0,0,0,0);",
        "  const EPS = 60000; // 1분 허용오차",
        "  function needClamp(xa){",
        "    if (!xa || !xa.range || xa.range.length !== 2) return false;",
        "    const r1 = new Date(xa.range[1]).getTime();",
        "    return (r1 - today.getTime()) > EPS;",
        "  }",
        "  function doClamp(){",
        "    try{",
        "      const xa = gd._fullLayout && gd._fullLayout.xaxis;",
        "      if (!xa || !xa.range) return;",
        "      if (!needClamp(xa)) return;",
        "      const r0 = xa.range[0];",
        "      gd.__isClamping = true;",
        "      Plotly.relayout(gd, {'xaxis.autorange': false, 'xaxis.range': [r0, today]})",
        "        .then(()=>{ gd.__isClamping = false; })",
        "        .catch(()=>{ gd.__isClamping = false; });",
        "    }catch(e){ gd.__isClamping = false; }",
        "  }",
        "  setTimeout(doClamp, 0);",
        "  gd.on('plotly_relayout', ev => {",
        "    if (gd.__isClamping) return;",
        "    requestAnimationFrame(doClamp);",
        "  });",
        "}",
        "function clampAllPlotsToToday(){",
        "  document.querySelectorAll('.plotly-graph-div').forEach(gd => clampAxisToToday(gd));",
        "}",
        "function showTab(id){",
        "  const sections=document.querySelectorAll('.tab-section');",
        "  sections.forEach(s=>s.style.display='none');",
        "  const tabbtns=document.querySelectorAll('.tabbtn');",
        "  tabbtns.forEach(b=>b.classList.remove('active'));",
        "  const sec=document.getElementById(id); if(sec){ sec.style.display='block'; }",
        "  const btn=document.querySelector(`[data-target='${id}']`); if(btn) btn.classList.add('active');",
        "  setTimeout(()=>{ resizeSection(id); clampAllPlotsToToday(); }, 30);",
        "  window.scrollTo({top:0,behavior:'smooth'});",
        "}",
        "window.addEventListener('resize', ()=>{",
        "  const visible = Array.from(document.querySelectorAll('.tab-section')).find(el => el.style.display !== 'none');",
        "  if(visible){ resizeSection(visible.id); clampAllPlotsToToday(); }",
        "});",
        "document.addEventListener('DOMContentLoaded',()=>{ showTab('tab-overview'); clampAllPlotsToToday(); });",
        "</script>",
        "</head><body>",
        f"<h1>Macro Dashboard — Per-Chart Frequency <span class='muted'>(Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')})</span></h1>",
        "<div class='tabs'>",
        "<button class='tabbtn active' data-target='tab-overview' onclick=\"showTab('tab-overview')\">Overview</button>",
        "<button class='tabbtn' data-target='tab-global-local' onclick=\"showTab('tab-global-local')\">Global Equities (Local)</button>",
        "<button class='tabbtn' data-target='tab-global-usd' onclick=\"showTab('tab-global-usd')\">Global Equities (USD)</button>",
        "<button class='tabbtn' data-target='tab-commodities' onclick=\"showTab('tab-commodities')\">Commodities</button>",
        "<button class='tabbtn' data-target='tab-crypto-majors' onclick=\"showTab('tab-crypto-majors')\">Crypto (Majors)</button>",
        "<button class='tabbtn' data-target='tab-crypto-alts' onclick=\"showTab('tab-crypto-alts')\">Crypto (Alts)</button>",
        "</div>",
    ]

    # Overview (inject Plotly.js once)
    from plotly.offline import plot as plot_offline
    def plot_div_first(fig):
        return plot_offline(fig, include_plotlyjs='cdn', output_type='div', config={'responsive': True})

    ov_html = ["<div class='tab-section' id='tab-overview' style='display:none'>"]
    ov_html.append("<div class='card'><h2>Rates</h2><div class='grid'>")
    first = True
    for t, f in figs_overview:
        if t in {"Rates — US10Y & US2Y", "10Y TIPS Real Yield", "10Y Breakeven", "10Y–2Y Curve"}:
            ov_html.append(plot_div_first(f) if first else plot_div(f))
            first = False
    ov_html.append("</div></div>")

    ov_html.append("<div class='card'><h2>Dollar / FX & Credit & Liquidity</h2><div class='grid'>")
    for t, f in figs_overview:
        if t not in {"Rates — US10Y & US2Y", "10Y TIPS Real Yield", "10Y Breakeven", "10Y–2Y Curve"}:
            ov_html.append(plot_div(f))
    ov_html.append("</div></div>")

    ov_html.append("<div class='card'><h2>Latest Δ — per selected frequency</h2>")
    ov_html.append(plot_div(bar_fig))
    ov_html.append("</div>")

    ov_html.append("</div>")
    html_parts.append("\n".join(ov_html))

    # Other tabs
    html_parts.append(section_html("tab-global-local", "Global Equities — Local", figs_local))
    html_parts.append(section_html("tab-global-usd",  "Global Equities — USD Terms", figs_usd))
    html_parts.append(section_html("tab-commodities", "Commodities", figs_comm))
    html_parts.append(section_html("tab-crypto-majors", "Crypto (Majors)", figs_cmaj))
    html_parts.append(section_html("tab-crypto-alts",   "Crypto (Alts)",   figs_calt))

    html_parts.append("</body></html>")
    HTML_PATH.write_text("\n".join(html_parts), encoding="utf-8")
    print("✅ Open:", HTML_PATH)

if __name__ == "__main__":
    main()
