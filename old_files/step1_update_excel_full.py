# step1_update_excel_full.py
# -----------------------------------------------------------
# Update Macro_Dashboard_Data.xlsx with 3y monthly history + signals
# + Global Equities (10) & Commodities (5) EOM series to history
# + USD terms (FX-adjusted) columns for global indices
# - Snapshot upsert, History, Signals, Meta
# - Dates as 'YYYY-MM-DD' strings (avoid Excel serials)
# -----------------------------------------------------------
import os
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
from pandas import ExcelWriter

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing in .env")

OUTPUT_XLSX = os.getenv("OUTPUT_XLSX", "Macro_Dashboard_Data.xlsx")
RECORD_TZ = os.getenv("RECORD_TZ", "local").lower()

today = dt.datetime.utcnow().date() if RECORD_TZ == "utc" else dt.date.today()
start_date = (today - dt.timedelta(days=3 * 365)).strftime("%Y-%m-%d")

fred = Fred(api_key=FRED_KEY)

def fred_series(code, start=None) -> pd.Series:
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

def last_val(s):
    if s is None:
        return None
    s2 = s.dropna()
    if len(s2) == 0:
        return None
    v = s2.iloc[-1]
    return v.item() if hasattr(v, "item") else float(v)

def delta_months_series(s, n):
    if s is None:
        return None
    s2 = s.dropna()
    if len(s2) < (n + 1):
        return pd.Series(dtype=float, index=s2.index)
    return (s2 - s2.shift(n)).dropna()

def curve_regime_series(curve_bp, ten_yield):
    if curve_bp is None or ten_yield is None:
        return pd.Series(dtype=object)
    d_curve = delta_months_series(curve_bp, 3)
    d_10y = delta_months_series(ten_yield, 3)
    idx = d_curve.index.intersection(d_10y.index)
    out = pd.Series(index=idx, dtype=object)
    out[(d_curve.loc[idx] > 0) & (d_10y.loc[idx] > 0)] = "Bear steepening"
    out[(d_curve.loc[idx] > 0) & (d_10y.loc[idx] < 0)] = "Bull steepening"
    return out

def with_date_col_str(df, date_col_name="Date"):
    if df is None or df.empty:
        return df
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        date_str = out.index.strftime("%Y-%m-%d")
    else:
        date_str = pd.to_datetime(out.index, errors="coerce").strftime("%Y-%m-%d")
    out.insert(0, date_col_name, date_str)
    out = out.reset_index(drop=True)
    return out

# ---------- FX map for USD rebasing ----------
FX_MAP = {
    "S&P 500 (^GSPC)":      {"fx": None,          "quote_type": None},
    "Euro Stoxx 50 (^STOXX50E)": {"fx": "EURUSD=X", "quote_type": "usd_per_local"},
    "Nikkei 225 (^N225)":   {"fx": "JPY=X",       "quote_type": "local_per_usd"},
    "Shanghai Composite (000001.SS)": {"fx": "CNY=X", "quote_type": "local_per_usd"},
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

# =========================
# 2) 데이터 로드 (3y)
# =========================
# Macro (FRED)
dgs10 = fred_series("DGS10", start=start_date)
dgs2  = fred_series("DGS2",  start=start_date)
tips10= fred_series("DFII10", start=start_date)
walcl = fred_series("WALCL", start=start_date)  # mn USD
hyoas = fred_series("BAMLH0A0HYM2", start=start_date)

# FX/Vol/BTC
dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
usdk_df, _       = yf_series_multi(["KRW=X"])
vix_df, _        = yf_series_multi(["^VIX"])
btc_df, btc_used = yf_series_multi(["BTC-USD", "XBT-USD", "BTCUSD=X"])

# Global Equities (10)
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
# Commodities (5) — USD quoted
commodities = {
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "Crude Oil (CL=F)": "CL=F",
    "Copper (HG=F)": "HG=F",
    "Natural Gas (NG=F)": "NG=F"
}

# =========================
# 3) 월말 시계열 & 파생
# =========================
dgs10_m = eom(dgs10)
dgs2_m  = eom(dgs2)
tips_m  = eom(tips10)
dxy_m   = eom(dxy_df)
usdk_m  = eom(usdk_df)
vix_m   = eom(vix_df)
hyoas_m = eom(hyoas)
walcl_m = eom(walcl)  # mn USD

breakeven_bp_m = (dgs10_m - tips_m) * 100.0 if (dgs10_m is not None and tips_m is not None) else None
curve_bp_m     = (dgs10_m - dgs2_m) * 100.0 if (dgs10_m is not None and dgs2_m is not None) else None
walcl_trn_m    = (walcl_m / 1_000_000.0) if walcl_m is not None else None

# MOVE proxy
move_proxy_daily_bp = None
if dgs10 is not None and len(dgs10.dropna()) >= 22:
    chg_bp = dgs10.dropna().diff().dropna() * 100.0
    roll = chg_bp.rolling(21).std().dropna() * np.sqrt(252.0)
    move_proxy_daily_bp = roll
move_proxy_m = eom(move_proxy_daily_bp) if move_proxy_daily_bp is not None else None

# Global EOM (local)
local_eom = {}
for name, tkr in indices.items():
    df, _ = yf_series_multi([tkr])
    if df is not None and not df.empty:
        s = eom(df)
        if s is not None and not s.empty:
            local_eom[name] = s

# FX EOM for USD terms
fx_eom = {}
for name, meta in FX_MAP.items():
    fx_tkr = meta["fx"]
    if fx_tkr:
        df_fx, _ = yf_series_multi([fx_tkr])
        if df_fx is not None and not df_fx.empty:
            fx_eom[name] = eom(df_fx)

# USD-terms indices
usd_eom = {}
for name, s_local in local_eom.items():
    meta = FX_MAP.get(name, {"fx": None, "quote_type": None})
    fx_s = fx_eom.get(name, None) if meta["fx"] else None
    usd_series = to_usd_terms(s_local, fx_s, meta["quote_type"])
    if usd_series is None and meta["fx"] is None:
        usd_series = s_local.copy()  # already USD (S&P 500)
    if usd_series is not None and not usd_series.empty:
        usd_eom[name + " [USD]"] = usd_series

# Commodities (USD quoted)
comm_eom = {}
for name, tkr in commodities.items():
    df, _ = yf_series_multi([tkr])
    if df is not None and not df.empty:
        s = eom(df)
        if s is not None and not s.empty:
            comm_eom[name] = s

# =========================
# 4) 시트 데이터프레임
# =========================
hist_cols = {}
# macro core
if dgs10_m is not None:        hist_cols["US10Y Nominal Yield (%)"]      = dgs10_m
if dgs2_m  is not None:        hist_cols["US2Y Nominal Yield (%)"]       = dgs2_m
if tips_m  is not None:        hist_cols["US10Y TIPS Real Yield (%)"]    = tips_m
if breakeven_bp_m is not None: hist_cols["10Y Breakeven (bps)"]          = breakeven_bp_m
if curve_bp_m is not None:     hist_cols["10Y–2Y Curve (bps)"]           = curve_bp_m
if dxy_m   is not None:        hist_cols["DXY (Dollar Index)"]           = dxy_m
if usdk_m  is not None:        hist_cols["USDKRW"]                        = usdk_m
if vix_m   is not None:        hist_cols["VIX (Equity Vol)"]             = vix_m
if hyoas_m is not None:        hist_cols["HY OAS (bps)"]                 = hyoas_m
if walcl_trn_m is not None:    hist_cols["Fed Balance Sheet (Trn USD)"]  = walcl_trn_m
if btc_df is not None:
    btc_m = eom(btc_df)
    if btc_m is not None:
        hist_cols["BTC Price (USD)"] = btc_m
if move_proxy_m is not None:   hist_cols["MOVE Proxy (bps)"]             = move_proxy_m

# global local + USD
for name, s in local_eom.items():
    hist_cols[name] = s
for name, s in usd_eom.items():
    hist_cols[name] = s
# commodities
for name, s in comm_eom.items():
    hist_cols[name] = s

inputs_hist = pd.DataFrame(hist_cols)
inputs_hist.index.name = "Date"
inputs_hist_out = with_date_col_str(inputs_hist, "Date")

# Signals (macro only)
def deltablock(s, mul=1.0):
    if s is None:
        return pd.DataFrame()
    d3  = delta_months_series(s, 3)
    d12 = delta_months_series(s, 12)
    out = pd.DataFrame(index=s.index)
    if d3 is not None:
        out["Δ3M"] = d3 * mul
    if d12 is not None:
        out["Δ1Y"] = d12 * mul
    return out

sig_frames = []
if tips_m is not None:
    sig_frames.append(deltablock(tips_m,   mul=100.0).rename(columns={"Δ3M": "Real Yield (10Y TIPS, bp) Δ3M",
                                                                      "Δ1Y": "Real Yield (10Y TIPS, bp) Δ1Y"}))
if curve_bp_m is not None:
    sig_frames.append(deltablock(curve_bp_m, mul=1.0).rename(columns={"Δ3M": "Curve (10Y–2Y, bp) Δ3M",
                                                                      "Δ1Y": "Curve (10Y–2Y, bp) Δ1Y"}))
if dxy_m is not None:
    sig_frames.append(deltablock(dxy_m,    mul=1.0).rename(columns={"Δ3M": "DXY Δ3M", "Δ1Y": "DXY Δ1Y"}))
if usdk_m is not None:
    sig_frames.append(deltablock(usdk_m,   mul=1.0).rename(columns={"Δ3M": "USDKRW Δ3M", "Δ1Y": "USDKRW Δ1Y"}))
if walcl_trn_m is not None:
    sig_frames.append(deltablock(walcl_trn_m, mul=1.0).rename(columns={"Δ3M": "Liquidity (Trn USD) Δ3M",
                                                                        "Δ1Y": "Liquidity (Trn USD) Δ1Y"}))

signals_hist = None
if sig_frames:
    signals_hist = pd.concat(sig_frames, axis=1)
    regime = curve_regime_series(curve_bp_m, dgs10_m)
    if regime is not None and not regime.empty:
        signals_hist = signals_hist.join(regime.rename("Curve Regime"), how="outer")

    def risk_scores(row):
        roff = 0; ron = 0
        rv = row.get("Real Yield (10Y TIPS, bp) Δ3M")
        if pd.notna(rv):
            if rv > 25:  roff += 1
            if rv < -25: ron += 1
        dx = row.get("DXY Δ3M")
        if pd.notna(dx):
            if dx > 2:   roff += 1
            if dx < -2:  ron  += 1
        cr = row.get("Curve Regime")
        if isinstance(cr, str):
            if cr == "Bear steepening": roff += 1
            elif cr == "Bull steepening": ron += 1
        liq = row.get("Liquidity (Trn USD) Δ3M")
        if pd.notna(liq) and liq > 0.1:
            ron += 1
        return pd.Series({"Risk-Off Score": roff, "Risk-On Score": ron})
    scores = signals_hist.apply(risk_scores, axis=1)
    signals_hist = pd.concat([signals_hist, scores], axis=1)

signals_hist_out = with_date_col_str(signals_hist, "Date") if (signals_hist is not None and not signals_hist.empty) else None

# Snapshot 'Inputs'
inputs_snapshot = {
    "Date (YYYY-MM-DD)": today.strftime("%Y-%m-%d"),
    "US10Y Nominal Yield (%)": last_val(dgs10),
    "US2Y Nominal Yield (%)": last_val(dgs2),
    "US10Y TIPS Real Yield (%)": last_val(tips10),
    "DXY (Dollar Index)": last_val(dxy_m),
    "USDKRW": last_val(usdk_m),
    "US CPI YoY (%)": None,
    "Fed Balance Sheet (Trn USD)": last_val(walcl_trn_m),
    "MOVE (Bond Vol)": last_val(move_proxy_m),
    "VIX (Equity Vol)": last_val(vix_m),
    "HY OAS (bps)": last_val(hyoas_m),
    "BTC Price (USD)": last_val(eom(btc_df)) if btc_df is not None else None,
}
# Add latest of global (local + USD) + commodities
for name, s in local_eom.items():
    inputs_snapshot[name] = last_val(s)
for name, s in usd_eom.items():
    inputs_snapshot[name] = last_val(s)
for name, s in comm_eom.items():
    inputs_snapshot[name] = last_val(s)

inputs_df = pd.DataFrame([inputs_snapshot])
inputs_df["Date (YYYY-MM-DD)"] = inputs_df["Date (YYYY-MM-DD)"].astype(str)

# Meta
meta_df = pd.DataFrame([
    {"Key": "Generated (local time)", "Value": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    {"Key": "Start Date", "Value": start_date},
    {"Key": "DXY ticker used", "Value": dxy_used},
    {"Key": "BTC ticker used", "Value": btc_used},
    {"Key": "Global indices", "Value": ", ".join(indices.keys())},
    {"Key": "USD terms map", "Value": "; ".join([f"{k}->{v['fx'] or 'USD'}({v['quote_type']})" for k,v in FX_MAP.items()])},
    {"Key": "Commodities", "Value": ", ".join(commodities.keys())},
])

# =========================
# 5) 엑셀 기록
# =========================
xlsx = Path(OUTPUT_XLSX)
mode = "a" if xlsx.exists() else "w"

if mode == "a":
    with ExcelWriter(xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        # Inputs 업서트
        try:
            prev = pd.read_excel(xlsx, sheet_name="Inputs")
        except Exception:
            prev = pd.DataFrame(columns=list(inputs_df.columns))
        if "Date (YYYY-MM-DD)" in prev.columns:
            mask = prev["Date (YYYY-MM-DD)"].astype(str) == inputs_df.iloc[0]["Date (YYYY-MM-DD)"]
            if mask.any():
                prev.loc[mask, :] = inputs_df.iloc[0].values
                prev.to_excel(w, sheet_name="Inputs", index=False)
            else:
                pd.concat([prev, inputs_df], ignore_index=True).to_excel(w, sheet_name="Inputs", index=False)
        else:
            inputs_df.to_excel(w, sheet_name="Inputs", index=False)

        inputs_hist_out.to_excel(w, sheet_name="Inputs_History_Monthly", index=False)
        if signals_hist_out is not None and not signals_hist_out.empty:
            signals_hist_out.to_excel(w, sheet_name="Signals_History_Monthly", index=False)
        meta_df.to_excel(w, sheet_name="Meta", index=False)
else:
    with ExcelWriter(xlsx, engine="openpyxl", mode="w") as w:
        inputs_df.to_excel(w, sheet_name="Inputs", index=False)
        inputs_hist_out.to_excel(w, sheet_name="Inputs_History_Monthly", index=False)
        if signals_hist_out is not None and not signals_hist_out.empty:
            signals_hist_out.to_excel(w, sheet_name="Signals_History_Monthly", index=False)
        meta_df.to_excel(w, sheet_name="Meta", index=False)

print(f"✅ Updated {OUTPUT_XLSX}")
print(" - Sheets: Inputs, Inputs_History_Monthly, Signals_History_Monthly, Meta")
