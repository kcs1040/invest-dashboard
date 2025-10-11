# step1_export_excel_full.py
# -----------------------------------------------------------
# Export full dataset (3y) to Excel for the Plotly dashboard:
# - Macro core (FRED + yfinance)
# - Derived series (breakeven, 10Y-2Y curve, Fed liquidity in Trn USD)
# - Global equities (Local & USD terms), Commodities
# - Crypto (Majors & Alts)
# - Δ panel (3M/1Y approximations) per frequency
# - Sheets split by frequency: Daily / Weekly / Monthly
# - Date columns saved as ISO strings to avoid Excel serials
# -----------------------------------------------------------

import os, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# ===================== Config =====================
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing. Put it in .env like: FRED_API_KEY=YOUR_KEY")

OUTDIR = Path(os.getenv("OUTDIR_XLSX", "data_pro"))
OUTDIR.mkdir(parents=True, exist_ok=True)
XLSX_PATH = OUTDIR / "Macro_Dashboard_Full.xlsx"

today = dt.date.today()
start_date = (today - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# ===================== Helpers =====================
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
    if isinstance(s, (list, tuple, np.ndarray)):
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
    Return dict {'daily','weekly','monthly'} WITHOUT filling missing days.
    weekly = W-FRI last, monthly = month-end last.
    """
    s = fred_series(code, start=start)
    if s is None or s.empty:
        return {'daily': None, 'weekly': None, 'monthly': None}
    d = s.copy()
    w = s.resample("W-FRI").last().dropna()
    m = s.resample("ME").last().dropna()
    return {'daily': d, 'weekly': w, 'monthly': m}

def yf_series_multi(candidates, period="3y", interval="1d"):
    for tkr in candidates:
        try:
            df = yf.download(
                tkr, period=period, interval=interval,
                progress=False, auto_adjust=False, group_by="column"
            )
            if df is not None and not df.empty and "Close" in df.columns:
                return df, tkr
        except Exception:
            pass
    return None, None

def market_series_all_freq(df_or_symbol):
    """yfinance df or symbol -> {'daily','weekly','monthly'} Close without any ffill."""
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
    d = s.copy()
    w = s.resample("W-FRI").last().dropna()
    m = s.resample("ME").last().dropna()
    return {'daily': d, 'weekly': w, 'monthly': m}

def to_wide_df(series_dict: dict[str, pd.Series] | None, index_name="Date") -> pd.DataFrame:
    """
    series_dict: {"Column Name": pd.Series, ...}
    -> Wide DataFrame with ISO date string column first.
    """
    if not series_dict:
        return pd.DataFrame()
    frames = []
    for name, s in series_dict.items():
        s = num_series(s)
        if s is None or s.empty:
            continue
        frames.append(s.rename(name).to_frame())
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).sort_index()
    # Move index to ISO string column to avoid Excel serials
    df = df.reset_index().rename(columns={"index": index_name})
    df[index_name] = pd.to_datetime(df[index_name]).dt.date.astype(str)
    return df

def delta_n(s, n):
    s = num_series(s)
    if s is None or len(s) < (n + 1):
        return None
    v = s.iloc[-1] - s.iloc[-(n + 1)]
    return float(v)

# ===================== USD rebasing helpers =====================
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
        return series_local.copy()
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

# ===================== Main =====================
def main():
    # ---------- Macro: FRED ----------
    dgs10 = fred_series_all_freq("DGS10",  start=start_date)   # 10Y nominal
    dgs2  = fred_series_all_freq("DGS2",   start=start_date)   # 2Y nominal
    tips10= fred_series_all_freq("DFII10", start=start_date)   # 10Y real
    walcl = fred_series_all_freq("WALCL",  start=start_date)   # Fed balance sheet (mn USD)
    hyoas = fred_series_all_freq("BAMLH0A0HYM2", start=start_date)  # HY OAS (bps)

    # ---------- Macro: Market (yfinance) ----------
    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
    usdk_df, _       = yf_series_multi(["KRW=X"])
    vix_df, _        = yf_series_multi(["^VIX"])
    btc_df, _        = yf_series_multi(["BTC-USD", "XBT-USD", "BTCUSD=X"])

    dxy  = market_series_all_freq(dxy_df) if dxy_df is not None else {'daily':None,'weekly':None,'monthly':None}
    usdk = market_series_all_freq(usdk_df) if usdk_df is not None else {'daily':None,'weekly':None,'monthly':None}
    vix  = market_series_all_freq(vix_df) if vix_df is not None else {'daily':None,'weekly':None,'monthly':None}
    btc  = market_series_all_freq(btc_df) if btc_df is not None else {'daily':None,'weekly':None,'monthly':None}

    # ---------- Derived ----------
    def map_apply(m1, m2, fn):
        return {k: (fn(m1[k], m2[k]) if (m1.get(k) is not None and m2.get(k) is not None) else None)
                for k in ["daily","weekly","monthly"]}
    breakeven = map_apply(dgs10, tips10, lambda a,b: (a - b) * 100.0)  # bps
    curve = map_apply(dgs10, dgs2,   lambda a,b: (a - b) * 100.0)      # bps
    walcl_trn = {k: (walcl[k]/1_000_000.0 if walcl.get(k) is not None else None)
                 for k in ["daily","weekly","monthly"]}                # Trn USD

    # ---------- Global equities (Local + USD) ----------
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
        "FTSE 100 (^FTSE)": "^FTSE",
    }
    # Local
    local_maps: dict[str, dict] = {}
    for nm, tkr in indices.items():
        df, _ = yf_series_multi([tkr])
        local_maps[nm] = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}
    # FX for USD rebasing
    fx_maps = {}
    for name, meta in FX_MAP.items():
        if meta["fx"]:
            df_fx, _ = yf_series_multi([meta["fx"]])
            fx_maps[name] = market_series_all_freq(df_fx) if df_fx is not None else {'daily':None,'weekly':None,'monthly':None}
    # USD-terms
    usd_maps: dict[str, dict] = {}
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

    # ---------- Commodities ----------
    commodities = {
        "Gold (GC=F)": "GC=F",
        "Silver (SI=F)": "SI=F",
        "Crude Oil (CL=F)": "CL=F",
        "Copper (HG=F)": "HG=F",
        "Natural Gas (NG=F)": "NG=F",
    }
    comm_maps = {}
    for nm, tkr in commodities.items():
        df, _ = yf_series_multi([tkr])
        comm_maps[nm] = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}

    # ---------- Crypto ----------
    crypto_majors = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD"]
    crypto_alts   = ["ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","LTC-USD"]
    def fetch_crypto(symbols):
        out = {}
        for t in symbols:
            df, _ = yf_series_multi([t])
            m = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}
            out[t.replace("-USD","")] = m
        return out
    cmaj_maps = fetch_crypto(crypto_majors)
    calt_maps = fetch_crypto(crypto_alts)

    # ---------- Δ panel (per freq) ----------
    DELTA_WINDOWS = {"daily": (63,252), "weekly": (13,52), "monthly": (3,12)}
    def deltas_for(s_map, unit_scale=1.0, label=""):
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
                out[f] = pd.Series({f"{label} Δ3M": (d3*unit_scale if d3 is not None else np.nan),
                                    f"{label} Δ1Y": (d12*unit_scale if d12 is not None else np.nan)})
        return out

    real_delta  = deltas_for(tips10, unit_scale=100.0, label="Real")
    curve_delta = deltas_for(curve,  unit_scale=1.0,  label="Curve")
    dxy_delta   = deltas_for(dxy,    label="DXY")
    usdk_delta  = deltas_for(usdk,   label="USDKRW")
    liq_delta   = deltas_for(walcl_trn, label="Liquidity (Trn)")

    # ===================== Write to Excel =====================
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:

        # ---- Macro Inputs & Derived (per freq) ----
        for freq in ["daily","weekly","monthly"]:
            macro_inputs = {
                "US10Y Nominal (%)": dgs10[freq],
                "US2Y Nominal (%)":  dgs2[freq],
                "10Y TIPS Real (%)": tips10[freq],
                "DXY":               dxy[freq],
                "USDKRW":            usdk[freq],
                "VIX":               vix[freq],
                "HY OAS (bps)":      hyoas[freq],
                "Fed Balance Sheet (Trn USD)": walcl_trn[freq],
                "BTC Price (USD)":   btc[freq],
            }
            df_inputs = to_wide_df(macro_inputs, index_name="Date")
            df_inputs.to_excel(xw, sheet_name=f"Macro_Inputs_{freq.capitalize()}", index=False)

            derived = {
                "10Y Breakeven (bps)": breakeven[freq],
                "10Y–2Y Curve (bps)":  curve[freq],
                "Fed Balance Sheet (Trn USD)": walcl_trn[freq],
            }
            df_derived = to_wide_df(derived, index_name="Date")
            df_derived.to_excel(xw, sheet_name=f"Derived_{freq.capitalize()}", index=False)

        # ---- Δ Panel (per freq) ----
        def delta_sheet(freq):
            parts = []
            for block in [real_delta, curve_delta, dxy_delta, usdk_delta, liq_delta]:
                if block.get(freq) is not None:
                    parts.append(block[freq])
            if parts:
                s = pd.concat(parts)
                df = s.rename("Value").to_frame()
                df = df.reset_index().rename(columns={"index":"Metric"})
                return df
            return pd.DataFrame(columns=["Metric","Value"])
        for freq in ["daily","weekly","monthly"]:
            df_delta = delta_sheet(freq)
            df_delta.to_excel(xw, sheet_name=f"Delta_{freq.capitalize()}", index=False)

        # ---- Global Equities (Local/USD) ----
        for freq in ["daily","weekly","monthly"]:
            # Local
            local_cols = {name: s_map[freq] for name, s_map in local_maps.items()}
            df_local = to_wide_df(local_cols, index_name="Date")
            df_local.to_excel(xw, sheet_name=f"Equities_Local_{freq.capitalize()}", index=False)
            # USD
            usd_cols = {name: s_map[freq] for name, s_map in usd_maps.items()}
            df_usd = to_wide_df(usd_cols, index_name="Date")
            df_usd.to_excel(xw, sheet_name=f"Equities_USD_{freq.capitalize()}", index=False)

        # ---- Commodities ----
        for freq in ["daily","weekly","monthly"]:
            comm_cols = {name: s_map[freq] for name, s_map in comm_maps.items()}
            df_comm = to_wide_df(comm_cols, index_name="Date")
            df_comm.to_excel(xw, sheet_name=f"Commodities_{freq.capitalize()}", index=False)

        # ---- Crypto ----
        for freq in ["daily","weekly","monthly"]:
            cmaj_cols = {name: s_map[freq] for name, s_map in cmaj_maps.items()}
            df_cmaj = to_wide_df(cmaj_cols, index_name="Date")
            df_cmaj.to_excel(xw, sheet_name=f"Crypto_Majors_{freq.capitalize()}", index=False)

            calt_cols = {name: s_map[freq] for name, s_map in calt_maps.items()}
            df_calt = to_wide_df(calt_cols, index_name="Date")
            df_calt.to_excel(xw, sheet_name=f"Crypto_Alts_{freq.capitalize()}", index=False)

        # ---- Meta/Info ----
        meta = pd.DataFrame({
            "Key": ["GeneratedAt", "FRED_Terms", "DXY_Source"],
            "Value": [dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                      "Daily=raw / Weekly=W-FRI last / Monthly=EOM last",
                      dxy_used or "unknown"]
        })
        meta.to_excel(xw, sheet_name="Meta", index=False)

    print(f"✅ Saved: {XLSX_PATH.resolve()}")
    print("   Sheets: Macro_Inputs_* / Derived_* / Delta_* / Equities_Local_* / Equities_USD_* / Commodities_* / Crypto_*")

if __name__ == "__main__":
    main()
