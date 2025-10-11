# step1_export_excel_full_plus_compact_pro.py
# -----------------------------------------------------------
# Full dataset (Daily/Weekly/Monthly) + Compact (Inputs & Signals)
# + Customizable Risk Score Rules + Extra Compact Metrics (Gold/Oil)
# - No ffill densify; use raw daily + downsample weekly/monthly (last)
# - All dates exported as ISO string to avoid Excel serials
# - Column/Sheet names aligned with Plotly dashboard
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
XLSX_PATH = OUTDIR / "Macro_Dashboard_Full_plus_Compact_PRO.xlsx"

today = dt.date.today()
start_date = (today - dt.timedelta(days=3*365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# ---- Risk Score Rules (커스터마이즈 포인트) ----
RISK_RULES = {
    # Risk-ON: 조건을 만족하면 +1
    "risk_on": {
        "vix_lt": 18.0,         # VIX < 18
        "hyoas_lt": 350.0,      # HY OAS < 350 bps
        "dxy_3m_lt": 0.0,       # DXY Δ3M < 0
        "real_3m_bp_lt": 0.0,   # 10Y TIPS Δ3M (bp) < 0
    },
    # Risk-OFF: 조건을 만족하면 +1
    "risk_off": {
        "vix_gt": 25.0,         # VIX > 25
        "hyoas_gt": 450.0,      # HY OAS > 450 bps
        "dxy_3m_gt": 0.0,       # DXY Δ3M > 0
        "real_3m_bp_gt": 0.0,   # 10Y TIPS Δ3M (bp) > 0
    }
}

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
    df = df.reset_index().rename(columns={"index": index_name})
    df[index_name] = pd.to_datetime(df[index_name]).dt.date.astype(str)
    return df

def delta_n(s, n):
    s = num_series(s)
    if s is None or len(s) < (n + 1):
        return None
    v = s.iloc[-1] - s.iloc[-(n + 1)]
    return float(v)

# ===================== USD rebasing =====================
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

# ===================== Compact helpers =====================
def eom(s: pd.Series | None) -> pd.Series | None:
    s = num_series(s)
    if s is None or s.empty:
        return None
    return s.resample("ME").last().dropna()

def curve_regime(curve_bps: float | None, dgs10_chg3m: float | None):
    if curve_bps is None or dgs10_chg3m is None:
        return "Unknown"
    if curve_bps >= 0 and dgs10_chg3m >= 0:
        return "Bear steepening"
    if curve_bps >= 0 and dgs10_chg3m < 0:
        return "Bull steepening"
    if curve_bps < 0 and dgs10_chg3m >= 0:
        return "Bear flattening"
    return "Bull flattening"

def risk_scores(vix, hyoas, dxy_chg3m, real_chg3m_bp, rules: dict):
    ro = 0; rf = 0
    on, off = rules["risk_on"], rules["risk_off"]
    if vix is not None and on.get("vix_lt")   is not None and vix <  on["vix_lt"]:   ro += 1
    if hyoas is not None and on.get("hyoas_lt")is not None and hyoas < on["hyoas_lt"]:ro += 1
    if dxy_chg3m is not None and on.get("dxy_3m_lt") is not None and dxy_chg3m < on["dxy_3m_lt"]: ro += 1
    if real_chg3m_bp is not None and on.get("real_3m_bp_lt") is not None and real_chg3m_bp < on["real_3m_bp_lt"]: ro += 1

    if vix is not None and off.get("vix_gt")  is not None and vix >  off["vix_gt"]:  rf += 1
    if hyoas is not None and off.get("hyoas_gt")is not None and hyoas > off["hyoas_gt"]:rf += 1
    if dxy_chg3m is not None and off.get("dxy_3m_gt") is not None and dxy_chg3m > off["dxy_3m_gt"]: rf += 1
    if real_chg3m_bp is not None and off.get("real_3m_bp_gt") is not None and real_chg3m_bp > off["real_3m_bp_gt"]: rf += 1
    return ro, rf

# ===================== Main =====================
def main():
    # ---------- Macro core ----------
    dgs10 = fred_series_all_freq("DGS10",  start=start_date)   # 10Y nominal
    dgs2  = fred_series_all_freq("DGS2",   start=start_date)   # 2Y nominal
    tips10= fred_series_all_freq("DFII10", start=start_date)   # 10Y real
    walcl = fred_series_all_freq("WALCL",  start=start_date)   # Fed BS (mn USD)
    hyoas = fred_series_all_freq("BAMLH0A0HYM2", start=start_date)  # HY OAS (bps)

    # CPI YoY
    try:
        cpi = fred_series("CPIAUCSL", start=start_date)
        cpi_m = cpi.resample("ME").last().dropna()
        cpi_yoy = ((cpi_m / cpi_m.shift(12)) - 1.0) * 100.0
    except Exception:
        cpi_yoy = None

    # MOVE (optional)
    move_df, _ = yf_series_multi(["^MOVE"])
    move = market_series_all_freq(move_df) if move_df is not None else {'daily':None,'weekly':None,'monthly':None}

    # Market (yfinance)
    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
    usdk_df, _       = yf_series_multi(["KRW=X"])
    vix_df, _        = yf_series_multi(["^VIX"])
    btc_df, _        = yf_series_multi(["BTC-USD", "XBT-USD", "BTCUSD=X"])

    dxy  = market_series_all_freq(dxy_df) if dxy_df is not None else {'daily':None,'weekly':None,'monthly':None}
    usdk = market_series_all_freq(usdk_df) if usdk_df is not None else {'daily':None,'weekly':None,'monthly':None}
    vix  = market_series_all_freq(vix_df) if vix_df is not None else {'daily':None,'weekly':None,'monthly':None}
    btc  = market_series_all_freq(btc_df) if btc_df is not None else {'daily':None,'weekly':None,'monthly':None}

    # Derived per freq
    def map_apply(m1, m2, fn):
        return {k: (fn(m1[k], m2[k]) if (m1.get(k) is not None and m2.get(k) is not None) else None)
                for k in ["daily","weekly","monthly"]}
    breakeven = map_apply(dgs10, tips10, lambda a,b: (a-b)*100.0)  # bps
    curve = map_apply(dgs10, dgs2,   lambda a,b: (a-b)*100.0)      # bps
    walcl_trn = {k: (walcl[k]/1_000_000.0 if walcl.get(k) is not None else None)
                 for k in ["daily","weekly","monthly"]}            # Trn USD

    # ---------- Global Equities (Local/USD) ----------
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

    # Commodities (Gold/Oil 등)
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

    # ===================== Write Excel =====================
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:

        # ---- Full dataset: Macro per freq ----
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
                "MOVE (Bond Vol)":   move[freq],
            }
            to_wide_df(macro_inputs, "Date").to_excel(xw, sheet_name=f"Macro_Inputs_{freq.capitalize()}", index=False)

            derived = {
                "10Y Breakeven (bps)": breakeven[freq],
                "10Y–2Y Curve (bps)":  curve[freq],
                "Fed Balance Sheet (Trn USD)": walcl_trn[freq],
            }
            to_wide_df(derived, "Date").to_excel(xw, sheet_name=f"Derived_{freq.capitalize()}", index=False)

        # ---- Full dataset: Equities/Commodities per freq ----
        for freq in ["daily","weekly","monthly"]:
            to_wide_df({n: m[f] for n,m in local_maps.items()}, "Date").to_excel(
                xw, sheet_name=f"Equities_Local_{freq.capitalize()}", index=False)
            to_wide_df({n: m[f] for n,m in usd_maps.items()},   "Date").to_excel(
                xw, sheet_name=f"Equities_USD_{freq.capitalize()}", index=False)
            to_wide_df({n: m[f] for n,m in comm_maps.items()},  "Date").to_excel(
                xw, sheet_name=f"Commodities_{freq.capitalize()}", index=False)

        # ---- Full dataset: Crypto per freq ----
        crypto_majors = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD"]
        crypto_alts   = ["ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","LTC-USD"]
        def fetch_crypto(symbols):
            out = {}
            for t in symbols:
                df, _ = yf_series_multi([t])
                out[t.replace("-USD","")] = market_series_all_freq(df) if df is not None else {'daily':None,'weekly':None,'monthly':None}
            return out
        cmaj_maps = fetch_crypto(crypto_majors)
        calt_maps = fetch_crypto(crypto_alts)
        for freq in ["daily","weekly","monthly"]:
            to_wide_df({n: m[freq] for n,m in cmaj_maps.items()}, "Date").to_excel(
                xw, sheet_name=f"Crypto_Majors_{freq.capitalize()}", index=False)
            to_wide_df({n: m[freq] for n,m in calt_maps.items()}, "Date").to_excel(
                xw, sheet_name=f"Crypto_Alts_{freq.capitalize()}", index=False)

        # ===================== Compact: Inputs & Signals =====================
        # EOM series
        dgs10_m = eom(dgs10["monthly"]); dgs2_m = eom(dgs2["monthly"]); tips_m = eom(tips10["monthly"])
        dxy_m   = eom(dxy["monthly"]);   usdk_m = eom(usdk["monthly"]); vix_m  = eom(vix["monthly"])
        hyoas_m = eom(hyoas["monthly"]); walcl_m= eom(walcl_trn["monthly"]); btc_m = eom(btc["monthly"])
        move_m  = eom(move["monthly"]) if move["monthly"] is not None else None

        # Extra compact assets: Gold/Oil (EOM)
        gold_m = eom(comm_maps.get("Gold (GC=F)", {}).get("monthly") if "Gold (GC=F)" in comm_maps else None)
        oil_m  = eom(comm_maps.get("Crude Oil (CL=F)", {}).get("monthly") if "Crude Oil (CL=F)" in comm_maps else None)

        # CPI YoY (EOM)
        cpi_yoy_m = eom(cpi_yoy) if 'cpi_yoy' in locals() and cpi_yoy is not None else None

        # Last date
        last_date = None
        for s in [dgs10_m, dgs2_m, tips_m, dxy_m, usdk_m, vix_m, hyoas_m, walcl_m, btc_m, move_m, cpi_yoy_m, gold_m, oil_m]:
            if s is not None and len(s):
                last_date = s.index[-1]
        last_date_str = last_date.date().isoformat() if last_date is not None else ""

        # Compact Inputs (최신값 스냅샷)
        inputs_row = {
            "Date (YYYY-MM-DD)": last_date_str,
            "US10Y Nominal Yield (%)": float(dgs10_m.iloc[-1]) if dgs10_m is not None and len(dgs10_m) else None,
            "US2Y Nominal Yield (%)":  float(dgs2_m.iloc[-1])  if dgs2_m is not None and len(dgs2_m)  else None,
            "10Y TIPS Real Yield (%)": float(tips_m.iloc[-1])  if tips_m is not None and len(tips_m)  else None,
            "10Y Breakeven (bps)": float(((dgs10_m - tips_m)*100.0).iloc[-1]) if dgs10_m is not None and tips_m is not None and len(dgs10_m) and len(tips_m) else None,
            "10Y–2Y Curve (bps)":  float(((dgs10_m - dgs2_m)*100.0).iloc[-1])  if dgs10_m is not None and dgs2_m is not None and len(dgs10_m) and len(dgs2_m) else None,
            "DXY (Dollar Index)":       float(dxy_m.iloc[-1]) if dxy_m is not None and len(dxy_m) else None,
            "USDKRW":                    float(usdk_m.iloc[-1]) if usdk_m is not None and len(usdk_m) else None,
            "VIX (Equity Vol)":         float(vix_m.iloc[-1]) if vix_m is not None and len(vix_m) else None,
            "HY OAS (bps)":             float(hyoas_m.iloc[-1]) if hyoas_m is not None and len(hyoas_m) else None,
            "Fed Balance Sheet (Trn USD)": float(walcl_m.iloc[-1]) if walcl_m is not None and len(walcl_m) else None,
            "US CPI YoY (%)":           float(cpi_yoy_m.iloc[-1]) if cpi_yoy_m is not None and len(cpi_yoy_m) else None,
            "MOVE (Bond Vol)":          float(move_m.iloc[-1]) if move_m is not None and len(move_m) else None,
            "BTC Price (USD)":          float(btc_m.iloc[-1]) if btc_m is not None and len(btc_m) else None,
            "Gold (USD)":               float(gold_m.iloc[-1]) if gold_m is not None and len(gold_m) else None,
            "Crude Oil (USD)":          float(oil_m.iloc[-1])  if oil_m  is not None and len(oil_m)  else None,
        }
        pd.DataFrame([inputs_row]).to_excel(xw, sheet_name="Compact_Inputs", index=False)

        # Compact Signals (Δ3M, Regime, Risk Scores)
        def delta3(s):  # 최근값 - 3개월 전
            s = num_series(s)
            if s is None or len(s) < 4:
                return None
            return float(s.iloc[-1] - s.iloc[-4])

        be_m     = (dgs10_m - tips_m) * 100.0 if dgs10_m is not None and tips_m is not None else None
        curve_m  = (dgs10_m - dgs2_m) * 100.0 if dgs10_m is not None and dgs2_m is not None else None
        real_chg3m_bp = delta3(tips_m)*100.0 if tips_m is not None and len(tips_m) >= 4 else None
        dxy_chg3m     = delta3(dxy_m)
        usdk_chg3m    = delta3(usdk_m)
        liq_chg3m     = delta3(walcl_m)
        vix_chg3m     = delta3(vix_m)
        hyoas_chg3m   = delta3(hyoas_m)
        spx_m         = eom(local_maps["S&P 500 (^GSPC)"]["monthly"])
        spx_chg3m     = delta3(spx_m)
        gold_chg3m    = delta3(gold_m)
        oil_chg3m     = delta3(oil_m)
        dgs10_chg3m   = delta3(dgs10_m)

        curve_bps  = float(curve_m.iloc[-1]) if curve_m is not None and len(curve_m) else None
        regime     = curve_regime(curve_bps, dgs10_chg3m)

        vix_last   = float(vix_m.iloc[-1])   if vix_m   is not None and len(vix_m)   else None
        hyoas_last = float(hyoas_m.iloc[-1]) if hyoas_m is not None and len(hyoas_m) else None
        ro, rf     = risk_scores(vix_last, hyoas_last, dxy_chg3m, real_chg3m_bp, RISK_RULES)

        signals_row = {
            "Last Date": inputs_row["Date (YYYY-MM-DD)"],
            "10Y–2Y Curve (bps)": curve_bps,
            "10Y Breakeven (bps)": float(be_m.iloc[-1]) if be_m is not None and len(be_m) else None,

            "Real Δ3M (bp)": real_chg3m_bp,
            "DXY Δ3M": dxy_chg3m,
            "USDKRW Δ3M": usdk_chg3m,
            "Liquidity Δ3M (Trn USD)": liq_chg3m,

            "VIX Δ3M": vix_chg3m,
            "HY OAS Δ3M (bp)": hyoas_chg3m,
            "S&P 500 Δ3M (local)": spx_chg3m,

            "Gold Δ3M (USD)": gold_chg3m,
            "Crude Oil Δ3M (USD)": oil_chg3m,

            "Curve Regime": regime,
            "Risk-On Score": ro,
            "Risk-Off Score": rf,
        }
        pd.DataFrame([signals_row]).to_excel(xw, sheet_name="Compact_Signals", index=False)

        # ---- Settings / Meta ----
        meta = pd.DataFrame({
            "Key": ["GeneratedAt", "FRED_Terms", "DXY_Source", "Notes"],
            "Value": [dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                      "Daily=raw / Weekly=W-FRI last / Monthly=EOM last (no ffill)",
                      dxy_used or "unknown",
                      "MOVE from ^MOVE if available; CPI YoY from FRED CPIAUCSL"],
        })
        meta.to_excel(xw, sheet_name="Meta", index=False)

        # Risk Rules 기록(가독성있게)
        rr = []
        for k,v in RISK_RULES["risk_on"].items():
            rr.append(("risk_on", k, v))
        for k,v in RISK_RULES["risk_off"].items():
            rr.append(("risk_off", k, v))
        pd.DataFrame(rr, columns=["Group","Rule","Threshold"]).to_excel(
            xw, sheet_name="Settings_RiskRules", index=False)

    print(f"✅ Saved: {XLSX_PATH.resolve()}")
    print("   Includes: Full(per freq) + Compact_Inputs + Compact_Signals + Settings_RiskRules + Meta")

if __name__ == "__main__":
    main()
