
# step1_macro_signals.py
import os, datetime as dt, pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
import yfinance as yf
from pathlib import Path

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing. Put it in .env (copy from .env.example).")
fred = Fred(api_key=FRED_KEY)

RECORD_TZ = os.getenv("RECORD_TZ", "local").lower()
OUTPUT_XLSX = os.getenv("OUTPUT_XLSX", "Macro_Dashboard_Data.xlsx")

today = dt.datetime.utcnow().date() if RECORD_TZ == "utc" else dt.date.today()
record_date_str = today.strftime("%Y-%m-%d")

def fred_series(code, start=None):
    s = fred.get_series_latest_release(code)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s.dropna()

def yf_last_close(ticker, period="9mo", interval="1d"):
    import pandas as pd
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None, None
    # 안전하게 스칼라로 변환 (FutureWarning 회피)
    last_ser = df["Close"].dropna().iloc[-1]
    last_val = last_ser.item() if hasattr(last_ser, "item") else float(last_ser)
    return df, float(last_val)


start_date = (today - dt.timedelta(days=280)).strftime("%Y-%m-%d")
series_map = {
    "DGS10": "US10Y Nominal Yield (%)",
    "DGS2": "US2Y Nominal Yield (%)",
    "DFII10": "US10Y TIPS Real Yield (%)",
    "CPIAUCSL": "US CPI Index",
    "WALCL": "Fed Balance Sheet (Millions USD)",
    "BAMLH0A0HYM2": "HY OAS (bps)",
}
fred_df = {label: fred_series(code, start=start_date) for code, label in series_map.items()}

cpi = fred_df["US CPI Index"]
cpi_yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0 if len(cpi) >= 13 else None

walcl = fred_df["Fed Balance Sheet (Millions USD)"]
fed_bs_trn = walcl.iloc[-1] / 1_000_000 if len(walcl) else None

tickers = {
    "DXY (Dollar Index)": "DX-Y.NYB",
    "USDKRW": "KRW=X",
    "VIX (Equity Vol)": "^VIX",
    "BTC Price (USD)": "BTC-USD",
}
yf_closes, yf_series = {}, {}
for label, t in tickers.items():
    df, last = yf_last_close(t)
    yf_series[label] = df
    yf_closes[label] = last

def eom(s):
    """
    End-of-month 샘플러.
    - s: Series 또는 DataFrame (DatetimeIndex 필수)
    - DataFrame이면 'Close' 컬럼을 우선, 없으면 첫 컬럼 사용
    - 월말(Month End) 값으로 다운샘플
    """
    import pandas as pd

    if s is None:
        return None
    if hasattr(s, "empty") and s.empty:
        return None

    # DataFrame → Series로 통일
    if isinstance(s, pd.DataFrame):
        if "Close" in s.columns:
            s = s["Close"]
        else:
            s = s.iloc[:, 0]  # 첫 컬럼

    # 이제 s는 Series라고 가정
    s = s.dropna()
    # 'M' 대신 'ME' (month-end)
    return s.resample("ME").last().dropna()



def last_val(s):
    try: return float(s.dropna().iloc[-1])
    except: return None

def delta_3m(s):
    try:
        s2 = s.dropna()
        if len(s2) < 4:
            return None
        diff = s2.iloc[-1] - s2.iloc[-4]
        # 스칼라/Series 모두 지원
        if hasattr(diff, "iloc"):
            diff = diff.iloc[0]
        return float(diff)
    except Exception:
        return None


dgs10 = eom(fred_df["US10Y Nominal Yield (%)"])
dgs2 = eom(fred_df["US2Y Nominal Yield (%)"])
tips10 = eom(fred_df["US10Y TIPS Real Yield (%)"])
dxy = eom(yf_series["DXY (Dollar Index)"]) if yf_series["DXY (Dollar Index)"] is not None and not yf_series["DXY (Dollar Index)"].empty else None
usdk = eom(yf_series["USDKRW"]) if yf_series["USDKRW"] is not None and not yf_series["USDKRW"].empty else None

fedbs = eom(fred_df["Fed Balance Sheet (Millions USD)"]) if len(walcl) else None

last_10y, last_2y, last_tips = last_val(fred_df["US10Y Nominal Yield (%)"]), last_val(fred_df["US2Y Nominal Yield (%)"]), last_val(fred_df["US10Y TIPS Real Yield (%)"])
last_dxy, last_usdk = yf_closes["DXY (Dollar Index)"], yf_closes["USDKRW"]

curve_m = (dgs10 - dgs2) if dgs10 is not None and dgs2 is not None else None
curve_3m = delta_3m(curve_m)
tips_3m_bp = (delta_3m(tips10)*100) if tips10 is not None and delta_3m(tips10) is not None else None
dxy_3m = delta_3m(dxy) if dxy is not None else None
usdk_3m = delta_3m(usdk) if usdk is not None else None
fedbs_3m = (delta_3m(fedbs)/1_000_000) if fedbs is not None and delta_3m(fedbs) is not None else None

breakeven = ((last_10y - last_tips)*100) if (last_10y is not None and last_tips is not None) else None

risk_off = 0; risk_on = 0; curve_dir = None
if tips_3m_bp is not None:
    if tips_3m_bp > 25: risk_off += 1
    if tips_3m_bp < -25: risk_on += 1
if dxy_3m is not None:
    if dxy_3m > 2: risk_off += 1
    if dxy_3m < -2: risk_on += 1
if curve_3m is not None and dgs10 is not None:
    ten_delta_3m = delta_3m(dgs10)
    if ten_delta_3m is not None:
        if curve_3m > 0 and ten_delta_3m > 0:
            curve_dir = "Bear steepening"; risk_off += 1
        elif curve_3m > 0 and ten_delta_3m < 0:
            curve_dir = "Bull steepening"; risk_on += 1
if fedbs_3m is not None and fedbs_3m > 0.1:
    risk_on += 1

inputs_row = {
    "Date (YYYY-MM-DD)": record_date_str,
    "US10Y Nominal Yield (%)": last_10y,
    "US2Y Nominal Yield (%)": last_2y,
    "US10Y TIPS Real Yield (%)": last_tips,
    "DXY (Dollar Index)": last_dxy,
    "USDKRW": last_usdk,
    "US CPI YoY (%)": cpi_yoy,
    "Fed Balance Sheet (Trn USD)": (last_val(walcl)/1_000_000) if len(walcl) else None,
    "MOVE (Bond Vol)": None,
    "VIX (Equity Vol)": yf_closes.get("VIX (Equity Vol)"),
    "HY OAS (bps)": last_val(fred_df["HY OAS (bps)"]),
    "BTC Price (USD)": yf_closes.get("BTC Price (USD)"),
}
signals = {
    "Last Date": record_date_str,
    "10Y–2Y Curve (bps)": (last_10y - last_2y)*100 if (last_10y is not None and last_2y is not None) else None,
    "10Y Breakeven (bps)": breakeven,
    "Real Yield Trend 3m (bp)": tips_3m_bp,
    "DXY Trend 3m (Δ)": dxy_3m,
    "USDKRW Trend 3m (Δ)": usdk_3m,
    "Liquidity Δ 3m (Trn USD)": fedbs_3m,
    "Curve Regime": curve_dir,
    "Risk-Off Score": risk_off,
    "Risk-On Score": risk_on,
}

inputs_df = pd.DataFrame([inputs_row])
signals_df = pd.DataFrame([signals])

from openpyxl import load_workbook
from pandas import ExcelWriter
xlsx = Path(OUTPUT_XLSX)
if xlsx.exists():
    try:
        existing = pd.read_excel(xlsx, sheet_name="Inputs")
    except Exception:
        existing = pd.DataFrame(columns=list(inputs_row.keys()))
    if "Date (YYYY-MM-DD)" in existing.columns:
        mask = existing["Date (YYYY-MM-DD)"].astype(str) == record_date_str
        if mask.any():
            existing.loc[mask, :] = inputs_df.iloc[0].values
            final_inputs = existing
        else:
            final_inputs = pd.concat([existing, inputs_df], ignore_index=True)
    else:
        final_inputs = pd.concat([existing, inputs_df], ignore_index=True)
else:
    final_inputs = inputs_df

with ExcelWriter(xlsx, engine="openpyxl", mode="w") as w:
    final_inputs.to_excel(w, sheet_name="Inputs", index=False)
    signals_df.to_excel(w, sheet_name="Signals", index=False)

inputs_df.to_csv("Inputs_latest.csv", index=False)
signals_df.to_csv("Signals_latest.csv", index=False)

print("=== INPUTS ===")
print(inputs_df.to_string(index=False))
print("\n=== SIGNALS ===")
print(signals_df.to_string(index=False))
print(f"\nSaved {OUTPUT_XLSX} (Inputs & Signals) + CSV copies.")
