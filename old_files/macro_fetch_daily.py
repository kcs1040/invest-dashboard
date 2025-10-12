# macro_fetch_daily.py
# -----------------------------------------------------------
# Daily updater for macro dashboard (FRED + yfinance)
# -----------------------------------------------------------
import os
import datetime as dt
import pandas as pd
from fredapi import Fred
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # .env 파일 자동 로드
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY not set. Put it in .env like: FRED_API_KEY=YOUR_KEY")

fred = Fred(api_key=FRED_KEY)
OUTPUT_XLSX = os.getenv("OUTPUT_XLSX", "Macro_Dashboard_Data.xlsx")
RECORD_TZ = os.getenv("RECORD_TZ", "local").lower()

record_date = dt.datetime.utcnow().date() if RECORD_TZ == "utc" else dt.date.today()
record_date_str = record_date.strftime("%Y-%m-%d")

# --- FRED series ---
fred_series = {
    "US10Y Nominal Yield (%)": "DGS10",
    "US2Y Nominal Yield (%)": "DGS2",
    "US10Y TIPS Real Yield (%)": "DFII10",
    "US CPI Index (CPIAUCSL)": "CPIAUCSL",
    "Fed Balance Sheet (Millions USD)": "WALCL",
    "HY OAS (bps)": "BAMLH0A0HYM2",
}
def fred_last_value(code):
    s = fred.get_series_latest_release(code)
    return float(s.dropna().iloc[-1]) if s is not None and len(s.dropna()) else None

fred_vals = {}
for label, code in fred_series.items():
    try:
        fred_vals[label] = fred_last_value(code)
    except Exception as e:
        fred_vals[label] = None
        print(f"[WARN] FRED fetch failed for {label} ({code}): {e}")

# CPI YoY 계산
try:
    cpi_series = fred.get_series_latest_release("CPIAUCSL").dropna()
    cpi_yoy = (cpi_series.iloc[-1] / cpi_series.iloc[-13] - 1.0) * 100.0 if len(cpi_series) >= 13 else None
except Exception:
    cpi_yoy = None

walcl_mn = fred_vals.get("Fed Balance Sheet (Millions USD)")
fed_bs_trn = walcl_mn / 1_000_000 if walcl_mn is not None else None

# --- yfinance series ---
yf_tickers = {
    "DXY (Dollar Index)": "DX-Y.NYB",
    "USDKRW": "KRW=X",
    "VIX (Equity Vol)": "^VIX",
    "BTC Price (USD)": "BTC-USD",
}
yf_vals = {}
for label, ticker in yf_tickers.items():
    try:
        data = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
        yf_vals[label] = float(data["Close"].dropna().iloc[-1]) if not data.empty else None
    except Exception as e:
        yf_vals[label] = None
        print(f"[WARN] yfinance fetch failed for {label} ({ticker}): {e}")

# --- Excel 한 행 데이터 ---
row = {
    "Date (YYYY-MM-DD)": record_date_str,
    "US10Y Nominal Yield (%)": fred_vals.get("US10Y Nominal Yield (%)"),
    "US2Y Nominal Yield (%)": fred_vals.get("US2Y Nominal Yield (%)"),
    "US10Y TIPS Real Yield (%)": fred_vals.get("US10Y TIPS Real Yield (%)"),
    "DXY (Dollar Index)": yf_vals.get("DXY (Dollar Index)"),
    "USDKRW": yf_vals.get("USDKRW"),
    "US CPI YoY (%)": cpi_yoy,
    "Fed Balance Sheet (Trn USD)": fed_bs_trn,
    "MOVE (Bond Vol)": None,
    "VIX (Equity Vol)": yf_vals.get("VIX (Equity Vol)"),
    "HY OAS (bps)": fred_vals.get("HY OAS (bps)"),
    "BTC Price (USD)": yf_vals.get("BTC Price (USD)"),
}
new_df = pd.DataFrame([row])

# --- Excel UPSERT (같은 날짜면 교체, 없으면 추가) ---
if os.path.exists(OUTPUT_XLSX):
    try:
        existing = pd.read_excel(OUTPUT_XLSX, sheet_name="Inputs")
    except Exception:
        existing = None
else:
    existing = None

if existing is not None and not existing.empty and "Date (YYYY-MM-DD)" in existing.columns:
    mask = existing["Date (YYYY-MM-DD)"].astype(str) == record_date_str
    if mask.any():
        existing.loc[mask, :] = new_df.iloc[0].values
        out_df = existing
    else:
        out_df = pd.concat([existing, new_df], ignore_index=True)
else:
    out_df = new_df

try:
    out_df["Date (YYYY-MM-DD)"] = pd.to_datetime(out_df["Date (YYYY-MM-DD)"]).dt.date.astype(str)
    out_df = out_df.sort_values("Date (YYYY-MM-DD)")
except Exception:
    pass

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
    out_df.to_excel(writer, sheet_name="Inputs", index=False)

print(f"[OK] Upserted {record_date_str} into {OUTPUT_XLSX}")
print(out_df.tail(5).to_string(index=False))
