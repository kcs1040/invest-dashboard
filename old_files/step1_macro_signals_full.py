# step1_macro_signals_full.py
# -----------------------------------------------------------
# Step 1 (Full, robust): 3년치 데이터로 거시 신호 계산 + 저장
# - FRED + yfinance 핵심 지표 수집
# - CPI YoY, 10Y Breakeven, Δ3M/Δ1Y, 곡선 레짐, 유동성 변화
# - MOVE proxy (10Y 일변동 21D 실현변동성, annualized, bps)
# - 견고한 리샘플(eom), yfinance 다중 티커 fallback
# - Macro_Dashboard_Data.xlsx (Inputs/Signals) + CSV, 업서트 저장
# -----------------------------------------------------------
import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
from pandas import ExcelWriter

# =========================
# 0) 설정/헬퍼
# =========================
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing. Put it in .env like: FRED_API_KEY=YOUR_KEY")

RECORD_TZ = os.getenv("RECORD_TZ", "local").lower()
OUTPUT_XLSX = os.getenv("OUTPUT_XLSX", "Macro_Dashboard_Data.xlsx")

today = dt.datetime.utcnow().date() if RECORD_TZ == "utc" else dt.date.today()
record_date_str = today.strftime("%Y-%m-%d")

# 3년치 기간
start_date = (today - dt.timedelta(days=3 * 365)).strftime("%Y-%m-%d")

fred = Fred(api_key=FRED_KEY)

def fred_series(code: str, start: str | None = None) -> pd.Series:
    """FRED 시리즈 로드 (DatetimeIndex, float Series)"""
    s = fred.get_series_latest_release(code)
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s.astype(float)

def yf_last_close_multi(candidates: list[str], period: str = "3y", interval: str = "1d") -> tuple[pd.DataFrame | None, float | None, str | None]:
    """
    yfinance에서 여러 티커를 순서대로 시도해 최초로 성공한 DF/Close/티커를 반환.
    """
    for tkr in candidates:
        try:
            df = yf.download(tkr, period=period, interval=interval, progress=False, auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                last_ser = df["Close"].dropna().iloc[-1]
                last_val = last_ser.item() if hasattr(last_ser, "item") else float(last_ser)
                return df, float(last_val), tkr
        except Exception:
            pass
    return None, None, None

def eom(x):
    """
    End-of-month 샘플러.
    - x: Series 또는 DataFrame (DatetimeIndex 필요)
    - DataFrame이면 'Close'가 있으면 그 컬럼, 없으면 첫 컬럼을 사용
    - (n,1) 모양도 squeeze로 1차원 Series로 변환
    - 월말(ME) 값으로 다운샘플
    """
    if x is None:
        return None
    if hasattr(x, "empty") and x.empty:
        return None

    # DataFrame -> Series
    if isinstance(x, pd.DataFrame):
        if "Close" in x.columns:
            x = x["Close"]
        else:
            x = x.iloc[:, 0]
        x = x.squeeze()

    # 여기서부터는 Series라고 가정
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    x = x.dropna()

    # 인덱스가 날짜가 아닐 수도 있으니 강제 변환
    x.index = pd.to_datetime(x.index, errors="coerce")
    x = x[~x.index.isna()]
    if x.empty:
        return None

    # 'M' 경고 회피: month-end 'ME'
    return x.resample("ME").last().dropna()

def last_val(s: pd.Series | None) -> float | None:
    if s is None:
        return None
    s2 = s.dropna()
    if len(s2) == 0:
        return None
    v = s2.iloc[-1]
    return v.item() if hasattr(v, "item") else float(v)

def delta_n_months(s: pd.Series | None, n_months: int) -> float | None:
    """월말 시리즈에서 n개월 변화(현재 - n개월 전)"""
    if s is None:
        return None
    s2 = s.dropna()
    if len(s2) < (n_months + 1):
        return None
    diff = s2.iloc[-1] - s2.iloc[-(n_months + 1)]
    if hasattr(diff, "item"):
        diff = diff.item()
    return float(diff)

# =========================
# 1) 데이터 로드
# =========================
# FRED (3년치)
fred_map = {
    "DGS10": "US10Y Nominal Yield (%)",
    "DGS2": "US2Y Nominal Yield (%)",
    "DFII10": "US10Y TIPS Real Yield (%)",
    "CPIAUCSL": "US CPI Index",
    "WALCL": "Fed Balance Sheet (Millions USD)",
    "BAMLH0A0HYM2": "HY OAS (bps)",
}
fred_df: dict[str, pd.Series] = {}
for code, label in fred_map.items():
    fred_df[label] = fred_series(code, start=start_date)

# CPI YoY (3년치로 충분)
cpi = fred_df["US CPI Index"]
cpi_yoy = None
if cpi is not None and len(cpi) >= 13:
    cpi_yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0

# Fed BS (조달러)
walcl_mn = fred_df["Fed Balance Sheet (Millions USD)"]
fed_bs_trn_last = (walcl_mn.iloc[-1] / 1_000_000) if walcl_mn is not None and len(walcl_mn) else None

# yfinance (3년치, 다중 티커 fallback)
# - DXY: "DX-Y.NYB" -> "DX=F" (선물)
# - VIX: "^VIX" (주요소스) -> "^VIX"만 사실상 실사용 가능, 그래도 목록 형태로 통일
# - BTC: "BTC-USD" -> "XBT-USD" -> "BTCUSD=X"
yf_candidates = {
    "DXY (Dollar Index)": ["DX-Y.NYB", "DX=F"],
    "USDKRW": ["KRW=X"],
    "VIX (Equity Vol)": ["^VIX"],
    "BTC Price (USD)": ["BTC-USD", "XBT-USD", "BTCUSD=X"],
}
yf_series: dict[str, pd.DataFrame | None] = {}
yf_closes: dict[str, float | None] = {}
yf_used_ticker: dict[str, str | None] = {}

for label, cands in yf_candidates.items():
    df, lastp, used = yf_last_close_multi(cands, period="3y", interval="1d")
    yf_series[label] = df
    yf_closes[label] = lastp
    yf_used_ticker[label] = used

# =========================
# 2) 월말 변환 및 파생계산
# =========================
# 월말 시리즈
dgs10_m  = eom(fred_df["US10Y Nominal Yield (%)"])
dgs2_m   = eom(fred_df["US2Y Nominal Yield (%)"])
tips10_m = eom(fred_df["US10Y TIPS Real Yield (%)"])
dxy_m    = eom(yf_series["DXY (Dollar Index)"])
usdk_m   = eom(yf_series["USDKRW"])
walcl_m  = eom(walcl_mn)  # (단위: 백만달러)

# 현재(일단위) 마지막 수치
last_10y  = last_val(fred_df["US10Y Nominal Yield (%)"])
last_2y   = last_val(fred_df["US2Y Nominal Yield (%)"])
last_tips = last_val(fred_df["US10Y TIPS Real Yield (%)"])
last_dxy  = yf_closes["DXY (Dollar Index)"]
last_usdk = yf_closes["USDKRW"]
last_vix  = yf_closes["VIX (Equity Vol)"]
last_btc  = yf_closes["BTC Price (USD)"]
last_hyoas = last_val(fred_df["HY OAS (bps)"])

# 곡선(10Y-2Y, bps)
curve_bps = None
if last_10y is not None and last_2y is not None:
    curve_bps = (last_10y - last_2y) * 100.0

# 브레이크이븐(=명목-실질, bps)
breakeven_bps = None
if (last_10y is not None) and (last_tips is not None):
    breakeven_bps = (last_10y - last_tips) * 100.0

# Δ3M / Δ1Y (월말 시리즈 기반)
def deltas_block(series: pd.Series | None) -> tuple[float | None, float | None]:
    return delta_n_months(series, 3), delta_n_months(series, 12)

curve_m = None
if (dgs10_m is not None) and (dgs2_m is not None):
    curve_m = (dgs10_m - dgs2_m) * 100.0  # bps

tips3m_bp, tips1y_bp = deltas_block(tips10_m)  # 실질금리 Δ (pp) → bp 변환
if tips3m_bp is not None: tips3m_bp *= 100.0
if tips1y_bp is not None: tips1y_bp *= 100.0

dxy3m, dxy1y   = deltas_block(dxy_m)
usdk3m, usdk1y = deltas_block(usdk_m)

fedbs3m_trn, fedbs1y_trn = None, None
if walcl_m is not None:
    walcl_trn_m = walcl_m / 1_000_000.0
    fedbs3m_trn = delta_n_months(walcl_trn_m, 3)
    fedbs1y_trn = delta_n_months(walcl_trn_m, 12)

curve3m_bp, curve1y_bp = deltas_block(curve_m)

# 곡선 레짐
curve_regime = None
if curve3m_bp is not None and dgs10_m is not None:
    ten3m = delta_n_months(dgs10_m, 3)
    if (ten3m is not None) and (curve3m_bp is not None):
        if curve3m_bp > 0 and ten3m > 0:
            curve_regime = "Bear steepening"
        elif curve3m_bp > 0 and ten3m < 0:
            curve_regime = "Bull steepening"

# MOVE proxy (10Y 일변동 21D 실현변동성, annualized, bps)
move_proxy = None
dgs10_daily = fred_series("DGS10", start=start_date)
if dgs10_daily is not None and len(dgs10_daily.dropna()) >= 22:
    chg_bp = dgs10_daily.dropna().diff().dropna() * 100.0
    rolling_std_bp = chg_bp.rolling(21).std().dropna()
    if len(rolling_std_bp) > 0:
        move_proxy = float(rolling_std_bp.iloc[-1] * np.sqrt(252.0))

# =========================
# 3) 점수화(Risk-On/Off)
# =========================
risk_off = 0
risk_on  = 0

if tips3m_bp is not None:
    if tips3m_bp > 25:   # 실질금리 3개월 +25bp↑
        risk_off += 1
    if tips3m_bp < -25:  # 실질금리 3개월 -25bp↓
        risk_on  += 1

if dxy3m is not None:
    if dxy3m > 2:
        risk_off += 1
    if dxy3m < -2:
        risk_on  += 1

if curve_regime == "Bear steepening":
    risk_off += 1
elif curve_regime == "Bull steepening":
    risk_on  += 1

if fedbs3m_trn is not None and fedbs3m_trn > 0.1:
    risk_on  += 1

# =========================
# 4) 저장용 테이블
# =========================
inputs_row = {
    "Date (YYYY-MM-DD)": record_date_str,
    "US10Y Nominal Yield (%)": last_10y,
    "US2Y Nominal Yield (%)": last_2y,
    "US10Y TIPS Real Yield (%)": last_tips,
    "DXY (Dollar Index)": last_dxy,
    "USDKRW": last_usdk,
    "US CPI YoY (%)": cpi_yoy,
    "Fed Balance Sheet (Trn USD)": fed_bs_trn_last,
    "MOVE (Bond Vol)": move_proxy,
    "VIX (Equity Vol)": last_vix,
    "HY OAS (bps)": last_hyoas,
    "BTC Price (USD)": last_btc,
}

signals_row = {
    "Last Date": record_date_str,
    "10Y–2Y Curve (bps)": curve_bps,
    "10Y Breakeven (bps)": breakeven_bps,
    "Real Yield Δ3M (bp)": tips3m_bp,
    "Real Yield Δ1Y (bp)": tips1y_bp,
    "DXY Δ3M": dxy3m,
    "DXY Δ1Y": dxy1y,
    "USDKRW Δ3M": usdk3m,
    "USDKRW Δ1Y": usdk1y,
    "Liquidity Δ3M (Trn USD)": fedbs3m_trn,
    "Liquidity Δ1Y (Trn USD)": fedbs1y_trn,
    "Curve Δ3M (bps)": curve3m_bp,
    "Curve Δ1Y (bps)": curve1y_bp,
    "Curve Regime": curve_regime,
    "Risk-Off Score": risk_off,
    "Risk-On Score": risk_on,
    # 디버그용: 어떤 티커가 실제로 사용됐는지 기록(유용)
    "DXY ticker used": yf_used_ticker["DXY (Dollar Index)"],
    "VIX ticker used": yf_used_ticker["VIX (Equity Vol)"],
    "BTC ticker used": yf_used_ticker["BTC Price (USD)"],
}

inputs_df  = pd.DataFrame([inputs_row])
signals_df = pd.DataFrame([signals_row])

# 업서트(같은 날짜면 교체)
xlsx_path = Path(OUTPUT_XLSX)
if xlsx_path.exists():
    try:
        existing = pd.read_excel(xlsx_path, sheet_name="Inputs")
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

with ExcelWriter(xlsx_path, engine="openpyxl", mode="w") as w:
    final_inputs.to_excel(w, sheet_name="Inputs", index=False)
    signals_df.to_excel(w, sheet_name="Signals", index=False)

# CSV 백업
inputs_df.to_csv("Inputs_latest.csv", index=False)
signals_df.to_csv("Signals_latest.csv", index=False)

# 콘솔 출력
print("=== INPUTS (latest) ===")
print(inputs_df.to_string(index=False))
print("\n=== SIGNALS ===")
print(signals_df.to_string(index=False))
print(f"\nSaved {OUTPUT_XLSX} (Inputs & Signals) + CSV copies.")
