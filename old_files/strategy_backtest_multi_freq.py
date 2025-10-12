# -*- coding: utf-8 -*-
"""
strategy_backtest_multi_freq.py
- Regime 전략: ME / 2W-FRI / W-FRI 리밸런싱, 월말 DCA, 거래비용, XIRR
- Benchmarks(단일자산: SPY, TLT, BTC-USD, GLD): "일간 기반 + 달력 월말 납입", 빈도 무관(ALL) 한 줄
- 최종 결과: 전략 3줄 + 벤치 4줄을 하나의 성과표와 시계열 CSV로 저장
"""

import os, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)  # pandas 3.x 경고 억제

from dateutil.relativedelta import relativedelta
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# ========== 환경설정 ==========
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing in .env")

fred = Fred(api_key=FRED_KEY)

START_YEARS = 10          # 과거 조회 연수
RFR_ASSUMPTION = 0.02     # Sharpe용 무위험 수익률 가정

# 유니버스(8개, 4:4)
TICKERS_RISKON    = ["SPY", "QQQ", "EEM", "BTC-USD"]
TICKERS_DEFENSIVE = ["IEF", "TLT", "GLD", "BIL"]
ALL_TICKERS = TICKERS_RISKON + TICKERS_DEFENSIVE

# 벤치마크 대상 (단일자산)
BENCH_TICKERS = ["SPY", "TLT", "BTC-USD", "GLD"]

# 레짐별 목표 가중치 (합=1.0)
WEIGHTS = {
    "risk_on":  {"SPY":0.30,"QQQ":0.20,"EEM":0.15,"BTC-USD":0.10,"IEF":0.10,"TLT":0.05,"GLD":0.05,"BIL":0.05},
    "neutral":  {"SPY":0.20,"QQQ":0.10,"EEM":0.10,"BTC-USD":0.00,"IEF":0.25,"TLT":0.15,"GLD":0.10,"BIL":0.10},
    "risk_off": {"SPY":0.07,"QQQ":0.03,"EEM":0.05,"BTC-USD":0.00,"IEF":0.35,"TLT":0.25,"GLD":0.15,"BIL":0.10},
}
THRESH_ON, THRESH_OFF = 2, -2              # Total 점수 임계치
DELTA_K = {"ME":3, "2W-FRI":3, "W-FRI":3}  # 빈도별 Δ창

# ========== 날짜/Series 유틸 ==========
def end_date() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()

def start_date(years: int = START_YEARS) -> pd.Timestamp:
    return end_date() - relativedelta(years=years, days=7)

def to_series_1d(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x["Close"] if "Close" in x.columns else x.iloc[:, 0]
        s = s.squeeze()
    else:
        arr = np.asarray(x)
        arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D, got shape {arr.shape}")
        s = pd.Series(arr, name=name)
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[~s.index.isna()]
        except Exception:
            pass
    return s

def eop_resample(s: pd.Series, rule: str) -> pd.Series:
    if rule == "M":
        rule = "ME"     # pandas 경고 회피
    s = to_series_1d(s).dropna()
    if s.empty:
        return s
    return s.resample(rule).last().dropna()

def fred_series(code: str) -> pd.Series:
    s = fred.get_series_latest_release(code)
    s = to_series_1d(s)
    return s.astype(float)

def yf_close(ticker: str, tries: int = 2, sleep_sec: float = 0.8) -> pd.Series:
    for _ in range(tries):
        try:
            df = yf.download(
                ticker,
                start=start_date(),
                end=end_date(),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s[~s.index.isna()]
                if len(s) >= 20:
                    return s
        except Exception:
            pass
        time.sleep(sleep_sec)
    return pd.Series(dtype=float, name=ticker)

def first_nonempty_yf(tickers) -> pd.Series:
    for t in tickers:
        s = yf_close(t)
        if s is not None and isinstance(s, pd.Series) and not s.empty:
            return s
    return pd.Series(dtype=float)

def ann_stats(nav: pd.Series, ppy: int) -> dict:
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"CAGR":np.nan,"VOL":np.nan,"MDD":np.nan,"Sharpe":np.nan}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    CAGR = (nav.iloc[-1] / nav.iloc[0])**(1/years) - 1 if years > 0 else np.nan
    VOL  = rets.std() * np.sqrt(ppy)
    MDD  = float((1 - nav / nav.cummax()).max())
    Sharpe = (CAGR - RFR_ASSUMPTION) / VOL if VOL > 0 else np.nan
    return {"CAGR":CAGR, "VOL":VOL, "MDD":MDD, "Sharpe":Sharpe}

# ---- XIRR (날짜기반 IRR) ----
def xnpv(rate: float, cash_flows: np.ndarray, dates: list[pd.Timestamp]) -> float:
    t0 = dates[0]
    return float(sum(cf / (1 + rate) ** ((d - t0).days / 365.25) for cf, d in zip(cash_flows, dates)))

def xirr(cash_flows: np.ndarray, dates: list[pd.Timestamp]) -> float:
    if not (np.any(cash_flows < 0) and np.any(cash_flows > 0)):
        return np.nan
    lo, hi = -0.99, 5.0
    f_lo, f_hi = xnpv(lo, cash_flows, dates), xnpv(hi, cash_flows, dates)
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = xnpv(mid, cash_flows, dates)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return mid

# ========== 데이터 로더 ==========
def load_price_panel() -> pd.DataFrame:
    frames, lengths = [], {}
    for t in ALL_TICKERS:
        s = yf_close(t)
        s = to_series_1d(s, name=t)
        lengths[t] = len(s)
        if len(s) > 0:
            s = s.copy(); s.name = t
            frames.append(s)
    print("Price lengths:", lengths)
    if not frames:
        raise RuntimeError("No price data fetched from Yahoo.")
    px = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    px = px.loc[:, px.count() > 10]
    if px.empty:
        raise RuntimeError("All price series too short/empty after filtering.")
    return px

def build_signals_base():
    dgs10 = fred_series("DGS10")
    dgs2  = fred_series("DGS2")
    dfii10= fred_series("DFII10")
    walcl = fred_series("WALCL")
    hyoas = fred_series("BAMLH0A0HYM2")
    dxy   = first_nonempty_yf(["DX-Y.NYB", "DX=F"])
    vix   = yf_close("^VIX")

    st, ed = start_date(), end_date()
    def clip(s):
        s = to_series_1d(s).dropna()
        return s[(s.index >= st) & (s.index <= ed)]

    return {
        "DGS10": clip(dgs10),
        "DGS2": clip(dgs2),
        "DFII10": clip(dfii10),
        "WALCL": clip(walcl),
        "HYOAS": clip(hyoas),
        "DXY": clip(dxy),
        "VIX": clip(vix),
    }

def regime_series(signals_daily: dict, freq: str) -> pd.DataFrame:
    freq = "ME" if freq == "M" else freq
    k = DELTA_K.get(freq, 3)

    DGS10 = eop_resample(signals_daily["DGS10"],  freq)
    DGS2  = eop_resample(signals_daily["DGS2"],   freq)
    DFII10= eop_resample(signals_daily["DFII10"], freq)
    DXY   = eop_resample(signals_daily["DXY"],    freq)
    VIX   = eop_resample(signals_daily["VIX"],    freq)
    WALCL = eop_resample(signals_daily["WALCL"],  freq)
    HYOAS = eop_resample(signals_daily["HYOAS"],  freq)

    idx = DGS10.index
    for s in [DGS2, DFII10, DXY, VIX, WALCL, HYOAS]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()

    curve = (DGS10 - DGS2).reindex(idx)
    real  = DFII10.reindex(idx)
    dxy_s = DXY.reindex(idx)
    vix_s = VIX.reindex(idx)
    hy_s  = HYOAS.reindex(idx)
    wal_s = WALCL.reindex(idx)

    real_dk  = real  - real.shift(k)
    dxy_dk   = dxy_s - dxy_s.shift(k)
    hyoas_dk = hy_s  - hy_s.shift(k)
    walcl_dk = wal_s - wal_s.shift(k)

    score = pd.DataFrame(index=idx)
    score["real_dk"]  = np.where(real_dk  < 0, +1, -1)
    score["dxy_dk"]   = np.where(dxy_dk   < 0, +1, -1)
    score["curve_lv"] = np.where(curve    > 0, +1, -1)
    score["vix_bin"]  = np.select([vix_s < 20, vix_s > 25], [1, -1], default=0)
    score["hyoas_dk"] = np.where(hyoas_dk < 0, +1, -1)
    score["walcl_dk"] = np.where(walcl_dk > 0, +1, -1)

    score = score.dropna()
    score["Total"] = score.sum(axis=1)
    score["Regime"] = np.where(
        score["Total"] >= THRESH_ON,  "risk_on",
        np.where(score["Total"] <= THRESH_OFF, "risk_off", "neutral")
    )
    return score[["Total", "Regime"]]

def price_panel_resampled(px_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq = "ME" if freq == "M" else freq
    d = {}
    for c in px_daily.columns:
        s = eop_resample(px_daily[c], freq)
        if not s.empty:
            d[c] = s
    if not d:
        return pd.DataFrame()
    return pd.DataFrame(d).dropna(how="all")

# ========== 달력 월말 ==========
def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(index) == 0:
        return []
    df = pd.DataFrame({"d": index})
    df["ym"] = df["d"].dt.to_period("M")
    last = df.groupby("ym", sort=True)["d"].max()
    return list(last.values)

def calendar_month_ends(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return month_end_dates(dates)

# ========== 백테스트(전략) ==========
def backtest_one(px_daily: pd.DataFrame, regime_df: pd.DataFrame, freq_label: str,
                 tc_percent: float, initial_capital: float, monthly_contribution: float, ppy: int):
    # 1) 가격 리샘플
    px = price_panel_resampled(px_daily, freq_label)
    if px is None or px.empty:
        raise RuntimeError(f"No resampled prices for freq={freq_label}")

    # 2) union 정렬 + ffill/bfill (레짐/가격 간 날짜 어긋남 흡수)
    idx_union = px.index.union(regime_df.index).sort_values()
    px = px.reindex(idx_union).ffill()
    regime = regime_df.reindex(idx_union)
    regime = regime.ffill().bfill().infer_objects(copy=False)
    if "Regime" not in regime.columns:
        raise RuntimeError("Regime column missing in regime dataframe.")
    regime["Regime"] = regime["Regime"].ffill().bfill().astype("string")

    # 3) 전부 NaN 가격행 제거 & 최소 가용 종목 수 필터(50%)
    px = px.dropna(how="all")
    min_cols = max(2, int(0.5 * px.shape[1]))
    px = px[px.count(axis=1) >= min_cols]
    if px.shape[0] < 2:
        raise RuntimeError("Not enough price rows after filtering.")
    regime = regime.reindex(px.index).ffill().bfill()

    # 4) 자산별 수익률 (첫 행 0%)
    rets_asset = px.pct_change()
    if rets_asset.shape[0] >= 1:
        rets_asset.iloc[0] = 0.0
    rets_asset = rets_asset.dropna(how="all")
    dates = rets_asset.index
    if len(dates) < 2:
        raise RuntimeError("Not enough periods after pct_change.")

    # 5) 월말 납입 날짜(해당 freq 인덱스 기준)
    contrib_dates = set(month_end_dates(dates))

    # 6) 거래비용 반영 포트 수익률
    twr_rets = pd.Series(index=dates, dtype=float)
    last_w = pd.Series(0.0, index=px.columns)
    equity = initial_capital

    cf_vals  = [-initial_capital]
    cf_dates = [dates[0]]

    for i, t in enumerate(dates):
        if t in contrib_dates and i > 0 and monthly_contribution != 0.0:
            equity += monthly_contribution
            cf_vals.append(-monthly_contribution)
            cf_dates.append(t)

        rtype = regime.at[t, "Regime"]
        if pd.isna(rtype) or rtype not in WEIGHTS:
            rtype = "neutral"

        target_w = pd.Series(WEIGHTS[rtype], index=px.columns).fillna(0.0)
        turnover = (target_w - last_w).abs().sum()
        tc_rate  = tc_percent / 100.0
        tc       = turnover * tc_rate

        pr     = float((rets_asset.loc[t] * target_w).sum())
        pr_net = pr - tc
        twr_rets.loc[t] = pr_net

        equity *= (1 + pr_net)
        last_w = target_w

    nav_twr = (1.0 + twr_rets).cumprod()

    eq_dca = pd.Series(index=dates, dtype=float)
    equity_re = initial_capital
    for i, t in enumerate(dates):
        if t in contrib_dates and i > 0 and monthly_contribution != 0.0:
            equity_re += monthly_contribution
        equity_re *= (1 + twr_rets.loc[t])
        eq_dca.iloc[i] = equity_re

    cf_vals.append(equity)
    cf_dates.append(dates[-1])
    mwr_xirr = xirr(np.array(cf_vals, dtype=float), list(cf_dates))

    return nav_twr.dropna(), eq_dca.dropna(), twr_rets.dropna(), mwr_xirr

# ========== 백테스트(벤치마크: 단일자산, 빈도 무관) ==========
def backtest_benchmark(px_daily: pd.DataFrame, ticker: str,
                       tc_percent: float, initial_capital: float,
                       monthly_contribution: float):
    """
    단일자산 벤치마크:
      - 일간 수익률 기반(TWR) → 빈도 독립 (항상 한 줄 결과, Freq='ALL')
      - 납입은 '달력 월말'에만 반영(DCA)
      - 거래비용: 최초 진입 1회만 반영(이후 turnover≈0)
    """
    if ticker not in px_daily.columns:
        raise RuntimeError(f"Benchmark {ticker}: price not in panel")
    s = to_series_1d(px_daily[ticker]).dropna()
    if len(s) < 2:
        raise RuntimeError(f"Benchmark {ticker}: not enough daily data")

    rets = s.pct_change()
    rets.iloc[0] = 0.0
    rets = rets.dropna(how="all")
    dates = rets.index

    contrib_dates = set(calendar_month_ends(dates))

    tc_rate = tc_percent / 100.0
    rets.iloc[0] = rets.iloc[0] - tc_rate  # 최초 진입 비용 차감

    nav_twr = (1.0 + rets).cumprod()

    eq = pd.Series(index=dates, dtype=float)
    equity = initial_capital
    cf_vals  = [-initial_capital]
    cf_dates = [dates[0]]

    for i, t in enumerate(dates):
        if t in contrib_dates and i > 0 and monthly_contribution != 0.0:
            equity += monthly_contribution
            cf_vals.append(-monthly_contribution)
            cf_dates.append(t)
        equity *= (1 + rets.loc[t])
        eq.iloc[i] = equity

    cf_vals.append(equity)
    cf_dates.append(dates[-1])
    mwr_xirr = xirr(np.array(cf_vals, dtype=float), list(cf_dates))

    return nav_twr.dropna(), eq.dropna(), rets.dropna(), mwr_xirr

# ========== 실행 ==========
def run_all(tc_percent: float, initial_capital: float, monthly_contribution: float):
    # 원천
    signals_daily = build_signals_base()
    px_daily = load_price_panel()

    results_nav = []
    results_eq  = []
    perf_rows   = []

    ppy_map = {"ME": 12, "2W-FRI": 26, "W-FRI": 52}

    # 1) 전략(Regime) — 3개 빈도
    for freq in ["ME", "2W-FRI", "W-FRI"]:
        print(f"[run] Strategy freq={freq} | tc={tc_percent}%, init={initial_capital}, monthly_contrib={monthly_contribution}")
        reg = regime_series(signals_daily, freq=freq)
        nav, eq, twr_rets, xirr_annual = backtest_one(
            px_daily, reg, freq_label=freq,
            tc_percent=tc_percent,
            initial_capital=initial_capital,
            monthly_contribution=monthly_contribution,
            ppy=ppy_map[freq]
        )
        twr_nav = (1.0 + twr_rets).cumprod()
        twr_nav.index = nav.index
        stats = ann_stats(twr_nav, ppy=ppy_map[freq])

        n_months = len(set(eq.index.to_period("M")))
        total_contrib = initial_capital + monthly_contribution * max(0, n_months - 1)

        results_nav.append(pd.DataFrame({"Strategy":"Regime", "Freq":freq, "NAV":nav}))
        results_eq.append(pd.DataFrame({"Strategy":"Regime", "Freq":freq, "Equity":eq}))

        perf_rows.append({
            "Strategy":"Regime",
            "Freq": freq,
            "TWR_CAGR": round(stats["CAGR"], 6),
            "TWR_VOL":  round(stats["VOL"],  6),
            "TWR_MDD":  round(stats["MDD"],  6),
            "TWR_Sharpe": round(stats["Sharpe"], 6),
            "MWR_XIRR":   round(xirr_annual,   6),
            "Final_Equity": round(eq.iloc[-1], 2),
            "Total_Contrib": round(total_contrib, 2)
        })

    # 2) 벤치마크(단일자산) — 일간 기반(빈도 무관, Freq='ALL')
    for tk in BENCH_TICKERS:
        print(f"[run] Benchmark {tk} (daily base, freq-agnostic)")
        nav_b, eq_b, rets_b, xirr_b = backtest_benchmark(
            px_daily, ticker=tk,
            tc_percent=tc_percent,
            initial_capital=initial_capital,
            monthly_contribution=monthly_contribution,
        )
        stats_b = ann_stats((1.0 + rets_b).cumprod(), ppy=252)

        n_months = len(set(eq_b.index.to_period("M")))
        total_contrib = initial_capital + monthly_contribution * max(0, n_months - 1)

        results_nav.append(pd.DataFrame({"Strategy":f"Bench-{tk}", "Freq":"ALL", "NAV":nav_b}))
        results_eq.append(pd.DataFrame({"Strategy":f"Bench-{tk}", "Freq":"ALL", "Equity":eq_b}))

        perf_rows.append({
            "Strategy": f"Bench-{tk}",
            "Freq": "ALL",
            "TWR_CAGR": round(stats_b["CAGR"], 6),
            "TWR_VOL":  round(stats_b["VOL"],  6),
            "TWR_MDD":  round(stats_b["MDD"],  6),
            "TWR_Sharpe": round(stats_b["Sharpe"], 6),
            "MWR_XIRR":   round(xirr_b, 6),
            "Final_Equity": round(eq_b.iloc[-1], 2),
            "Total_Contrib": round(total_contrib, 2)
        })

    # 저장
    outdir = Path("data_pro"); outdir.mkdir(parents=True, exist_ok=True)

    nav_df = pd.concat(results_nav, axis=0)
    eq_df  = pd.concat(results_eq, axis=0)
    perf   = pd.DataFrame(perf_rows).sort_values(["Strategy","Freq"])

    # 숫자 표기: 지수형 방지(정수화)
    perf["Final_Equity"]  = perf["Final_Equity"].round(0).astype("int64")
    perf["Total_Contrib"] = perf["Total_Contrib"].round(0).astype("int64")

    # 길이형 저장
    nav_df.set_index(["Strategy","Freq"], inplace=False).to_csv(outdir / "strategy_and_bench_nav_long.csv", index=False)
    eq_df.set_index(["Strategy","Freq"], inplace=False).to_csv(outdir  / "strategy_and_bench_equity_long.csv", index=False)
    perf.to_csv(outdir / "strategy_and_bench_perf.csv", index=False)

    # 보기좋게 출력(천단위 콤마)
    print("\n=== Performance (TWR) & MWR(XIRR with monthly contributions) — Strategy + Benchmarks ===")
    print(perf.to_string(index=False, formatters={
        "Final_Equity":  lambda v: f"{v:,}",
        "Total_Contrib": lambda v: f"{v:,}"
    }))
    print(f"\nSaved: {outdir/'strategy_and_bench_nav_long.csv'}")
    print(f"Saved: {outdir/'strategy_and_bench_equity_long.csv'}")
    print(f"Saved: {outdir/'strategy_and_bench_perf.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc", type=float, default=0.2,
                        help="거래비용(%) — turnover * tc%, 예: 0.2 → 0.2%")
    parser.add_argument("--init", type=float, default=1_000_000.0,
                        help="초기 자본 (화폐단위)")
    parser.add_argument("--contrib_month", type=float, default=300_000.0,
                        help="매월 고정 납입 금액 (0이면 납입 없음)")
    args = parser.parse_args()

    run_all(tc_percent=args.tc, initial_capital=args.init, monthly_contribution=args.contrib_month)
