# -*- coding: utf-8 -*-
"""
strategy_backtest_multi_freq.py (tc & DCA enabled)
- FRED + yfinance 신호 시계열 생성
- 월/격주/주 리밸런싱 레짐 전략 백테스트
- 8자산(4 Risk-on, 4 Defensive)
- 거래비용을 파라미터로(예: --tc 0.2 = 0.2%); DCA(고정 납입) 반영
- 산출: 빈도별 NAV(TWR 기준), DCA Equity, TWR 성과, MWR(IRR)
"""

import os, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dateutil.relativedelta import relativedelta

import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# ========== 기본 설정 ==========
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing in environment/.env")

fred = Fred(api_key=FRED_KEY)
START_YEARS = 10
RFR_ASSUMPTION = 0.02  # Sharpe용 무위험 가정

# 유니버스(8개, 4:4)
TICKERS_RISKON    = ["SPY", "QQQ", "EEM", "BTC-USD"]
TICKERS_DEFENSIVE = ["IEF", "TLT", "GLD", "BIL"]
ALL_TICKERS = TICKERS_RISKON + TICKERS_DEFENSIVE

# 레짐별 타겟 가중(합=1) — 필요시 조정 가능
WEIGHTS = {
    "risk_on":  {"SPY":0.30,"QQQ":0.20,"EEM":0.15,"BTC-USD":0.10,"IEF":0.10,"TLT":0.05,"GLD":0.05,"BIL":0.05},
    "neutral":  {"SPY":0.20,"QQQ":0.10,"EEM":0.10,"BTC-USD":0.00,"IEF":0.25,"TLT":0.15,"GLD":0.10,"BIL":0.10},
    "risk_off": {"SPY":0.07,"QQQ":0.03,"EEM":0.05,"BTC-USD":0.00,"IEF":0.35,"TLT":0.25,"GLD":0.15,"BIL":0.10},
}
THRESH_ON, THRESH_OFF = 2, -2

# 빈도별 Δ창(기간 수) — 필요 시 조정 가능 (민감도)
DELTA_K = {"ME":3, "2W-FRI":3, "W-FRI":3}

# ========== 유틸 ==========
def end_date():
    return pd.Timestamp.today().normalize()

def start_date(years=START_YEARS):
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
    # datetime index 보정
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s[~s.index.isna()]
        except Exception:
            pass
    return s

def eop_resample(s: pd.Series, rule: str) -> pd.Series:
    # pandas 'M' 경고 회피 → 'ME'(month-end) 사용
    if rule == "M":
        rule = "ME"
    s = to_series_1d(s).dropna()
    if s.empty: return s
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

def ann_stats(nav: pd.Series, ppy: int) -> dict:
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"CAGR":np.nan,"VOL":np.nan,"MDD":np.nan,"Sharpe":np.nan}
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    CAGR = (nav.iloc[-1] / nav.iloc[0])**(1/years) - 1 if years>0 else np.nan
    VOL = rets.std() * np.sqrt(ppy)
    MDD = float((1 - nav / nav.cummax()).max())
    Sharpe = (CAGR - RFR_ASSUMPTION) / VOL if VOL>0 else np.nan
    return {"CAGR":CAGR,"VOL":VOL,"MDD":MDD,"Sharpe":Sharpe}

def np_irr_equal_period(cash_flows: np.ndarray) -> float:
    """동일 간격 CF의 IRR (per-period). numpy.irr 대체 (단순 이분법)"""
    # CF가 모두 양/음이면 IRR 정의 안됨
    if not (np.any(cash_flows < 0) and np.any(cash_flows > 0)):
        return np.nan

    def npv(rate):
        return np.sum(cash_flows / ((1+rate) ** np.arange(len(cash_flows))))

    lo, hi = -0.99, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        v = npv(mid)
        if abs(v) < 1e-10: return mid
        # 부호 기준 이분
        if npv(lo) * v < 0:
            hi = mid
        else:
            lo = mid
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
    px = pd.concat(frames, axis=1).sort_index()
    px = px.dropna(how="all")
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
    dxy   = yf_close("DX-Y.NYB")
    if dxy.empty: dxy = yf_close("DX=F")
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
    score["Regime"] = np.where(score["Total"]>=THRESH_ON, "risk_on",
                        np.where(score["Total"]<=THRESH_OFF, "risk_off","neutral"))
    return score[["Total","Regime"]]

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

# ========== 백테스트(거래비용 & DCA 반영) ==========
def backtest_one(px_daily: pd.DataFrame, regime_df: pd.DataFrame, freq_label: str,
                 tc_percent: float,
                 initial_capital: float,
                 contribution_per_period: float,
                 ppy: int):
    """
    - tc_percent: 거래비용(%) (예: 0.2 → 0.2%); turnover * tc_percent 차감
    - initial_capital: 초기 자본 (예: 1_000_000)
    - contribution_per_period: 리밸런싱 주기마다 고정 납입 금액
    - 리밸런싱 시점에 '납입 → 타겟가중으로 즉시 재배분' 순서
    반환:
      nav_twr: TWR 기준 NAV (초기 1.0, 거래비용 반영)
      eq_dca: 납입 누적을 포함한 실제 포트 가치(화폐단위)
      twr_rets: 기간 수익률(거래비용 반영, 납입과 무관)
      mwr_irr_annual: 납입/최종가치 기준 IRR(연율)
    """
    px = price_panel_resampled(px_daily, freq_label)
    if px.empty:
        raise RuntimeError(f"No resampled prices for freq={freq_label}")

    idx = px.index.intersection(regime_df.index)
    px = px.reindex(idx).dropna()
    regime = regime_df.reindex(idx).ffill().dropna()

    rets_asset = px.pct_change().dropna()
    dates = rets_asset.index

    # 결과 컨테이너
    nav_twr = pd.Series(index=dates, dtype=float)  # 1.0 시작
    eq_dca  = pd.Series(index=dates, dtype=float)  # 화폐단위(초기/납입 반영)
    twr_rets= pd.Series(index=dates, dtype=float)  # 기간 수익률(비중/거래비용 반영)

    # 초기 포트
    equity = initial_capital
    last_w = pd.Series(0.0, index=px.columns)
    nav_twr.iloc[0] = 1.0
    eq_dca.iloc[0]  = initial_capital

    # IRR 계산용 캐시플로우 (동일 간격 가정)
    cash_flows = []  # 음수=투입, 양수=회수
    cash_flows.append(-initial_capital)  # t0

    for i, t in enumerate(dates):
        # 1) 납입 (리밸런싱 주기마다 고정)
        if i > 0 and contribution_per_period != 0.0:
            equity += contribution_per_period
            cash_flows.append(-contribution_per_period)

        # 2) 타겟 가중 계산
        rtype = regime.loc[t, "Regime"]
        target_w = pd.Series(WEIGHTS[rtype]).reindex(px.columns).fillna(0.0)

        # 3) 거래비용 (turnover * tc_rate)
        turnover = (target_w - last_w).abs().sum()
        tc_rate = tc_percent / 100.0  # 0.2% → 0.002
        tc = turnover * tc_rate

        # 4) TWR용 포트 수익률 (거래비용 차감)
        pr = float((rets_asset.loc[t] * target_w).sum())  # 포트 수익률(전)
        pr_net = pr - tc                                  # 거래비용 차감
        twr_rets.loc[t] = pr_net

        # 5) NAV(TWR) 업데이트 (초기 1.0 기준 누적)
        if i == 0:
            nav_twr.iloc[i] = 1.0
        else:
            nav_twr.iloc[i] = nav_twr.iloc[i-1] * (1 + pr_net)

        # 6) DCA Equity 업데이트 (실제 자본: 납입 반영 후 수익률 적용)
        equity *= (1 + pr_net)
        eq_dca.iloc[i] = equity

        last_w = target_w

    # IRR(MWR): 동일간격 IRR → 연율화
    cash_flows.append(equity)  # 마지막 회수
    cf = np.array(cash_flows, dtype=float)
    irr_per = np_irr_equal_period(cf)
    mwr_irr_annual = (1 + irr_per) ** ppy - 1 if not np.isnan(irr_per) else np.nan

    return nav_twr.dropna(), eq_dca.dropna(), twr_rets.dropna(), mwr_irr_annual

def run_all(tc_percent: float, initial_capital: float, contribution: float):
    # 원천
    signals_daily = build_signals_base()
    px_daily = load_price_panel()

    results_nav = {}
    results_eq  = {}
    perf_rows   = []

    # 빈도별 연간 기간수 (annualization)
    ppy_map = {"ME": 12, "2W-FRI": 26, "W-FRI": 52}

    for freq in ["ME", "2W-FRI", "W-FRI"]:
        print(f"[run] regime & backtest for freq={freq} | tc={tc_percent}%, init={initial_capital}, contrib/period={contribution}")
        reg = regime_series(signals_daily, freq=freq)
        nav, eq, twr_rets, irr_annual = backtest_one(
            px_daily, reg, freq_label=freq,
            tc_percent=tc_percent,
            initial_capital=initial_capital,
            contribution_per_period=contribution,
            ppy=ppy_map[freq]
        )
        results_nav[freq] = nav
        results_eq[freq]  = eq

        # TWR 성과(전략 자체)
        twr_nav = (1 + twr_rets).cumprod()
        twr_nav.index = nav.index  # 정렬
        stats = ann_stats(twr_nav, ppy=ppy_map[freq])

        perf_rows.append({
            "Freq": freq,
            "TWR_CAGR": round(stats["CAGR"], 6),
            "TWR_VOL":  round(stats["VOL"],  6),
            "TWR_MDD":  round(stats["MDD"],  6),
            "TWR_Sharpe": round(stats["Sharpe"], 6),
            "MWR_IRR":   round(irr_annual,  6),
            "Final_Equity": round(eq.iloc[-1], 2),
            "Total_Contrib": round(initial_capital + contribution * (len(eq)-1), 2)
        })

    # 저장
    outdir = Path("data_pro"); outdir.mkdir(parents=True, exist_ok=True)
    nav_df = pd.DataFrame({k:v for k,v in results_nav.items()}).dropna(how="all")
    eq_df  = pd.DataFrame({k:v for k,v in results_eq.items()}).dropna(how="all")
    perf   = pd.DataFrame(perf_rows).set_index("Freq")

    nav_df.to_csv(outdir/"strategy_nav_by_freq.csv")       # TWR NAV (=기존과 호환)
    eq_df.to_csv(outdir/"strategy_equity_dca_by_freq.csv") # 납입 반영 실제 포트 가치
    perf.to_csv(outdir/"strategy_perf_by_freq_tc_dca.csv")

    print("\n=== Performance (TWR) & MWR(IRR with contributions) ===")
    print(perf.to_string())
    print(f"\nSaved: {outdir/'strategy_nav_by_freq.csv'}")
    print(f"Saved: {outdir/'strategy_equity_dca_by_freq.csv'}")
    print(f"Saved: {outdir/'strategy_perf_by_freq_tc_dca.csv'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc", type=float, default=0.2,
                        help="거래비용(%) — turnover * tc%, 예: 0.2 → 0.2%")
    parser.add_argument("--init", type=float, default=1_000_000.0,
                        help="초기 자본 (화폐단위)")
    parser.add_argument("--contrib", type=float, default=0.0,
                        help="리밸런싱 주기당 고정 납입 금액 (freq마다 동일 금액)")
    args = parser.parse_args()

    run_all(tc_percent=args.tc, initial_capital=args.init, contribution=args.contrib)
