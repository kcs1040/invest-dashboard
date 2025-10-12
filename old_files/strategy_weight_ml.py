# -*- coding: utf-8 -*-
"""
strategy_weight_ml.py (Regime-aware ML weights, BIL<=25%)

- Universe(6): SPY, IEF, BIL, EEM, GLD, QQQ
- Regime: rule-based (existing macro rules), freq ∈ {ME, 2W-FRI, W-FRI}
- Learning:
  * μ (expected return): blend of RidgeCV(momentum features) & historical mean
  * Σ (cov): EWMA covariance (recent info)
  * Weights: constrained Sharpe maximization via gradient ascent
    - long-only, sum=1
    - caps by regime (with global BIL cap 25%)
    - min equity share per regime (SPY+QQQ+EEM)
    - cash penalty (lambda_cash) to avoid cash crowding
- Trading:
  * Rebalance by freq, monthly DCA at calendar month-ends
  * Turnover * transaction_cost% deducted each rebalance
- Output: learned weights per regime, NAV/Equity time-series, performance table

Run:
(.venv) ➜ python strategy_weight_ml.py --freq ME --tc 0.2 --init 1000000 --contrib_month 300000 --lookback_min 36
"""

import os, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from dateutil.relativedelta import relativedelta
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# ---------- Optional: scikit-learn (for RidgeCV).
# If not installed, code will fallback to mean-only μ.
try:
    from sklearn.linear_model import RidgeCV
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ===== Config =====
ASSETS6 = ["SPY","IEF","BIL","EEM","GLD","QQQ"]
EQUITY_NAMES = ["SPY","QQQ","EEM"]

RFR_ASSUMPTION = 0.02
START_YEARS = 10

# === Global hard cap for BIL (short-term T-bill) across ALL regimes ===
GLOBAL_BIL_CAP = 0.25  # 25%

# Regime-specific caps (will be intersected with GLOBAL_BIL_CAP)
CAPS_BY_REGIME = {
    "risk_on":  {"SPY":0.45,"QQQ":0.35,"EEM":0.25,"GLD":0.25,"IEF":0.50,"BIL":0.30},
    "neutral":  {"SPY":0.35,"QQQ":0.25,"EEM":0.20,"GLD":0.30,"IEF":0.60,"BIL":0.50},
    "risk_off": {"SPY":0.20,"QQQ":0.15,"EEM":0.10,"GLD":0.35,"IEF":0.70,"BIL":0.80},
}
# Minimum equity fraction per regime (sum of SPY+QQQ+EEM)
MIN_EQUITY = {
    "risk_on": 0.40,
    "neutral": 0.20,
    "risk_off": 0.00,
}
# Cash penalty to avoid cash crowding (applies on BIL weight)
LAMBDA_CASH = {
    "risk_on": 0.50,
    "neutral": 0.25,
    "risk_off": 0.10,
}

# ===== Env =====
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing in .env")
fred = Fred(api_key=FRED_KEY)

# ===== Dates =====
def end_date(): return pd.Timestamp.today().normalize()
def start_date(years=START_YEARS): return end_date() - relativedelta(years=years, days=7)

# ===== Helpers =====
def to_series_1d(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x["Close"] if "Close" in x.columns else x.iloc[:,0]
        s = s.squeeze()
    else:
        arr = np.asarray(x).squeeze()
        if arr.ndim != 1: raise ValueError(f"Expected 1D, got {arr.shape}")
        s = pd.Series(arr, name=name)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()]
    return s

def yf_close(ticker: str, tries=2, sleep_sec=0.8) -> pd.Series:
    for _ in range(tries):
        try:
            df = yf.download(ticker, start=start_date(), end=end_date(),
                             interval="1d", progress=False, auto_adjust=True)
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

def fred_series(code: str) -> pd.Series:
    s = fred.get_series_latest_release(code)
    return to_series_1d(s).astype(float)

def eop_resample(s: pd.Series, rule: str) -> pd.Series:
    if rule == "M": rule = "ME"
    s = to_series_1d(s).dropna()
    if s.empty: return s
    return s.resample(rule).last().dropna()

def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(index)==0: return []
    df = pd.DataFrame({"d": index})
    df["ym"] = df["d"].dt.to_period("M")
    last = df.groupby("ym", sort=True)["d"].max()
    return list(last.values)

def ann_stats(nav: pd.Series, ppy: int) -> dict:
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    if rets.empty: return {"CAGR":np.nan,"VOL":np.nan,"MDD":np.nan,"Sharpe":np.nan}
    years = (nav.index[-1]-nav.index[0]).days/365.25
    CAGR = (nav.iloc[-1]/nav.iloc[0])**(1/years)-1 if years>0 else np.nan
    VOL  = rets.std()*np.sqrt(ppy)
    MDD  = float((1 - nav/nav.cummax()).max())
    Sharpe = (CAGR - RFR_ASSUMPTION)/VOL if VOL>0 else np.nan
    return {"CAGR":CAGR,"VOL":VOL,"MDD":MDD,"Sharpe":Sharpe}

def xnpv(rate, cfs, dates):
    t0 = dates[0]; return float(sum(cf/((1+rate)**((d-t0).days/365.25)) for cf,d in zip(cfs,dates)))
def xirr(cfs, dates):
    if not (np.any(cfs<0) and np.any(cfs>0)): return np.nan
    lo, hi = -0.99, 5.0
    f_lo, f_hi = xnpv(lo,cfs,dates), xnpv(hi,cfs,dates)
    if f_lo*f_hi>0: return np.nan
    for _ in range(200):
        mid = (lo+hi)/2; f_mid = xnpv(mid,cfs,dates)
        if abs(f_mid)<1e-10: return mid
        if f_lo*f_mid<0: hi,f_hi=mid,f_mid
        else: lo,f_lo=mid,f_mid
    return mid

# ===== Data =====
def load_price_panel_6() -> pd.DataFrame:
    frames, lengths = [], {}
    for t in ASSETS6:
        s = yf_close(t); s = to_series_1d(s, name=t); lengths[t]=len(s)
        if len(s)>0: frames.append(s.rename(t))
    print("Price lengths (6):", lengths)
    if not frames: raise RuntimeError("No price data.")
    px = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    px = px.loc[:, px.count()>10]
    if px.empty: raise RuntimeError("All six price series empty/too short.")
    return px

# ===== Regime (rule-based) =====
THRESH_ON, THRESH_OFF = 2, -2
DELTA_K = {"ME":3, "2W-FRI":3, "W-FRI":3}

def build_signals_base():
    dgs10 = fred_series("DGS10")
    dgs2  = fred_series("DGS2")
    dfii10= fred_series("DFII10")
    walcl = fred_series("WALCL")
    hyoas = fred_series("BAMLH0A0HYM2")
    # dxy: 야후 선물/인덱스 폴백
    dxy = pd.Series(dtype=float)
    for t in ["DX-Y.NYB", "DX=F"]:
        s = yf_close(t)
        if s is not None and len(s)>0: dxy=s; break
    vix  = yf_close("^VIX")

    st, ed = start_date(), end_date()
    def clip(s):
        s = to_series_1d(s).dropna()
        return s[(s.index>=st)&(s.index<=ed)]

    return {
        "DGS10": clip(dgs10), "DGS2": clip(dgs2), "DFII10": clip(dfii10),
        "WALCL": clip(walcl), "HYOAS": clip(hyoas),
        "DXY": clip(dxy),     "VIX": clip(vix)
    }

def regime_series(signals_daily: dict, freq: str) -> pd.DataFrame:
    freq = "ME" if freq=="M" else freq
    k = DELTA_K.get(freq,3)
    def R(s): return eop_resample(signals_daily[s],freq)
    DGS10, DGS2, DFII10 = R("DGS10"), R("DGS2"), R("DFII10")
    DXY, VIX, WALCL, HYOAS = R("DXY"), R("VIX"), R("WALCL"), R("HYOAS")

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
                        np.where(score["Total"]<=THRESH_OFF, "risk_off", "neutral"))
    return score[["Total","Regime"]]

# ===== Resample & returns =====
def resample_prices(px_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq = "ME" if freq=="M" else freq
    d={}
    for c in px_daily.columns:
        s = eop_resample(px_daily[c], freq)
        if not s.empty: d[c]=s
    if not d: return pd.DataFrame()
    df = pd.DataFrame(d).dropna(how="all")
    min_cols = max(2, int(0.5 * df.shape[1]))
    return df[df.count(axis=1) >= min_cols]

def returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    rets = px.pct_change()
    if len(rets)>=1: rets.iloc[0] = 0.0
    return rets.dropna(how="all")

# ===== μ, Σ estimation (ML + EWMA) =====
def ewma_cov(returns: pd.DataFrame, halflife: int = 36) -> np.ndarray:
    r = returns.fillna(0.0).copy()
    vol = r.ewm(halflife=halflife, min_periods=max(12, halflife//2)).std().iloc[-1].replace(0, np.nan)
    vol = vol.fillna(vol.median() if vol.notna().any() else 1.0)
    z = r / vol
    w = np.exp(-np.log(2)/halflife*np.arange(len(z))[::-1])
    w = w / w.sum()
    zc = z  # mean ~0 by construction here
    C = (zc.T * w) @ zc.values
    V = np.diag(vol.values)
    cov = V @ C @ V
    return cov

def blended_mu_with_momentum(rets: pd.DataFrame, alpha: float = 0.6) -> np.ndarray:
    cols = list(rets.columns)
    mu_mean = rets.mean().values
    if not SKLEARN_OK or len(rets) < 260:  # fallback if sklearn missing or too short
        return alpha*mu_mean + (1-alpha)*mu_mean

    m1  = rets.rolling(21).mean()
    m3  = rets.rolling(63).mean()
    m6  = rets.rolling(126).mean()
    m12 = rets.rolling(252).mean()
    X = pd.concat([m1, m3, m6, m12], axis=1, keys=["m1","m3","m6","m12"]).dropna()
    if X.empty: return mu_mean
    Y = rets.loc[X.index]

    mu_mom = []
    alphas = np.logspace(-4, 2, 15)
    for c in cols:
        model = RidgeCV(alphas=alphas, fit_intercept=True)
        model.fit(X.values, Y[c].values)
        y_hat = model.predict(X.values)
        mu_mom.append(float(np.mean(y_hat)))
    mu_mom = np.array(mu_mom)

    return alpha*mu_mom + (1-alpha)*mu_mean

# ===== Constraints & optimizer =====
def project_with_caps_and_min_eq(w, caps, min_eq, col_order):
    # nonneg & sum=1
    w = np.clip(w, 0.0, None)
    s = w.sum()
    w = (w if s>0 else np.ones_like(w)/len(w)) / (s if s>0 else 1.0)

    # apply individual caps
    w = np.minimum(w, caps)
    w = w / w.sum()

    # enforce min equity sum
    idx_eq = [col_order.index(n) for n in EQUITY_NAMES if n in col_order]
    eq_sum = w[idx_eq].sum() if idx_eq else 0.0
    if eq_sum < min_eq and idx_eq:
        deficit = min_eq - eq_sum
        idx_non = [i for i,_ in enumerate(col_order) if i not in idx_eq]
        non_sum = w[idx_non].sum() if idx_non else 0.0
        if non_sum > 1e-12:
            w[idx_non] = np.clip(w[idx_non] - deficit * (w[idx_non]/non_sum), 0, None)
            w = w / w.sum()
        # if still < min_eq due to rounding, add equally across equity legs
        eq_sum = w[idx_eq].sum()
        if eq_sum < min_eq:
            add = min_eq - eq_sum
            for i in idx_eq:
                w[i] += add/len(idx_eq)
            w = np.maximum(w, 0.0); w = w / w.sum()
    return w

def regime_caps_vector(regime_name: str, cols: list[str]) -> np.ndarray:
    # regime caps intersected with GLOBAL_BIL_CAP
    caps = np.array([CAPS_BY_REGIME[regime_name].get(c, 1.0) for c in cols], dtype=float)
    if "BIL" in cols:
        j = cols.index("BIL")
        caps[j] = min(caps[j], GLOBAL_BIL_CAP)  # enforce 25% hard cap
    return caps

def learn_weights_sharpe_plus(rets: pd.DataFrame,
                              regime_name: str,
                              alpha_mom: float = 0.6,
                              halflife_cov: int = 36) -> np.ndarray:
    cols = list(rets.columns)
    mu = blended_mu_with_momentum(rets[cols], alpha=alpha_mom)
    cov = ewma_cov(rets[cols], halflife=halflife_cov)

    caps = regime_caps_vector(regime_name, cols)
    min_eq = MIN_EQUITY[regime_name]
    lam_cash = LAMBDA_CASH[regime_name]

    # init w
    w = np.ones(len(cols))/len(cols)
    w = project_with_caps_and_min_eq(w, caps, min_eq, cols)

    lr = 0.1
    eps = 1e-10
    for _ in range(800):
        den = w @ cov @ w + eps
        num = w @ mu
        grad_sharpe = (mu*np.sqrt(den) - num*(cov@w)/np.sqrt(den)) / den
        grad_penalty = np.zeros_like(w)
        if "BIL" in cols:
            grad_penalty[cols.index("BIL")] = -lam_cash  # penalize cash weight
        grad = grad_sharpe + grad_penalty
        w = w + lr*grad
        w = project_with_caps_and_min_eq(w, caps, min_eq, cols)

    return w

# ===== Backtest (fixed weights per regime) =====
def backtest_with_fixed_weights(px_daily: pd.DataFrame, regime_df: pd.DataFrame, freq: str,
                                w_risk_on: np.ndarray, w_neutral: np.ndarray, w_risk_off: np.ndarray,
                                tc_percent: float,
                                initial_capital: float,
                                monthly_contribution: float,
                                ppy: int):
    px = resample_prices(px_daily, freq)
    if px.empty: raise RuntimeError("Resampled prices empty.")
    rets = returns_from_prices(px)
    dates = rets.index

    regime = regime_df.reindex(dates).ffill().bfill()
    if "Regime" not in regime.columns: raise RuntimeError("Regime missing.")

    contrib_dates = set(month_end_dates(dates))

    twr = pd.Series(index=dates, dtype=float)
    equity = initial_capital
    cf_vals = [-initial_capital]
    cf_dates = [dates[0]]

    cols = list(px.columns)
    idx_map = [cols.index(a) if a in cols else None for a in ASSETS6]
    last_w = np.zeros(len(ASSETS6))

    def pick_w(reg):
        if reg=="risk_on": return w_risk_on
        if reg=="risk_off": return w_risk_off
        return w_neutral

    for i, t in enumerate(dates):
        if t in contrib_dates and i>0 and monthly_contribution!=0.0:
            equity += monthly_contribution
            cf_vals.append(-monthly_contribution); cf_dates.append(t)

        reg = str(regime.at[t,"Regime"])
        w_full = pick_w(reg).copy()

        # mask missing assets; renormalize
        mask = np.array([ci is not None and not np.isnan(rets.iloc[i, ci]) for ci in idx_map], dtype=bool)
        if mask.sum()==0:
            pr_net = 0.0
        else:
            w_eff = np.where(mask, w_full, 0.0)
            if w_eff.sum()<=0: 
                w_eff[:] = 0.0; 
                if "BIL" in ASSETS6: w_eff[ASSETS6.index("BIL")] = 1.0
            else:
                w_eff = w_eff / w_eff.sum()

            # transaction cost
            turnover = np.abs(w_eff - last_w).sum()
            tc = turnover * (tc_percent/100.0)

            r = 0.0
            for j, ok in enumerate(mask):
                if ok:
                    r += w_eff[j]*rets.iloc[i, idx_map[j]]
            pr_net = r - tc
            last_w = w_eff

        twr.iloc[i] = pr_net
        equity *= (1 + pr_net)

    nav = (1.0 + twr).cumprod()

    # Equity with DCA
    eq = pd.Series(index=dates, dtype=float)
    equity_re = initial_capital
    for i, t in enumerate(dates):
        if t in contrib_dates and i>0 and monthly_contribution!=0.0:
            equity_re += monthly_contribution
        equity_re *= (1 + twr.iloc[i])
        eq.iloc[i] = equity_re

    cf_vals.append(equity); cf_dates.append(dates[-1])
    xirr_annual = xirr(np.array(cf_vals, dtype=float), list(cf_dates))

    return nav.dropna(), eq.dropna(), twr.dropna(), xirr_annual

# ===== Main =====
def run(freq: str, tc_percent: float, initial_capital: float, monthly_contribution: float,
        lookback_min: int = 24):
    px_daily = load_price_panel_6()
    signals  = build_signals_base()
    reg_df   = regime_series(signals, freq=freq)
    px = resample_prices(px_daily, freq)
    rets = returns_from_prices(px)

    idx = rets.index.intersection(reg_df.index).sort_values()
    rets = rets.reindex(idx).dropna(how="all")
    reg_df = reg_df.reindex(idx).ffill().bfill()

    # Learn regime-fixed weights (improved)
    learned = {}
    for regime_name in ["risk_on","neutral","risk_off"]:
        mask = (reg_df["Regime"] == regime_name)
        rets_sub = rets.loc[mask]
        if len(rets_sub) < lookback_min:
            rets_sub = rets.tail(max(lookback_min, int(len(rets)*0.3)))
        w = learn_weights_sharpe_plus(
            rets_sub[ASSETS6].dropna(how="all"),
            regime_name=regime_name,
            alpha_mom=0.6,
            halflife_cov=36
        )
        learned[regime_name] = w

    ppy_map = {"ME":12, "2W-FRI":26, "W-FRI":52}
    nav, eq, twr, xirr_annual = backtest_with_fixed_weights(
        px_daily, reg_df, freq=freq,
        w_risk_on=learned["risk_on"],
        w_neutral=learned["neutral"],
        w_risk_off=learned["risk_off"],
        tc_percent=tc_percent,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        ppy=ppy_map[freq]
    )
    stats = ann_stats((1.0+twr).cumprod(), ppy=ppy_map[freq])

    outdir = Path("data_pro"); outdir.mkdir(parents=True, exist_ok=True)
    nav.to_csv(outdir / f"ml_nav_{freq}.csv")
    eq.to_csv(outdir  / f"ml_equity_{freq}.csv")

    wt_df = pd.DataFrame({
        "Asset": ASSETS6,
        "risk_on": learned["risk_on"],
        "neutral": learned["neutral"],
        "risk_off": learned["risk_off"]
    })
    wt_df.to_csv(outdir / f"ml_weights_{freq}.csv", index=False)

    perf = pd.DataFrame([{
        "Freq": freq,
        "TWR_CAGR": round(stats["CAGR"],6),
        "TWR_VOL":  round(stats["VOL"],6),
        "TWR_MDD":  round(stats["MDD"],6),
        "TWR_Sharpe": round(stats["Sharpe"],6),
        "MWR_XIRR":   round(xirr_annual,6),
        "Final_Equity": round(eq.iloc[-1],2),
        "Total_Contrib": int(initial_capital + monthly_contribution * max(0, len(set(eq.index.to_period('M')))-1))
    }])
    perf["Final_Equity"] = perf["Final_Equity"].round(0).astype("int64")
    perf.to_csv(outdir / f"ml_perf_{freq}.csv", index=False)

    # Console view
    fmt = {c: (lambda v: f"{v:.2%}") for c in ["risk_on","neutral","risk_off"]}
    print("\n=== Learned Weights (fixed per regime, BIL<=25%) ===")
    print(wt_df.to_string(index=False, formatters=fmt))
    print("\n=== Performance (ML weights) ===")
    print(perf.to_string(index=False, formatters={"Final_Equity": lambda v: f"{v:,}"}))
    print(f"\nSaved: {outdir/f'ml_weights_{freq}.csv'}")
    print(f"Saved: {outdir/f'ml_nav_{freq}.csv'}")
    print(f"Saved: {outdir/f'ml_equity_{freq}.csv'}")
    print(f"Saved: {outdir/f'ml_perf_{freq}.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["ME","2W-FRI","W-FRI"], default="ME", help="리밸런싱/리샘플 빈도")
    ap.add_argument("--tc", type=float, default=0.2, help="거래비용(%)")
    ap.add_argument("--init", type=float, default=1_000_000.0, help="초기 자본")
    ap.add_argument("--contrib_month", type=float, default=300_000.0, help="월말 납입액(0이면 없음)")
    ap.add_argument("--lookback_min", type=int, default=36, help="Regime 학습 최소 기간(구간 수)")
    args = ap.parse_args()

    run(freq=args.freq, tc_percent=args.tc, initial_capital=args.init,
        monthly_contribution=args.contrib_month, lookback_min=args.lookback_min)
