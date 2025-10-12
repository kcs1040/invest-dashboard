# -*- coding: utf-8 -*-
"""
strategy_weight_ml_wf.py  (WF + VolTarget + Return-Enhanced + Weights Export + Regime Averages)

- Universe(6): SPY, IEF, BIL, EEM, GLD, QQQ
- Regime5: very_risk_off / risk_off / neutral / risk_on / very_risk_on
- Walk-forward + small grid tuning (Ridge-based μ with momentum) + volatility targeting
- Constraints: long-only, sum=1, BIL cap, min equity floor across regimes
- DCA monthly, trading cost by turnover × tc%
- Exports:
  1) data_pro/wf_nav_<freq>.csv
  2) data_pro/wf_equity_<freq>.csv
  3) data_pro/wf_perf_<freq>.csv
  4) data_pro/wf_weights_<freq>.csv                (date-wise w & exposures)
  5) data_pro/wf_positions_<freq>.csv              (date-wise live position plan)
  6) data_pro/wf_learned_weights_by_regime_<freq>.csv  (NEW: avg weights by regime)
- Console prints a full table: avg weights per regime (5 columns)

Run:
(.venv) ➜ python strategy_weight_ml_wf.py --freq W-FRI --target20 --fast
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

try:
    from sklearn.linear_model import RidgeCV
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ASSETS6 = ["SPY","IEF","BIL","EEM","GLD","QQQ"]
EQUITY_NAMES = ["SPY","QQQ","EEM"]
RFR_ASSUMPTION = 0.02
START_YEARS_DEFAULT = 10

LEVERAGE_MAP_2X = {
    "SPY": ("SPY", "SSO"),
    "QQQ": ("QQQ", "QLD"),
    "IEF": ("IEF", "UST"),
    "GLD": ("GLD", "UGL"),
    "EEM": ("EEM", None),
    "BIL": ("BIL", None),
}

# ---------- helpers ----------
def end_date(): return pd.Timestamp.today().normalize()
def start_date(years=START_YEARS_DEFAULT): return end_date() - relativedelta(years=years, days=7)

def to_series_1d(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series): s=x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x["Close"] if "Close" in x.columns else x.iloc[:,0]; s=s.squeeze()
    else:
        arr = np.asarray(x).squeeze()
        if arr.ndim != 1: raise ValueError(f"Expected 1D, got {arr.shape}")
        s = pd.Series(arr, name=name)
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce"); s = s[~s.index.isna()]
    return s

def yf_close(ticker: str, years=START_YEARS_DEFAULT, tries=2, sleep_sec=0.8) -> pd.Series:
    for _ in range(tries):
        try:
            df = yf.download(ticker, start=start_date(years), end=end_date(),
                             interval="1d", progress=False, auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s[~s.index.isna()]
                if len(s) >= 20: return s
        except Exception: pass
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
    df = pd.DataFrame({"d": index}); df["ym"]=df["d"].dt.to_period("M")
    return list(df.groupby("ym", sort=True)["d"].max().values)

def ann_stats(nav: pd.Series, ppy: int) -> dict:
    nav = nav.dropna(); rets = nav.pct_change().dropna()
    if rets.empty: return {"CAGR":np.nan,"VOL":np.nan,"MDD":np.nan,"Sharpe":np.nan}
    years = (nav.index[-1]-nav.index[0]).days/365.25
    CAGR = (nav.iloc[-1]/nav.iloc[0])**(1/years)-1 if years>0 else np.nan
    VOL  = rets.std()*np.sqrt(ppy); MDD  = float((1 - nav/nav.cummax()).max())
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
        mid=(lo+hi)/2; f_mid=xnpv(mid,cfs,dates)
        if abs(f_mid)<1e-10: return mid
        if f_lo*f_mid<0: hi,f_hi=mid,f_mid
        else: lo,f_lo=mid,f_mid
    return mid

# ---------- env ----------
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY: raise RuntimeError("FRED_API_KEY missing in .env")
fred = Fred(api_key=FRED_KEY)

# ---------- data ----------
def load_price_panel_6(years=START_YEARS_DEFAULT) -> pd.DataFrame:
    frames, lengths = [], {}
    for t in ASSETS6:
        s = yf_close(t, years=years); s = to_series_1d(s, name=t); lengths[t]=len(s)
        if len(s)>0: frames.append(s.rename(t))
    print("Price lengths (6):", lengths, flush=True)
    if not frames: raise RuntimeError("No price data.")
    px = pd.concat(frames, axis=1).sort_index().dropna(how="all")
    px = px.loc[:, px.count()>10]
    if px.empty: raise RuntimeError("All six price series empty/too short.")
    return px

# ---------- regimes ----------
DELTA_K = {"ME":3, "2W-FRI":3, "W-FRI":3}

def build_signals_base(years=START_YEARS_DEFAULT):
    dgs10 = fred_series("DGS10"); dgs2 = fred_series("DGS2"); dfii10 = fred_series("DFII10")
    walcl = fred_series("WALCL"); hyoas = fred_series("BAMLH0A0HYM2")
    dxy = pd.Series(dtype=float)
    for t in ["DX-Y.NYB","DX=F"]:
        s = yf_close(t, years=years)
        if s is not None and len(s)>0: dxy=s; break
    vix  = yf_close("^VIX", years=years)
    st, ed = start_date(years), end_date()
    def clip(s): s=to_series_1d(s).dropna(); return s[(s.index>=st)&(s.index<=ed)]
    return {"DGS10":clip(dgs10), "DGS2":clip(dgs2), "DFII10":clip(dfii10),
            "WALCL":clip(walcl), "HYOAS":clip(hyoas),
            "DXY":clip(dxy), "VIX":clip(vix)}

def regime_series_5(signals_daily: dict, freq: str) -> pd.DataFrame:
    freq = "ME" if freq=="M" else freq; k = DELTA_K.get(freq,3)
    def R(s): return eop_resample(signals_daily[s],freq)
    DGS10, DGS2, DFII10 = R("DGS10"), R("DGS2"), R("DFII10")
    DXY, VIX, WALCL, HYOAS = R("DXY"), R("VIX"), R("WALCL"), R("HYOAS")
    idx = DGS10.index
    for s in [DGS2, DFII10, DXY, VIX, WALCL, HYOAS]: idx = idx.intersection(s.index)
    idx = idx.sort_values()
    curve=(DGS10-DGS2).reindex(idx); real=DFII10.reindex(idx)
    dxy_s=DXY.reindex(idx); vix_s=VIX.reindex(idx); hy_s=HYOAS.reindex(idx); wal_s=WALCL.reindex(idx)
    real_dk=real-real.shift(k); dxy_dk=dxy_s-dxy_s.shift(k); hyoas_dk=hy_s-hy_s.shift(k); walcl_dk=wal_s-wal_s.shift(k)
    score=pd.DataFrame(index=idx)
    score["real_dk"]=np.where(real_dk<0,+1,-1)
    score["dxy_dk"]=np.where(dxy_dk<0,+1,-1)
    score["curve_lv"]=np.where(curve>0,+1,-1)
    score["vix_bin"]=np.select([vix_s<20, vix_s>25],[1,-1],default=0)
    score["hyoas_dk"]=np.where(hyoas_dk<0,+1,-1)
    score["walcl_dk"]=np.where(walcl_dk>0,+1,-1)
    score=score.dropna(); score["Total"]=score.sum(axis=1)
    tot=score["Total"]
    conds=[(tot<=-4),((tot>=-3)&(tot<=-1)),(tot==0),((tot>=1)&(tot<=3)),(tot>=4)]
    labels=["very_risk_off","risk_off","neutral","risk_on","very_risk_on"]
    score["Regime5"]=np.select(conds, labels, default="neutral")
    return score[["Total","Regime5"]]

# ---------- returns ----------
def resample_prices(px_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    freq = "ME" if freq=="M" else freq
    d={}; 
    for c in px_daily.columns:
        s = eop_resample(px_daily[c], freq)
        if not s.empty: d[c]=s
    if not d: return pd.DataFrame()
    df = pd.DataFrame(d).dropna(how="all")
    min_cols=max(2,int(0.5*df.shape[1]))
    return df[df.count(axis=1)>=min_cols]

def returns_from_prices(px: pd.DataFrame) -> pd.DataFrame:
    rets=px.pct_change()
    if len(rets)>=1: rets.iloc[0]=0.0
    return rets.dropna(how="all")

# ---------- μ, Σ ----------
def ewma_cov(returns: pd.DataFrame, halflife: int = 36) -> np.ndarray:
    r=returns.fillna(0.0).copy()
    vol=r.ewm(halflife=halflife, min_periods=max(12,halflife//2)).std().iloc[-1].replace(0,np.nan)
    vol=vol.fillna(vol.median() if vol.notna().any() else 1.0)
    z=r/vol
    w=np.exp(-np.log(2)/halflife*np.arange(len(z))[::-1]); w=w/w.sum()
    zc=z
    C=(zc.T*w)@zc.values
    V=np.diag(vol.values)
    cov=V@C@V
    return cov

def blended_mu_with_momentum(rets: pd.DataFrame, alpha: float = 0.6) -> np.ndarray:
    cols=list(rets.columns); mu_mean=rets.mean().values
    if not SKLEARN_OK or len(rets)<260: return alpha*mu_mean+(1-alpha)*mu_mean
    m1=rets.rolling(21).mean(); m3=rets.rolling(63).mean()
    m6=rets.rolling(126).mean(); m12=rets.rolling(252).mean()
    X=pd.concat([m1,m3,m6,m12],axis=1,keys=["m1","m3","m6","m12"]).dropna()
    if X.empty: return mu_mean
    Y=rets.loc[X.index]
    mu_mom=[]
    alphas=np.logspace(-4,2,15)
    for c in cols:
        model=RidgeCV(alphas=alphas, fit_intercept=True)
        model.fit(X.values, Y[c].values)
        y_hat=model.predict(X.values)
        mu_mom.append(float(np.mean(y_hat)))
    mu_mom=np.array(mu_mom)
    return alpha*mu_mom+(1-alpha)*mu_mean

# ---------- constraints ----------
def regime_caps_vector(regime_name: str, cols: list[str], caps_scale: float, bil_cap: float) -> np.ndarray:
    base={
        "very_risk_off":{"SPY":0.20,"QQQ":0.15,"EEM":0.10,"GLD":0.40,"IEF":0.80,"BIL":bil_cap},
        "risk_off":     {"SPY":0.30,"QQQ":0.25,"EEM":0.15,"GLD":0.40,"IEF":0.70,"BIL":bil_cap},
        "neutral":      {"SPY":0.50,"QQQ":0.40,"EEM":0.25,"GLD":0.35,"IEF":0.60,"BIL":bil_cap},
        "risk_on":      {"SPY":0.65,"QQQ":0.55,"EEM":0.30,"GLD":0.30,"IEF":0.50,"BIL":bil_cap},
        "very_risk_on": {"SPY":0.75,"QQQ":0.65,"EEM":0.35,"GLD":0.25,"IEF":0.45,"BIL":bil_cap},
    }
    for k in ["SPY","QQQ","EEM"]:
        for reg in base: base[reg][k]=min(0.95, base[reg][k]*caps_scale)
    caps=np.array([base.get(regime_name, base["neutral"]).get(c,1.0) for c in cols], dtype=float)
    if "BIL" in cols:
        j=cols.index("BIL"); caps[j]=min(caps[j], bil_cap)
    return caps

def project_with_caps_and_min_eq(w, caps, min_eq, col_order):
    w=np.clip(w,0.0,None); s=w.sum()
    w=(w if s>0 else np.ones_like(w)/len(w))/(s if s>0 else 1.0)
    w=np.minimum(w,caps); w=w/w.sum()
    idx_eq=[col_order.index(n) for n in EQUITY_NAMES if n in col_order]
    eq_sum=w[idx_eq].sum() if idx_eq else 0.0
    if eq_sum<min_eq and idx_eq:
        deficit=min_eq-eq_sum
        idx_non=[i for i,_ in enumerate(col_order) if i not in idx_eq]
        non_sum=w[idx_non].sum() if idx_non else 0.0
        if non_sum>1e-12:
            w[idx_non]=np.clip(w[idx_non]-deficit*(w[idx_non]/non_sum),0,None)
            w=w/w.sum()
        eq_sum=w[idx_eq].sum()
        if eq_sum<min_eq:
            add=min_eq-eq_sum
            for i in idx_eq: w[i]+=add/len(idx_eq)
            w=np.maximum(w,0.0); w=w/w.sum()
    return w

# ---------- optimizer ----------
def optimize_weights(rets: pd.DataFrame, regime_name: str, alpha_mom: float, halflife_cov: int,
                     lambda_cash: float, min_equity: float, n_iter: int, lr: float,
                     mu_bonus: float, caps_scale: float, bil_cap: float) -> np.ndarray:
    cols=list(rets.columns)
    mu=blended_mu_with_momentum(rets[cols], alpha=alpha_mom)
    cov=ewma_cov(rets[cols], halflife=halflife_cov)
    caps=regime_caps_vector(regime_name, cols, caps_scale=caps_scale, bil_cap=bil_cap)
    w=np.ones(len(cols))/len(cols); w=project_with_caps_and_min_eq(w,caps,min_equity,cols)
    eps=1e-10
    for _ in range(n_iter):
        den=w@cov@w+eps; num=w@mu
        grad_sharpe=(mu*np.sqrt(den)-num*(cov@w)/np.sqrt(den))/den
        grad_penalty=np.zeros_like(w)
        if "BIL" in cols: grad_penalty[cols.index("BIL")]=-lambda_cash
        grad=grad_sharpe + mu_bonus*mu + grad_penalty
        w=w+lr*grad
        w=project_with_caps_and_min_eq(w,caps,min_equity,cols)
    w=np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w=np.clip(w,0.0,None); s=w.sum()
    w = (w if s>1e-12 else np.ones_like(w)/len(w)) / (s if s>1e-12 else 1.0)
    return w

# ---------- tuning ----------
def tune_hyper_and_weights(rets_train: pd.DataFrame, regime_name: str, valid_frac: float, fast: bool,
                           mu_bonus: float, min_equity_floor: float, lambda_cash_scale: float,
                           caps_scale: float, bil_cap: float):
    n=len(rets_train); cut=max(12,int(n*(1-valid_frac))); tr,va=rets_train.iloc[:cut],rets_train.iloc[cut:]
    if len(va)<6: va=tr
    cols=list(tr.columns); va=va.reindex(columns=cols)
    LAMBDA_CASH5={"very_risk_off":0.03,"risk_off":0.05,"neutral":0.10,"risk_on":0.20,"very_risk_on":0.30}
    for k in LAMBDA_CASH5: LAMBDA_CASH5[k]*=lambda_cash_scale
    if fast:
        grid_alpha=[0.6]; grid_hl=[36]; lam_mults=[1.0]; min_adds=[0.0]; n_iter_opt=120; lr_opt=0.10
    else:
        grid_alpha=[0.4,0.6,0.8]; grid_hl=[24,36,60]; lam_mults=[0.7,1.0,1.3]; min_adds=[0.0,0.05]; n_iter_opt=400; lr_opt=0.10
    best=(None,-1e9,None)
    for a in grid_alpha:
        for hl in grid_hl:
            for lm in lam_mults:
                for madd in min_adds:
                    lam=LAMBDA_CASH5[regime_name]*lm
                    mineq=max(min_equity_floor, min(0.90, (0.30 if regime_name=="very_risk_off" else 0.45)+madd))
                    w_try=optimize_weights(tr, regime_name, a, hl, lam, mineq, n_iter_opt, lr_opt, mu_bonus, caps_scale, bil_cap)
                    va_vals=va.fillna(0.0).values; w_vec=np.asarray(w_try,float)
                    if va_vals.shape[1]!=w_vec.shape[0]: w_vec=w_vec[:va_vals.shape[1]]
                    port=va_vals@w_vec
                    nav_va=(1.0+pd.Series(port,index=va.index)).cumprod()
                    stats=ann_stats(nav_va, ppy=12); sharpe=stats["Sharpe"] if np.isfinite(stats["Sharpe"]) else -1e9
                    if sharpe>best[1]: best=((a,hl,lam,mineq),sharpe,w_try)
    params,_,w_best=best
    if params is None:
        params=(0.6,36,0.10,max(min_equity_floor,0.45))
        w_best=optimize_weights(tr, regime_name, *params, n_iter=200, lr=0.12, mu_bonus=mu_bonus, caps_scale=caps_scale, bil_cap=bil_cap)
    return params, w_best

# ---------- positions ----------
def exposures_to_positions(exposures: dict[str,float], redistribute_excess: bool = True) -> list[dict]:
    plan=[]; excess_pool=0.0
    for a,e in exposures.items():
        base_tkr,lev2x_tkr=LEVERAGE_MAP_2X.get(a,(a,None))
        base_pct,lev2x_pct=0.0,0.0
        if lev2x_tkr:
            if e<=1.0: base_pct=e
            elif e<=2.0: base_pct=2.0-e; lev2x_pct=e-1.0
            else: lev2x_pct=1.0; excess_pool+=(e-2.0)
        else:
            if e<=1.0: base_pct=e
            else: base_pct=1.0; excess_pool+=(e-1.0)
        plan.append({"asset":a,"base_ticker":base_tkr,"base_pct":round(base_pct,6),
                     "lev2x_ticker":lev2x_tkr if lev2x_tkr else "","lev2x_pct":round(lev2x_pct,6)})
    if redistribute_excess and excess_pool>1e-9:
        room=sum(1.0-p["lev2x_pct"] for p in plan if p["lev2x_ticker"])
        if room>1e-9:
            for p in plan:
                if p["lev2x_ticker"]:
                    add=(1.0-p["lev2x_pct"])/room*excess_pool
                    p["lev2x_pct"]=round(min(1.0,p["lev2x_pct"]+add),6)
    return plan

# ---------- walk-forward ----------
def walk_forward(px_daily: pd.DataFrame, reg_df_5: pd.DataFrame, freq: str,
                 train_years: int, min_win: int,
                 tc_percent: float, initial_capital: float, monthly_contribution: float,
                 max_lev: float, fast: bool, mu_bonus: float, min_equity_floor: float,
                 lambda_cash_scale: float, caps_scale: float, bil_cap: float,
                 sigma_targets: dict, export_weights_path: Path, export_positions_path: Path,
                 redistribute_excess: bool):

    px = resample_prices(px_daily, freq)
    if px.empty: raise RuntimeError("Resampled prices empty.")
    rets = returns_from_prices(px); dates=rets.index
    reg = reg_df_5.reindex(dates).ffill().bfill()
    if "Regime5" not in reg.columns: raise RuntimeError("Regime5 missing.")

    contrib_dates=set(month_end_dates(dates))
    twr=pd.Series(index=dates,dtype=float); equity=initial_capital
    cf_vals=[-initial_capital]; cf_dates=[dates[0]]
    last_w=np.zeros(len(ASSETS6))
    cols=list(px.columns); idx_map=[cols.index(a) if a in cols else None for a in ASSETS6]

    ppy_map={"ME":12,"2W-FRI":26,"W-FRI":52}; ppy=ppy_map[freq]
    cache_weights={}
    weights_log=[]; positions_log=[]

    # NEW: regime average accumulator
    regimes_list=["very_risk_off","risk_off","neutral","risk_on","very_risk_on"]
    weights_accum={rg: np.zeros(len(ASSETS6)) for rg in regimes_list}
    weights_count={rg: 0 for rg in regimes_list}

    for i in range(len(dates)):
        t=dates[i]
        if i%4==0: print(f"[{t.date()}] progress {i+1}/{len(dates)}", flush=True)
        end_train=i-1
        if end_train<1: twr.iloc[i]=0.0; continue
        win_len=max(min_win, train_years*ppy)
        start_train=max(0, end_train-win_len+1)
        rets_train=rets.iloc[start_train:end_train+1]
        reg_train=reg.iloc[start_train:end_train+1]
        if len(rets_train)<min_win: twr.iloc[i]=0.0; continue

        month_key=t.to_period('M'); regime_t=str(reg.at[t,"Regime5"])
        if (month_key,regime_t) in cache_weights:
            learned=cache_weights[(month_key,regime_t)]
        else:
            learned={}
            for regime_name in regimes_list:
                mask=(reg_train["Regime5"]==regime_name)
                rsub=rets_train.loc[mask]
                if len(rsub)<max(12,int(0.2*len(rets_train))):
                    rsub=rets_train.tail(max(min_win,int(0.5*len(rets_train))))
                _,w=tune_hyper_and_weights(
                    rsub[ASSETS6].dropna(how="all"), regime_name,
                    valid_frac=0.25, fast=fast, mu_bonus=mu_bonus,
                    min_equity_floor=min_equity_floor,
                    lambda_cash_scale=lambda_cash_scale,
                    caps_scale=caps_scale, bil_cap=bil_cap
                )
                learned[regime_name]=w
            cache_weights[(month_key,regime_t)]=learned

        w_full=learned.get(regime_t, learned.get("neutral")).copy()

        # accumulate regime averages
        weights_accum[regime_t]+=w_full
        weights_count[regime_t]+=1

        # vol targeting
        look=min(60,len(rets_train)); rt_win=rets_train.tail(look)
        w_eff_tmp=np.array([w_full[ASSETS6.index(a)] if a in ASSETS6 else 0.0 for a in ASSETS6])
        w_eff_tmp=w_eff_tmp/max(w_eff_tmp.sum(),1e-12)
        port_rets=(rt_win[ASSETS6]@w_eff_tmp).dropna()
        est_sigma=port_rets.std()*np.sqrt(ppy) if len(port_rets)>2 else 0.0
        sigma_target=sigma_targets.get(regime_t, sigma_targets["neutral"])
        lev=1.0
        if est_sigma>1e-8: lev=min(max_lev, max(0.5, sigma_target/est_sigma))

        # logs
        exposures={a: float(lev*(w_full[ASSETS6.index(a)] if a in ASSETS6 else 0.0)) for a in ASSETS6}
        row={"date":t,"regime":regime_t,"lev":round(float(lev),6)}
        for a in ASSETS6: row[f"w_{a}"]=round(float(w_full[ASSETS6.index(a)]),6)
        for a in ASSETS6: row[f"exp_{a}"]=round(exposures[a],6)
        weights_log.append(row)

        pos=exposures_to_positions(exposures, redistribute_excess=redistribute_excess)
        for p in pos:
            positions_log.append({"date":t,"regime":regime_t,"asset":p["asset"],
                                  "base_ticker":p["base_ticker"],"base_pct":p["base_pct"],
                                  "lev2x_ticker":p["lev2x_ticker"],"lev2x_pct":p["lev2x_pct"],
                                  "lev":round(float(lev),6)})

        # DCA
        if t in contrib_dates and i>0 and monthly_contribution!=0.0:
            equity+=monthly_contribution; cf_vals.append(-monthly_contribution); cf_dates.append(t)

        # P&L
        mask_ok=np.array([ci is not None and not np.isnan(rets.iloc[i,ci]) for ci in idx_map],bool)
        if mask_ok.sum()==0: pr_net=0.0
        else:
            w_eff=np.where(mask_ok, w_full, 0.0); w_eff=w_eff/max(w_eff.sum(),1e-12)
            turnover=np.abs(w_eff-last_w).sum(); tc=turnover*(tc_percent/100.0)
            r_base=0.0
            for j,ok in enumerate(mask_ok):
                if ok: r_base+=w_eff[j]*rets.iloc[i, idx_map[j]]
            pr_net=lev*r_base - tc; last_w=w_eff
        twr.iloc[i]=pr_net; equity*= (1+pr_net)

    nav=(1.0+twr.fillna(0.0)).cumprod()
    # equity with DCA
    eq=pd.Series(index=dates,dtype=float); equity_re=initial_capital
    md=set(month_end_dates(dates))
    for i,t in enumerate(dates):
        if t in md and i>0 and monthly_contribution!=0.0: equity_re+=monthly_contribution
        equity_re*=(1+twr.iloc[i]); eq.iloc[i]=equity_re
    cf_vals.append(equity); cf_dates.append(dates[-1])
    xirr_annual=xirr(np.array(cf_vals,float), list(cf_dates))

    # export logs
    outdir=export_weights_path.parent; outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weights_log).to_csv(export_weights_path, index=False)
    pd.DataFrame(positions_log).to_csv(export_positions_path, index=False)

    # NEW: regime average table
    avg_tbl=[]
    for rg in regimes_list:
        cnt=max(1,weights_count[rg])
        avg=weights_accum[rg]/cnt
        row={"Asset":None,"very_risk_off":None,"risk_off":None,"neutral":None,"risk_on":None,"very_risk_on":None}
    # build column-wise table by assets
    avg_by_asset=[]
    for ai,a in enumerate(ASSETS6):
        avg_by_asset.append({
            "Asset":a,
            "very_risk_off":weights_accum["very_risk_off"][ai]/max(1,weights_count["very_risk_off"]),
            "risk_off":     weights_accum["risk_off"][ai]/max(1,weights_count["risk_off"]),
            "neutral":      weights_accum["neutral"][ai]/max(1,weights_count["neutral"]),
            "risk_on":      weights_accum["risk_on"][ai]/max(1,weights_count["risk_on"]),
            "very_risk_on": weights_accum["very_risk_on"][ai]/max(1,weights_count["very_risk_on"]),
        })
    avg_df=pd.DataFrame(avg_by_asset)
    avg_df.to_csv(outdir/f"wf_learned_weights_by_regime_{freq}.csv", index=False)

    return nav.dropna(), eq.dropna(), twr.dropna(), xirr_annual, avg_df

# ---------- main ----------
def run(args):
    sigma_targets={
        "very_risk_off": args.sigma_vro,
        "risk_off":      args.sigma_ro,
        "neutral":       args.sigma_neu,
        "risk_on":       args.sigma_ron,
        "very_risk_on":  args.very_risk_on_sigma,
    }
    px_daily=load_price_panel_6(years=args.history_years)
    signals=build_signals_base(years=args.history_years)
    reg5=regime_series_5(signals, freq=args.freq)

    outdir=Path("data_pro")
    nav,eq,twr,xirr_annual,avg_df=walk_forward(
        px_daily, reg5, freq=args.freq,
        train_years=args.train_years, min_win=args.min_win,
        tc_percent=args.tc, initial_capital=args.init,
        monthly_contribution=args.contrib_month,
        max_lev=args.max_lev, fast=args.fast,
        mu_bonus=args.mu_bonus, min_equity_floor=args.min_equity_floor,
        lambda_cash_scale=args.lambda_cash_scale, caps_scale=args.caps_scale, bil_cap=args.bil_cap,
        sigma_targets=sigma_targets,
        export_weights_path=outdir/f"wf_weights_{args.freq}.csv",
        export_positions_path=outdir/f"wf_positions_{args.freq}.csv",
        redistribute_excess=not args.no_redistribute
    )
    ppy_map={"ME":12,"2W-FRI":26,"W-FRI":52}
    stats=ann_stats((1.0+twr).cumprod(), ppy=ppy_map[args.freq])
    nav.to_csv(outdir/f"wf_nav_{args.freq}.csv"); eq.to_csv(outdir/f"wf_equity_{args.freq}.csv")
    perf=pd.DataFrame([{
        "Freq":args.freq,"TWR_CAGR":round(stats["CAGR"],6),"TWR_VOL":round(stats["VOL"],6),
        "TWR_MDD":round(stats["MDD"],6),"TWR_Sharpe":round(stats["Sharpe"],6),
        "MWR_XIRR":round(xirr_annual,6),
        "Final_Equity":round(eq.iloc[-1],2),
        "Total_Contrib":int(args.init + args.contrib_month * max(0, len(set(eq.index.to_period('M')))-1))
    }])
    perf["Final_Equity"]=perf["Final_Equity"].round(0).astype("int64")
    perf.to_csv(outdir/f"wf_perf_{args.freq}.csv", index=False)

    print("\n=== Performance (WF + VolTarget + Return-Enhanced, 5-Regimes) ===")
    print(perf.to_string(index=False, formatters={"Final_Equity": lambda v: f"{v:,}"}))

    # ★ 터미널: Regime 평균 가중치 표 출력
    def pct(x): 
        try: return f"{100*x:5.2f}%"
        except: return "  n/a "
    cols=["very_risk_off","risk_off","neutral","risk_on","very_risk_on"]
    print("\n=== Learned Weights (average by regime over walk-forward) ===")
    header="Asset " + " ".join([f"{c:>14}" for c in cols])
    print(header)
    for _,r in avg_df.iterrows():
        line=f"{r['Asset']:>5} " + " ".join([f"{pct(r[c]):>14}" for c in cols])
        print(line)

    print(f"\nSaved: {outdir/f'wf_nav_{args.freq}.csv'}")
    print(f"Saved: {outdir/f'wf_equity_{args.freq}.csv'}")
    print(f"Saved: {outdir/f'wf_perf_{args.freq}.csv'}")
    print(f"Saved: {outdir/f'wf_weights_{args.freq}.csv'}")
    print(f"Saved: {outdir/f'wf_positions_{args.freq}.csv'}")
    print(f"Saved: {outdir/f'wf_learned_weights_by_regime_{args.freq}.csv'}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["ME","2W-FRI","W-FRI"], default="ME")
    ap.add_argument("--tc", type=float, default=0.2)
    ap.add_argument("--init", type=float, default=1_000_000.0)
    ap.add_argument("--contrib_month", type=float, default=300_000.0)
    ap.add_argument("--train_years", type=int, default=5)
    ap.add_argument("--min_win", type=int, default=24)
    ap.add_argument("--max_lev", type=float, default=2.0)
    ap.add_argument("--history_years", type=int, default=10)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mu_bonus", type=float, default=0.25)
    ap.add_argument("--sigma_vro", type=float, default=0.08)
    ap.add_argument("--sigma_ro",  type=float, default=0.11)
    ap.add_argument("--sigma_neu", type=float, default=0.15)
    ap.add_argument("--sigma_ron", type=float, default=0.20)
    ap.add_argument("--very_risk_on_sigma", type=float, default=0.26)
    ap.add_argument("--min_equity_floor", type=float, default=0.30)
    ap.add_argument("--bil_cap", type=float, default=0.25)
    ap.add_argument("--lambda_cash_scale", type=float, default=1.0)
    ap.add_argument("--caps_scale", type=float, default=1.0)
    ap.add_argument("--no_redistribute", action="store_true")
    ap.add_argument("--target20", action="store_true")
    args=ap.parse_args()
    if args.target20:
        args.mu_bonus=max(args.mu_bonus,0.35)
        args.very_risk_on_sigma=max(args.very_risk_on_sigma,0.30)
        args.sigma_neu=max(args.sigma_neu,0.17)
        args.sigma_ron=max(args.sigma_ron,0.22)
        args.caps_scale=max(args.caps_scale,1.15)
        args.max_lev=max(args.max_lev,2.0)
        args.min_equity_floor=max(args.min_equity_floor,0.30)
    load_dotenv()
    FRED_KEY=os.getenv("FRED_API_KEY")
    if not FRED_KEY: raise RuntimeError("FRED_API_KEY missing in .env")
    run(args)