# step1_visualize_full.py
# -----------------------------------------------------------
# Robust visualization + HTML dashboard for macro signals
# -----------------------------------------------------------
import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# -----------------------------------------------------------
# 0. 환경 설정
# -----------------------------------------------------------
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    raise RuntimeError("FRED_API_KEY missing. Put it in .env like: FRED_API_KEY=YOUR_KEY")

OUTDIR = os.getenv("OUTDIR", "charts")
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

today = dt.date.today()
start_date = (today - dt.timedelta(days=3 * 365)).strftime("%Y-%m-%d")
fred = Fred(api_key=FRED_KEY)

# -----------------------------------------------------------
# 1. 헬퍼 함수들
# -----------------------------------------------------------
def fred_series(code, start=None):
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
            df = yf.download(tkr, period=period, interval=interval, progress=False, auto_adjust=False)
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

def save_line(ts, title, ylab, fname):
    plt.figure()
    plt.plot(ts.index, ts.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylab)
    plt.tight_layout()
    path = Path(OUTDIR) / fname
    plt.savefig(path)
    plt.close()
    return str(path)

def save_two_lines(ts1, ts2, label1, label2, title, ylab, fname):
    plt.figure()
    plt.plot(ts1.index, ts1.values, label=label1)
    plt.plot(ts2.index, ts2.values, label=label2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    path = Path(OUTDIR) / fname
    plt.savefig(path)
    plt.close()
    return str(path)

def bar_series(series, title, ylab, fname):
    plt.figure()
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.ylabel(ylab)
    plt.xticks(rotation=15)
    plt.tight_layout()
    path = Path(OUTDIR) / fname
    plt.savefig(path)
    plt.close()
    return str(path)

def latest_delta(s, n):
    s = s.dropna()
    if len(s) < (n + 1):
        return None
    val = s.iloc[-1] - s.iloc[-(n + 1)]
    if hasattr(val, "item"):
        val = val.item()
    return float(val)

# -----------------------------------------------------------
# 2. 시각화 + HTML 생성
# -----------------------------------------------------------
def main():
    # --- FRED ---
    dgs10 = fred_series("DGS10", start=start_date)
    dgs2  = fred_series("DGS2", start=start_date)
    tips10 = fred_series("DFII10", start=start_date)
    walcl = fred_series("WALCL", start=start_date)  # Fed Balance Sheet (Million USD)
    hyoas = fred_series("BAMLH0A0HYM2", start=start_date)

    # --- yfinance (fallbacks) ---
    dxy_df, dxy_used = yf_series_multi(["DX-Y.NYB", "DX=F"])
    usdk_df, _       = yf_series_multi(["KRW=X"])
    vix_df, _        = yf_series_multi(["^VIX"])
    btc_df, btc_used = yf_series_multi(["BTC-USD", "XBT-USD", "BTCUSD=X"])

    # --- 월말 리샘플 ---
    dgs10_m = eom(dgs10)
    dgs2_m  = eom(dgs2)
    tips_m  = eom(tips10)
    dxy_m   = eom(dxy_df)
    usdk_m  = eom(usdk_df)
    vix_m   = eom(vix_df)
    hyoas_m = eom(hyoas)
    walcl_m = eom(walcl)

    # --- 파생 ---
    be_m = (dgs10_m - tips_m) * 100.0 if dgs10_m is not None and tips_m is not None else None
    curve_m = (dgs10_m - dgs2_m) * 100.0 if dgs10_m is not None and dgs2_m is not None else None

    # --- MOVE Proxy ---
    move_proxy = None
    if dgs10 is not None and len(dgs10.dropna()) >= 22:
        chg_bp = dgs10.dropna().diff().dropna() * 100.0
        rolling_std_bp = chg_bp.rolling(21).std().dropna()
        if len(rolling_std_bp):
            move_proxy = float(rolling_std_bp.iloc[-1] * np.sqrt(252.0))

    # --- 차트 생성 ---
    imgs = []
    if dgs10_m is not None and dgs2_m is not None and len(dgs10_m) and len(dgs2_m):
        imgs.append(save_two_lines(dgs10_m, dgs2_m, "US10Y", "US2Y",
                                   "US10Y vs US2Y (Monthly)", "%", "ts_us10_vs_us2.png"))
    if tips_m is not None and len(tips_m):
        imgs.append(save_line(tips_m, "10Y TIPS Real Yield (Monthly)", "%", "ts_tips10.png"))
    if be_m is not None and len(be_m):
        imgs.append(save_line(be_m, "10Y Breakeven (bps, Monthly)", "bps", "ts_breakeven10.png"))
    if curve_m is not None and len(curve_m):
        imgs.append(save_line(curve_m, "10Y–2Y Curve (bps, Monthly)", "bps", "ts_curve_10y_2y.png"))
    if dxy_m is not None and len(dxy_m):
        title = "DXY (Monthly)" + (f" — used {dxy_used}" if dxy_used else "")
        imgs.append(save_line(dxy_m, title, "Index", "ts_dxy.png"))
    if usdk_m is not None and len(usdk_m):
        imgs.append(save_line(usdk_m, "USDKRW (Monthly)", "KRW per USD", "ts_usdkrw.png"))
    if vix_m is not None and len(vix_m):
        imgs.append(save_line(vix_m, "VIX (Monthly)", "Index", "ts_vix.png"))
    if hyoas_m is not None and len(hyoas_m):
        imgs.append(save_line(hyoas_m, "HY OAS (Monthly)", "bps", "ts_hyoas.png"))

    # --- Δ3M / Δ1Y 막대 ---
    def add_delta_block(name, s, mul=1.0):
        if s is None or len(s) < 13:
            return None
        d3 = s.iloc[-1] - s.iloc[-4] if len(s) >= 4 else None
        d12 = s.iloc[-1] - s.iloc[-13] if len(s) >= 13 else None
        if d3 is None and d12 is None:
            return None
        d3v = float(d3) * mul if d3 is not None else None
        d12v = float(d12) * mul if d12 is not None else None
        return pd.Series({f"{name} Δ3M": d3v, f"{name} Δ1Y": d12v}).dropna()

    bar_parts = [
        add_delta_block("Real (10Y TIPS, bp)", tips_m, 100.0),
        add_delta_block("Curve (10Y–2Y, bp)", curve_m, 1.0),
        add_delta_block("DXY", dxy_m, 1.0),
        add_delta_block("USDKRW", usdk_m, 1.0)
    ]

    if walcl_m is not None:
        walcl_trn_m = walcl_m / 1_000_000.0
        bar_parts.append(add_delta_block("Liquidity (Fed BS, Trn)", walcl_trn_m, 1.0))

    bar_img = None
    merged = pd.concat([p for p in bar_parts if p is not None], axis=0) if any(p is not None for p in bar_parts) else None
    if merged is not None and not merged.empty:
        bar_img = bar_series(merged, "Latest Δ3M / Δ1Y", "Δ (unit per label)", "bars_latest_deltas.png")

    # --- HTML Report ---
    sections = []
    sections.append({
        "title": "Rates — US10Y & US2Y",
        "desc": "장단기 금리 수준과 스프레드는 경기/리스크 선호에 민감합니다.",
        "imgs": [img for img in imgs if "us10_vs_us2" in img]
    })
    sections.append({
        "title": "Real Yield & Breakeven",
        "desc": "실질금리(10Y TIPS)와 브레이크이븐(기대 인플레)의 변화를 월말 기준으로 추적합니다.",
        "imgs": [img for img in imgs if "tips10" in img or "breakeven" in img]
    })
    sections.append({
        "title": "Dollar / FX & Credit",
        "desc": "달러 강세(DXY), 원/달러, 하이일드 스프레드(HY OAS)를 묶어 유동성과 리스크 프리미엄을 확인합니다.",
        "imgs": [img for img in imgs if "dxy" in img or "usdkrw" in img or "hyoas" in img]
    })
    if bar_img:
        sections.append({
            "title": "Latest Δ3M / Δ1Y",
            "desc": "최근 3개월/1년 변화량을 한눈에 비교합니다.",
            "imgs": [bar_img]
        })

    html_path = Path(OUTDIR) / "index.html"
    html_body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Macro Dashboard — Step 1</title>",
        "<style>body{font-family:-apple-system,Roboto,sans-serif;max-width:1000px;margin:40px auto;padding:0 16px;}",
        ".card{box-shadow:0 6px 20px rgba(0,0,0,.08);border-radius:14px;padding:16px;margin:18px 0;}",
        "h1{font-size:28px} h2{font-size:20px;margin-bottom:12px} img{max-width:100%;border-radius:10px;}",
        ".grid{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.grid{grid-template-columns:1fr 1fr;}}",
        ".muted{color:#666;font-size:13px}</style></head><body>",
        f"<h1>Macro Dashboard — Step 1 <span class='muted'>(Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')})</span></h1>"
    ]
    for sec in sections:
        html_body.append("<div class='card'>")
        html_body.append(f"<h2>{sec['title']}</h2>")
        html_body.append(f"<p>{sec['desc']}</p>")
        html_body.append("<div class='grid'>")
        for img in sec.get("imgs", []):
            html_body.append(f"<div><img src='{img}'/><div class='muted'>{img}</div></div>")
        html_body.append("</div></div>")
    html_body.append("</body></html>")

    html_path.write_text("\n".join(html_body), encoding="utf-8")
    print("✅ Charts saved to:", OUTDIR)
    print("✅ Open HTML report:", html_path)

# -----------------------------------------------------------
if __name__ == "__main__":
    main()
