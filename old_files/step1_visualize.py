
# step1_visualize.py
# -----------------------------------------------------------
# Read Macro_Dashboard_Data.xlsx, compute derived series, and save charts to ./charts
# Rules: matplotlib only, one chart per figure, no explicit colors/styles.
# -----------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_XLSX = os.getenv("INPUT_XLSX", "Macro_Dashboard_Data.xlsx")
OUTDIR = os.getenv("OUTDIR", "charts")
os.makedirs(OUTDIR, exist_ok=True)

def read_inputs(path):
    df = pd.read_excel(path, sheet_name="Inputs")
    # ensure Date parsing
    df["Date (YYYY-MM-DD)"] = pd.to_datetime(df["Date (YYYY-MM-DD)"], errors="coerce")
    df = df.dropna(subset=["Date (YYYY-MM-DD)"]).sort_values("Date (YYYY-MM-DD)")
    df = df.set_index("Date (YYYY-MM-DD)")
    return df

def end_of_month(s):
    s = s.dropna()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    return s.resample("ME").last().dropna()

def safe_plot(ts, title, ylab, fname):
    plt.figure()
    plt.plot(ts.index, ts.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname))
    plt.close()

def safe_plot_two(ts1, ts2, label1, label2, title, ylab, fname):
    plt.figure()
    plt.plot(ts1.index, ts1.values, label=label1)
    plt.plot(ts2.index, ts2.values, label=label2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname))
    plt.close()

def bar_latest(series, title, ylab, fname):
    # series: pd.Series with named bars
    plt.figure()
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.ylabel(ylab)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname))
    plt.close()

def main():
    df = read_inputs(INPUT_XLSX)

    # Core series
    us10 = df["US10Y Nominal Yield (%)"]
    us2  = df["US2Y Nominal Yield (%)"]
    tips = df["US10Y TIPS Real Yield (%)"]
    dxy  = df["DXY (Dollar Index)"]
    usdk = df["USDKRW"]
    move = df["MOVE (Bond Vol)"]
    vix  = df["VIX (Equity Vol)"]
    hy   = df["HY OAS (bps)"]
    be   = (us10 - tips) * 100.0

    # Monthly resamples
    us10_m = end_of_month(us10)
    us2_m  = end_of_month(us2)
    tips_m = end_of_month(tips)
    dxy_m  = end_of_month(dxy)
    usdk_m = end_of_month(usdk)
    move_m = end_of_month(move) if move.notna().sum() > 0 else None
    vix_m  = end_of_month(vix) if vix.notna().sum() > 0 else None
    hy_m   = end_of_month(hy) if hy.notna().sum() > 0 else None
    be_m   = end_of_month(be)

    # Curve (bps)
    curve_m = None
    if len(us10_m) and len(us2_m):
        curve_m = (us10_m - us2_m) * 100.0

    # === Time-series charts ===
    if len(us10_m) and len(us2_m):
        safe_plot_two(us10_m, us2_m, "US10Y", "US2Y", "US10Y vs US2Y (Monthly)", "%", "ts_us10_vs_us2.png")

    if len(tips_m):
        safe_plot(tips_m, "10Y TIPS Real Yield (Monthly)", "%", "ts_tips10.png")

    if len(be_m):
        safe_plot(be_m, "10Y Breakeven (bps, Monthly)", "bps", "ts_breakeven10.png")

    if curve_m is not None and len(curve_m):
        safe_plot(curve_m, "10Y–2Y Curve (bps, Monthly)", "bps", "ts_curve_10y_2y.png")

    if len(dxy_m):
        safe_plot(dxy_m, "DXY (Monthly)", "Index", "ts_dxy.png")

    if len(usdk_m):
        safe_plot(usdk_m, "USDKRW (Monthly)", "KRW per USD", "ts_usdkrw.png")

    if move_m is not None and len(move_m):
        safe_plot(move_m, "MOVE Proxy (bps, Monthly)", "bps", "ts_move_proxy.png")

    if vix_m is not None and len(vix_m):
        safe_plot(vix_m, "VIX (Monthly)", "Index", "ts_vix.png")

    if hy_m is not None and len(hy_m):
        safe_plot(hy_m, "HY OAS (Monthly)", "bps", "ts_hyoas.png")

    # === Delta bar (latest Δ3M / Δ1Y) ===
    def latest_delta(series, months):
        s = series.dropna()
        if len(s) < (months + 1):
            return None
        return float(s.iloc[-1] - s.iloc[-(months + 1)])

    bars = {}

    if len(tips_m) >= 13:
        v3 = latest_delta(tips_m, 3);  v12 = latest_delta(tips_m, 12)
        if v3 is not None:  bars["Real Δ3M (bp)"] = v3 * 100.0
        if v12 is not None: bars["Real Δ1Y (bp)"] = v12 * 100.0

    if curve_m is not None and len(curve_m) >= 13:
        v3 = latest_delta(curve_m, 3);  v12 = latest_delta(curve_m, 12)
        if v3 is not None:  bars["Curve Δ3M (bp)"] = v3
        if v12 is not None: bars["Curve Δ1Y (bp)"] = v12

    if len(dxy_m) >= 13:
        v3 = latest_delta(dxy_m, 3);  v12 = latest_delta(dxy_m, 12)
        if v3 is not None:  bars["DXY Δ3M"] = v3
        if v12 is not None: bars["DXY Δ1Y"] = v12

    if len(usdk_m) >= 13:
        v3 = latest_delta(usdk_m, 3);  v12 = latest_delta(usdk_m, 12)
        if v3 is not None:  bars["USDKRW Δ3M"] = v3
        if v12 is not None: bars["USDKRW Δ1Y"] = v12

    # 👉 문제의 한 줄을 두 줄로 분리 (walrus 제거)
    walcl_col = "Fed Balance Sheet (Trn USD)"
    walcl_m = end_of_month(df[walcl_col]) if walcl_col in df.columns else None
    if walcl_m is not None and len(walcl_m) >= 13:
        v3 = latest_delta(walcl_m, 3);  v12 = latest_delta(walcl_m, 12)
        if v3 is not None:  bars["Liquidity Δ3M (Trn)"] = v3
        if v12 is not None: bars["Liquidity Δ1Y (Trn)"] = v12

    if any(v is not None for v in bars.values()):
        bar_series = pd.Series({k: v for k, v in bars.items() if v is not None})
        if not bar_series.empty:
            bar_latest(bar_series, "Latest Δ3M / Δ1Y", "Δ (unit per label)", "bars_latest_deltas.png")


    print(f"Saved charts to: {OUTDIR}/")
    for f in sorted(os.listdir(OUTDIR)):
        print(" -", f)

if __name__ == "__main__":
    main()
