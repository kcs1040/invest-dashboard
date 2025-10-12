import pandas as pd, numpy as np
import yfinance as yf
from pathlib import Path

# 리샘플 규칙: 주간은 금요일 기준(W-FRI), 2주는 2W-FRI, 월말은 M.
RESAMPLE_RULES = {
    "Weekly": "W-FRI",
    "Biweekly": "2W-FRI",
    "Monthly": "M"
}
ANN_FACTOR = {"Weekly":52, "Biweekly":26, "Monthly":12}

def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    px = yf.download(tickers, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all")

def resample_prices(px: pd.DataFrame, rule: str) -> pd.DataFrame:
    # 기간 끝가격(리밸런싱 시점) 사용
    return px.resample(rule).last()

def backtest_once(signals: pd.DataFrame, tickers: list[str], label: str) -> pd.DataFrame:
    rule = RESAMPLE_RULES[label]
    px = fetch_prices(tickers)
    px = resample_prices(px, rule).dropna(how="all")
    # 수익률은 리밸런싱 주기에 맞춘 단위 수익률
    ret = px.pct_change().dropna(how="all")

    # 시그널 날짜를 해당 주기 인덱스에 맞춤 (과거 데이터 사용 원칙: ffill)
    sig = signals.copy()
    if "Date" in sig.columns:
        sig = sig.set_index("Date")
    sig.index = pd.to_datetime(sig.index)
    sig = sig.sort_index()
    sig = sig.reindex(ret.index, method="ffill")

    # 타깃 가중치 (티커명과 동일한 컬럼 필요)
    missing_cols = [t for t in tickers if t not in sig.columns]
    if missing_cols:
        raise ValueError(f"Missing weight columns in signals: {missing_cols}")

    w = sig[tickers].copy()
    # 포트폴리오 수익 = Σ w_ticker * r_ticker (동일 기간 수익률)
    port_ret = (w * ret).sum(axis=1)
    curve = (1 + port_ret).cumprod()
    out = pd.DataFrame({"ret":port_ret, "cum":curve})
    out.index.name = "Date"
    return out

def mdd(series: pd.Series) -> float:
    rollmax = series.cummax()
    dd = series/rollmax - 1.0
    return float(dd.min()) if len(dd) else 0.0

def ann_cagr(cum_series: pd.Series, ann_factor: int) -> float:
    if len(cum_series) < 2:
        return 0.0
    total = float(cum_series.iloc[-1])
    years = len(cum_series) / ann_factor
    if years <= 0 or total <= 0:
        return 0.0
    return total**(1/years) - 1

def ann_sharpe(period_ret: pd.Series, ann_factor: int, rf: float = 0.0) -> float:
    mu = period_ret.mean()*ann_factor
    sd = period_ret.std()*np.sqrt(ann_factor)
    return 0.0 if sd == 0 or np.isnan(sd) else (mu - rf)/sd

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True, help="data/strategy_signals.csv")
    ap.add_argument("--tickers", required=True, help="SPY,QQQ,EEM,BTC-USD,IEF,TLT,GLD,BIL")
    ap.add_argument("--out_html", required=True, help="docs/frequency_compare.html")
    ap.add_argument("--min_start", default="2015-01-01", help="데이터 시작(특히 BTC 고려)")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]
    sig = pd.read_csv(args.signals, parse_dates=["Date"])
    sig = sig[sig["Date"] >= pd.to_datetime(args.min_start)].copy()

    results = {}
    for label in ["Weekly","Biweekly","Monthly"]:
        bt = backtest_once(sig, tickers, label)
        results[label] = bt

    # === Plotly Report ===
    import plotly.graph_objs as go
    import plotly.subplots as sp
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Cumulative Return", "Periodic Return"))

    # 누적수익
    for label, df in results.items():
        fig.add_trace(go.Scatter(x=df.index, y=df["cum"], mode="lines", name=f"{label} (cum)"), row=1, col=1)
    # 구간 수익
    for label, df in results.items():
        fig.add_trace(go.Scatter(x=df.index, y=df["ret"], mode="lines", name=f"{label} (ret)"), row=2, col=1)

    fig.update_layout(title="Rebalancing Frequency Comparison (Weekly / 2-Weekly / Monthly)",
                      xaxis_title="Date")

    # === 성과 표 ===
    metrics = []
    for label, df in results.items():
        ann = ANN_FACTOR[label]
        metrics.append({
            "Freq": label,
            "CAGR": f"{ann_cagr(df['cum'], ann):.2%}",
            "MDD": f"{mdd(df['cum']):.2%}",
            "Sharpe": f"{ann_sharpe(df['ret'], ann):.2f}"
        })
    table = pd.DataFrame(metrics).set_index("Freq")

    table_html = table.to_html(border=0)

    html = f"""
    <html><head><meta charset='utf-8'><title>Rebalancing Frequency Comparison</title></head>
    <body>
    <h2>Macro Strategy – 8-Asset Portfolio</h2>
    <p>Tickers: {", ".join(tickers)}</p>
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    <h3>Performance Metrics</h3>
    {table_html}
    </body></html>
    """
    Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_html).write_text(html, encoding="utf-8")
    print(f"[OK] saved: {args.out_html}")
