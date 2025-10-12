import pandas as pd
from pathlib import Path

# === 8자산 비중 (Risk-on / Neutral / Risk-off) ===
WEIGHTS_8 = {
    "risk_on": {
        "SPY":0.25, "QQQ":0.20, "EEM":0.15, "BTC-USD":0.15,
        "IEF":0.10, "TLT":0.05, "GLD":0.05, "BIL":0.05
    },
    "neutral": {
        "SPY":0.15, "QQQ":0.10, "EEM":0.10, "BTC-USD":0.05,
        "IEF":0.15, "TLT":0.15, "GLD":0.20, "BIL":0.10
    },
    "risk_off": {
        "SPY":0.05, "QQQ":0.05, "EEM":0.00, "BTC-USD":0.00,
        "IEF":0.25, "TLT":0.25, "GLD":0.25, "BIL":0.15
    }
}

# 각 지표의 '리스크 방향' 지정 (+1: Risk-on과 같은 방향, -1: 반대 방향)
# Compact_Signals 시트에서 **z-score**가 이미 계산됐다고 가정합니다.
SIGNAL_DIRECTION = {
    "real_rate": -1,   # 실질금리↑ = 위험회피(리스크오프) → 음수 부호
    "DXY": -1,         # 달러↑ = 위험회피 경향
    "curve": +1,       # 수익률곡선 정상화/스티프닝↑ = 경기 개선 시그널
    "VIX": -1,         # 변동성↑ = 위험회피
    "HY_OAS": -1,      # 하이일드 스프레드↑ = 신용위험↑ = 위험회피
    "liquidity": +1    # 유동성↑ = 위험선호
}

# 열 이름 후보 (시트에서 실제 컬럼명이 다를 수 있어 유연 매칭)
CANDIDATES = {
    "real_rate": ["real_rate_z","real_rate","z_real_rate","real_yield_z","real_yield"],
    "DXY": ["DXY_z","DXY","z_DXY","usd_index_z","usd_index"],
    "curve": ["curve_z","curve","z_curve","term_spread_z","term_spread"],
    "VIX": ["VIX_z","VIX","z_VIX"],
    "HY_OAS": ["HY_OAS_z","HY_OAS","z_HY_OAS","hy_oas","credit_spread","credit_spread_z"],
    "liquidity": ["liquidity_z","liquidity","z_liquidity","liquidity_index_z","liq_z"]
}

def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def compute_macro_score(row: pd.Series, picked_cols: dict[str,str]) -> float:
    vals = []
    for key, sign in SIGNAL_DIRECTION.items():
        col = picked_cols.get(key)
        if col is not None and pd.notna(row.get(col)):
            vals.append(sign * float(row[col]))  # z-score × 방향
    return sum(vals)/len(vals) if vals else 0.0

def decide_regime(score: float, theta: float) -> str:
    if score > theta:  return "risk_on"
    if score < -theta: return "risk_off"
    return "neutral"

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Macro_Dashboard_Full_plus_Compact_PRO.xlsx")
    ap.add_argument("--sheet", default="Compact_Signals")
    ap.add_argument("--out", required=True, help="data/strategy_signals.csv")
    ap.add_argument("--theta", type=float, default=0.5, help="리스크 온/오프 임계값 (z-평균)")
    ap.add_argument("--weights", default="8", choices=["8"], help="현재는 8자산만 지원")
    ap.add_argument("--date_col", default="Date", help="날짜 컬럼명 (기본: Date)")
    args = ap.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet, engine="openpyxl")
    if args.date_col not in df.columns:
        raise ValueError(f"'{args.date_col}' column not found in {args.sheet}")

    # 컬럼 유연 매칭
    picked = {}
    for k, cand_list in CANDIDATES.items():
        col = pick_column(df, cand_list)
        if col is None:
            print(f"[WARN] Cannot find column for signal '{k}'. Candidates={cand_list}")
        picked[k] = col

    # 스코어/레짐 계산
    df["macro_score"] = df.apply(lambda r: compute_macro_score(r, picked), axis=1)
    df["regime"] = df["macro_score"].apply(lambda x: decide_regime(x, args.theta))

    # 날짜 정리 (월말 스냅샷이 섞여 있어도 됨)
    df = df.sort_values(args.date_col)
    out = df[[args.date_col, "macro_score", "regime"]].copy()

    # 상태별 타깃 가중치 병합
    W = WEIGHTS_8
    wdf = out.join(out["regime"].map(W).apply(pd.Series))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wdf.to_csv(args.out, index=False)
    print(f"[OK] saved: {args.out}")
