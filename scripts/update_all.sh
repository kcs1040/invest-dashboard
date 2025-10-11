#!/usr/bin/env bash
set -Eeuo pipefail

LOG_DIR="$HOME/Library/Logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/invest_dashboard.log"

{
  echo "===== RUN @ $(date '+%Y-%m-%d %H:%M:%S') ====="

  PROJECT_DIR="/Users/cskim/Documents/code_work/Invest_2nd"
  VENV_DIR="$PROJECT_DIR/.venv"

  # PATH 보강 (homebrew, system)
  export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"

  cd "$PROJECT_DIR"
  echo "[info] pwd=$(pwd)"

  # 가상환경 활성화
  if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[info] python=$(command -v python)"
    python --version
  else
    echo "[warn] venv not found at $VENV_DIR; trying system python"
    which python || true
  fi

  # 1) 엑셀 생성
  echo "[run] python invest_excel.py"
  python invest_excel.py

  # 2) 대시보드 생성
  echo "[run] python invest_dashboard.py"
  python invest_dashboard.py

  # 3) GitHub Pages용 파일 반영 (폴더명 확인!)
  SRC_HTML="$PROJECT_DIR/dash_pro_perchart/index.html"
  if [ ! -f "$SRC_HTML" ]; then
    ALT_HTML="$PROJECT_DIR/dash_pro/index.html"
    if [ -f "$ALT_HTML" ]; then
      SRC_HTML="$ALT_HTML"
    fi
  fi
  echo "[info] dashboard source html: $SRC_HTML"

  mkdir -p "$PROJECT_DIR/docs"
  cp -f "$SRC_HTML" "$PROJECT_DIR/docs/index.html"

  # 4) 산출물 보관(선택)
  mkdir -p "$PROJECT_DIR/docs/data_pro"
  cp -f "$PROJECT_DIR/data_pro/Macro_Dashboard_Full_plus_Compact_PRO.xlsx" "$PROJECT_DIR/docs/data_pro/" || true

  # 5) (옵션) 자동 커밋/푸시
  if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[git] add/commit/push"
    git -C "$PROJECT_DIR" add docs/index.html docs/data_pro/*.xlsx 2>/dev/null || true
    git -C "$PROJECT_DIR" commit -m "auto: update dashboard & excel $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || true
    git -C "$PROJECT_DIR" push 2>/dev/null || true
  fi

  echo "===== DONE @ $(date '+%Y-%m-%d %H:%M:%S') ====="
} >>"$LOG_FILE" 2>&1
