Step 1 Macro Signals — Quickstart
1) mkdir -p ~/Documents/code_work/invest_2nd && cd ~/Documents/code_work/invest_2nd
2) (copy these files here)
3) python3 -m venv .venv && source .venv/bin/activate
4) pip install -r requirements.txt
5) cp .env.example .env  &&  edit .env to set FRED_API_KEY
6) python step1_macro_signals.py
Outputs: Macro_Dashboard_Data.xlsx (Inputs, Signals), plus CSVs.
