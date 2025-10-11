
# How to schedule daily updates

## macOS / Linux (cron)
1) Find Python path: `which python`  (e.g., /usr/local/bin/python or /opt/homebrew/bin/python3)
2) Edit crontab: `crontab -e`
3) Add a line (Asia/Seoul 09:05 every day):
   `5 9 * * * FRED_API_KEY=YOUR_KEY /path/to/python /path/to/macro_fetch_daily.py >> /path/to/macro_fetch.log 2>&1`

- If you want UTC date stamp, add `RECORD_TZ=utc` before the command.
- To customize output file: `OUTPUT_XLSX=/path/to/Macro_Dashboard_Data.xlsx`

## Windows (Task Scheduler)
1) Open "Task Scheduler" → Create Basic Task…
2) Trigger: Daily, 9:05 AM
3) Action: Start a Program
   Program/script: `C:\Path\to\python.exe`
   Arguments: `C:\Path\to\macro_fetch_daily.py`
   Start in: `C:\Path\to\` (folder containing your .env with FRED_API_KEY)
4) In your user environment variables, set `FRED_API_KEY` (or create a `.env` next to the script).

## Notes
- The script is idempotent per calendar day: it replaces the row for the same date if you run multiple times.
- WALCL (Fed balance sheet) updates weekly (Wed level, typically posted Thu US time). CPI monthly. Others daily/intraday.
- For reproducibility, consider choosing `RECORD_TZ=utc` or running after NY close to avoid intraday swings on DXY/VIX/BTC.
