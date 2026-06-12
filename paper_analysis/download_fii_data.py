"""
FII/DII Daily Data Downloader
==============================

Downloads daily FII (Foreign Institutional Investor) equity activity
from NSE India's public data portal.

Source: NSE India — https://www.nseindia.com
Data: Daily FII/DII equity buy/sell/net values (₹ Crore)
Range: 2019-01-01 to today
Output: alpha-core/data/fii_dii_daily.csv

Run this script ONCE on your local machine (not in sandbox):
  cd alpha-core
  python paper_analysis/download_fii_data.py

If NSE returns 403 (blocks scraping), use the MANUAL DOWNLOAD fallback below.
"""

import pandas as pd
import requests
import time
import json
from pathlib import Path
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "fii_dii_daily.csv"


# ── NSE Session Setup ─────────────────────────────────────────────────────────
def get_nse_session() -> requests.Session:
    """
    NSE requires a cookie from the homepage before accepting API requests.
    This mimics a browser visit to get the session cookie.
    """
    session = requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
    }
    session.headers.update(headers)

    # Hit homepage to get cookie
    print("Establishing NSE session...")
    try:
        resp = session.get("https://www.nseindia.com", timeout=15)
        resp.raise_for_status()
        print(f"  Session established. Status: {resp.status_code}")
    except Exception as e:
        print(f"  Warning: Homepage request failed: {e}")
        print("  Will try API directly...")

    time.sleep(2)
    return session


# ── Fetch FII data from NSE API ───────────────────────────────────────────────
def fetch_nse_fii_chunk(session: requests.Session,
                         from_date: str, to_date: str) -> list:
    """
    Fetch one date chunk from NSE's FII/DII API.
    from_date, to_date: "DD-MM-YYYY"
    Returns list of dicts.
    """
    url = (
        f"https://www.nseindia.com/api/fiidiiTradeReact"
        f"?type=fiiDii&from={from_date}&to={to_date}"
    )
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        print(f"  {from_date} → {to_date}: {len(data)} records")
        return data
    except Exception as e:
        print(f"  Failed {from_date} → {to_date}: {e}")
        return []


def download_nse_fii(start: str = "2019-01-01") -> pd.DataFrame:
    """
    Download FII data in 3-month chunks (NSE API limit).
    """
    session = get_nse_session()
    all_records = []

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.today()

    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=90), end_dt)
        from_str = current.strftime("%d-%m-%Y")
        to_str   = chunk_end.strftime("%d-%m-%Y")

        records = fetch_nse_fii_chunk(session, from_str, to_str)
        all_records.extend(records)
        time.sleep(1.5)  # be polite to NSE
        current = chunk_end + timedelta(days=1)

    if not all_records:
        print("No data fetched from NSE API.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"\nRaw NSE data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


# ── Parse NSE JSON response ───────────────────────────────────────────────────
def parse_nse_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    NSE API returns columns like:
      date | category | buyValue | sellValue | netValue
    We want FII equity rows only.

    Typical category values: "FII/FPI", "DII", "MF"
    """
    if df.empty:
        return df

    # Normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # Filter FII rows
    if "category" in df.columns:
        fii_mask = df["category"].str.upper().str.contains("FII|FPI", na=False)
        fii_df   = df[fii_mask].copy()
        dii_mask = df["category"].str.upper().str.contains("DII", na=False)
        dii_df   = df[dii_mask].copy()
    else:
        fii_df = df.copy()
        dii_df = pd.DataFrame()

    def clean_series(df, col):
        if col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        )

    # Parse dates
    date_col = next((c for c in fii_df.columns if "date" in c), "date")
    dates = pd.to_datetime(fii_df[date_col], dayfirst=True, errors="coerce")

    result = pd.DataFrame(index=dates)
    result.index.name = "date"
    result["fii_buy"]  = clean_series(fii_df, "buyvalue").values
    result["fii_sell"] = clean_series(fii_df, "sellvalue").values
    result["fii_net"]  = clean_series(fii_df, "netvalue").values

    # Add DII if available
    if not dii_df.empty:
        dii_dates = pd.to_datetime(dii_df[date_col], dayfirst=True, errors="coerce")
        dii_net   = clean_series(dii_df, "netvalue")
        dii_series = pd.Series(dii_net.values, index=dii_dates, name="dii_net")
        result = result.join(dii_series, how="left")

    result = result.dropna(subset=["fii_net"])
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="first")]

    return result


# ── Manual download instructions ──────────────────────────────────────────────
MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MANUAL DOWNLOAD INSTRUCTIONS                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  If the automated download fails (NSE blocks bots), download manually:     ║
║                                                                            ║
║  1. Open: https://www.nseindia.com/reports-indices-historical-fii-dii       ║
║  2. Set Date Range: 01-Jan-2019 to today                                   ║
║  3. Click: FII / FPI (Equity only)                                         ║
║  4. Click Download CSV                                                     ║
║  5. Save file to: alpha-core/data/fii_dii_daily.csv                        ║
║                                                                            ║
║  Alternative source (pre-aggregated):                                      ║
║  https://ticker.finology.in/market/fiidii                                  ║
║  (Export CSV → save as alpha-core/data/fii_dii_daily.csv)                 ║
║                                                                            ║
║  Required columns in CSV:                                                  ║
║    Date | Buy Value (₹ Cr) | Sell Value (₹ Cr) | Net Value (₹ Cr)        ║
║                                                                            ║
║  The table3 script handles various NSE column name formats automatically.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("FII/DII DAILY DATA DOWNLOADER")
    print("=" * 80)

    print("\nAttempting automated download from NSE India...")
    raw_df = download_nse_fii(start="2019-01-01")

    if raw_df.empty:
        print(MANUAL_INSTRUCTIONS)
        exit(1)

    parsed = parse_nse_response(raw_df)

    if parsed.empty:
        print("Parsing failed.")
        print(MANUAL_INSTRUCTIONS)
        exit(1)

    parsed.to_csv(OUT_PATH)
    print(f"\n✓ Saved: {OUT_PATH}")
    print(f"  Records: {len(parsed)}")
    print(f"  Date range: {parsed.index[0].date()} → {parsed.index[-1].date()}")
    print(f"  Columns: {parsed.columns.tolist()}")
    print(f"\nSample (last 5 rows):")
    print(parsed.tail(5).to_string())
    print(f"\nNext step: python paper_analysis/table3_fii_leading_indicator.py")
