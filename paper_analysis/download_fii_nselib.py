"""
FII Daily Data via nselib (NSDL FPI Investment Activity)
=========================================================

Uses the `nselib` library to pull NSDL's daily FPI equity net investment
for each trading day from 2019-01-01 to today.

INSTALL FIRST (run once in VS Code terminal):
    pip install nselib

Then run:
    cd alpha-core
    python paper_analysis/download_fii_nselib.py

Output: alpha-core/data/fii_dii_daily.csv
  columns: date, fii_net, fii_buy, fii_sell

Runtime: ~15-20 minutes (1700+ days × 0.4s each).
Progress prints every 50 days.
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "fii_dii_daily.csv"

# ── Step 1: Import nselib ─────────────────────────────────────────────────────
# Monkey-patch nselib 2.5.1 bug: NSEdataNotFound missing from libutil
try:
    import nselib.libutil as _lu
    if not hasattr(_lu, 'NSEdataNotFound'):
        class NSEdataNotFound(Exception):
            pass
        _lu.NSEdataNotFound = NSEdataNotFound
    from nselib import cash_market
    print("✓ nselib loaded")
except Exception as e:
    print(f"✗ nselib import failed: {type(e).__name__}: {e}")
    exit(1)


# ── Step 2: Detect output format of nsdl_fpi_investment_activity ──────────────
def detect_format():
    """Fetch one recent date and print the raw output so format is visible."""
    print("\n--- FORMAT DETECTION: fetching 2025-10-30 ---")
    try:
        df = cash_market.nsdl_fpi_investment_activity(trade_date='30-10-2025')
        if df is None:
            print("  Returned None")
            return None
        print(f"  Type: {type(df)}")
        if isinstance(df, pd.DataFrame):
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  dtypes:\n{df.dtypes}")
            print(f"\nFull sample:\n{df.to_string()}")
        elif isinstance(df, list):
            print(f"  Length: {len(df)}")
            print(f"  First item: {df[0] if df else 'empty'}")
        else:
            print(f"  Value: {df}")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return None


# ── Step 3: Parse one day's response → (buy, sell, net) ──────────────────────
def parse_fpi_equity(raw) -> tuple[float, float, float]:
    """
    Extract FPI *equity* gross buy, sell, net from one day's NSDL response.

    NSDL response format (as of 2024/2025):
      DataFrame with columns like:
        'Category', 'Gross Purchases (Rs. Crore)', 'Gross Sales (Rs. Crore)', 'Net Investment (Rs. Crore)'
      OR columns like:
        'category', 'gross_purchases', 'gross_sales', 'net_investment'
      One row per asset class: Equity, Debt-General Limit, Debt-VRR, Debt-FAR, Hybrid, ...

    Returns: (buy, sell, net) as floats in ₹ Crore, or (nan, nan, nan) on failure.
    """
    import numpy as np

    if raw is None:
        return np.nan, np.nan, np.nan

    if isinstance(raw, list):
        # Convert list of dicts to DataFrame
        try:
            raw = pd.DataFrame(raw)
        except Exception:
            return np.nan, np.nan, np.nan

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return np.nan, np.nan, np.nan

    # Normalise column names: lowercase, strip spaces/special chars
    df = raw.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Identify category column
    cat_col = next(
        (c for c in df.columns if "category" in c or "type" in c or "instrument" in c),
        None
    )

    if cat_col is None:
        # No category column — might be a single-row response, try first numeric cols
        numeric = df.select_dtypes(include="number")
        if len(numeric.columns) >= 3:
            buy, sell, net = (
                pd.to_numeric(numeric.iloc[0, 0], errors="coerce"),
                pd.to_numeric(numeric.iloc[0, 1], errors="coerce"),
                pd.to_numeric(numeric.iloc[0, 2], errors="coerce"),
            )
            return buy, sell, net
        return np.nan, np.nan, np.nan

    # Identify 'Equity' row (case-insensitive, partial match)
    equity_mask = df[cat_col].astype(str).str.lower().str.contains("equity", na=False)
    # Exclude "hybrid equity", "mutual fund equity schemes" — keep the primary Equity row
    equity_mask &= ~df[cat_col].astype(str).str.lower().str.contains("mutual|hybrid|aif|scheme", na=False)

    equity_rows = df[equity_mask]
    if equity_rows.empty:
        # Fallback: take first row
        equity_rows = df.iloc[[0]]

    row = equity_rows.iloc[0]

    # Identify buy/sell/net columns
    def find_col(df_cols, keywords):
        for kw in keywords:
            for c in df_cols:
                if kw in c:
                    return c
        return None

    buy_col  = find_col(df.columns, ["buy", "purchase", "gross_p"])
    sell_col = find_col(df.columns, ["sell", "sale", "gross_s"])
    net_col  = find_col(df.columns, ["net"])

    def to_float(val):
        import numpy as np
        try:
            return float(str(val).replace(",", "").strip())
        except Exception:
            return np.nan

    buy  = to_float(row[buy_col])  if buy_col  else float("nan")
    sell = to_float(row[sell_col]) if sell_col else float("nan")
    net  = to_float(row[net_col])  if net_col  else float("nan")

    # If net not found but buy/sell available, compute it
    import numpy as np
    if np.isnan(net) and not (np.isnan(buy) or np.isnan(sell)):
        net = buy - sell

    return buy, sell, net


# ── Step 4: Generate all weekdays ─────────────────────────────────────────────
def trading_days(start="2019-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    days = []
    current = s
    while current <= e:
        if current.weekday() < 5:  # Mon–Fri
            days.append(current)
        current += timedelta(days=1)
    return days


# ── Step 5: Main download loop ────────────────────────────────────────────────
def download_all(start="2019-01-01"):
    days = trading_days(start)
    print(f"\nFetching {len(days)} weekdays from {start} to today...")
    print("Expected runtime: ~15-20 minutes\n")

    records = []
    failed_dates = []

    for i, dt in enumerate(days):
        date_str = dt.strftime("%d-%m-%Y")
        try:
            raw = cash_market.nsdl_fpi_investment_activity(trade_date=date_str)
            buy, sell, net = parse_fpi_equity(raw)

            records.append({
                "date":     dt.strftime("%Y-%m-%d"),
                "fii_buy":  buy,
                "fii_sell": sell,
                "fii_net":  net,
            })

            if i % 50 == 0 or i < 3:
                print(f"  [{i:4d}/{len(days)}] {date_str}  net={net:.2f}" if net == net else
                      f"  [{i:4d}/{len(days)}] {date_str}  net=NaN")

        except Exception as e:
            failed_dates.append(date_str)
            if i % 100 == 0:
                print(f"  [{i:4d}/{len(days)}] {date_str}  ERROR: {e}")

        time.sleep(0.4)   # ~0.4s per call to be polite

    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])
    result = result.set_index("date").sort_index()

    # Drop days where net is NaN (holidays, weekends that slipped through, etc.)
    n_before = len(result)
    result = result.dropna(subset=["fii_net"])
    n_after = len(result)
    print(f"\nDropped {n_before - n_after} NaN rows (holidays / no data)")

    if failed_dates:
        print(f"\nFailed completely on {len(failed_dates)} dates:")
        for d in failed_dates[:20]:
            print(f"  {d}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("FII DAILY DATA FETCHER — via nselib / NSDL FPI Investment Activity")
    print("=" * 70)

    # Quick format check on one date first
    sample = detect_format()

    if sample is None:
        print("\n✗ Could not fetch sample. Check internet connection and nselib install.")
        exit(1)

    print("\n" + "=" * 70)
    input("Format looks OK? Press Enter to start full download, Ctrl+C to abort.\n")

    result = download_all(start="2019-01-01")

    print(f"\n{'='*70}")
    print(f"Done! {len(result)} trading days with FII data.")
    print(f"Date range: {result.index[0].date()} → {result.index[-1].date()}")
    print(f"\nSample (last 5):\n{result.tail()}")

    result.to_csv(OUT_PATH)
    print(f"\n✓ Saved to: {OUT_PATH}")
    print(f"\nNext step: python paper_analysis/table3_fii_leading_indicator.py")
