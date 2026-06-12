"""
FII Daily Data — Direct NSDL Production Archive Scraper
=======================================================
Bypasses nselib entirely. POSTs directly to:
  https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx

Why: nselib (a) prefers pilot server which lacks pre-2021 data,
     (b) classifies tables only by exact "investment route" header
     which breaks for older NSDL HTML formats.

This script:
  1. Forces PRODUCTION URL always
  2. Refreshes VIEWSTATE/cookies every REFRESH_EVERY requests
  3. Tries 3 parsing strategies for old + new HTML formats
  4. Saves incrementally every SAVE_EVERY records
  5. Skips already-downloaded dates (resume-safe)

Output: alpha-core/data/fii_nsdl_direct.csv
  columns: date (index), fii_buy, fii_sell, fii_net  [Rs. Crore]

Runtime: ~60-90 minutes for 1939 dates (1.5s delay + 2 HTTP calls/date)
"""

import re
import time
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta, date
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH   = DATA_DIR / "fii_nsdl_direct.csv"

ARCHIVE_URL     = "https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx"
DELAY_SECS      = 1.5   # polite delay between requests
REFRESH_EVERY   = 40    # refresh VIEWSTATE + session every N dates
SAVE_EVERY      = 50    # write CSV every N dates
START_DATE      = "2019-01-01"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
    "Referer": ARCHIVE_URL,
}

HOLIDAY_SIGNALS = [
    "No Data To Display",
    "no data to display",
    "invalid date",
    "please enter valid date",
    "no records",
]

# ── Session & VIEWSTATE ───────────────────────────────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_viewstate(session: requests.Session) -> dict:
    """GET the archive page and extract ASP.NET hidden fields."""
    r = session.get(ARCHIVE_URL, timeout=30)
    r.raise_for_status()
    fields = {}
    for f in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"):
        m = re.search(rf'name="{f}"\s+id="{f}"\s+value="([^"]*)"', r.text)
        fields[f] = m.group(1) if m else ""
    if not fields.get("__VIEWSTATE"):
        raise RuntimeError(
            f"Could not extract __VIEWSTATE. Response snippet:\n{r.text[:500]}"
        )
    return fields


# ── Fetch one date ────────────────────────────────────────────────────────────
def fetch_date_html(session: requests.Session, fields: dict, dt: date) -> str:
    """POST to NSDL archive for a specific date. Returns response HTML."""
    date_str = dt.strftime("%d-%b-%Y")   # e.g., "05-Jan-2020"
    payload = {
        "__EVENTTARGET":       "btnSubmit1",
        "__EVENTARGUMENT":     "",
        "__VIEWSTATE":         fields["__VIEWSTATE"],
        "__VIEWSTATEGENERATOR": fields["__VIEWSTATEGENERATOR"],
        "__EVENTVALIDATION":   fields["__EVENTVALIDATION"],
        "txtDate":             date_str,
        "hdnDate":             date_str,
        "HdnValexceldata":     "",
        "hdnFlag":             "",
    }
    r = session.post(ARCHIVE_URL, data=payload, timeout=30)
    r.raise_for_status()
    return r.text


# ── Parse HTML → (buy, sell, net) ────────────────────────────────────────────
def _flatten_cols(t: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns to single strings."""
    t = t.copy()
    if t.columns.nlevels > 1:
        t.columns = [
            " ".join(str(x) for x in c if str(x) != "nan" and "Unnamed" not in str(x)).strip()
            for c in t.columns
        ]
    else:
        t.columns = [str(c).strip() for c in t.columns]
    return t


def _to_float(val) -> float:
    """Parse numeric string like '1,234.56' or '(456.78)' → float."""
    try:
        s = str(val).strip().replace(",", "")
        negative = s.startswith("(") and s.endswith(")")
        s = s.strip("()").replace("Rs.", "").replace("Rs", "").strip()
        v = float(s)
        return -v if negative else v
    except Exception:
        return np.nan


def _find_col(cols, keywords):
    """Return first column name that contains any keyword."""
    for kw in keywords:
        for c in cols:
            if kw in c.lower():
                return c
    return None


def _extract_equity_row(t: pd.DataFrame):
    """
    Given a DataFrame (already flat-column), find the 'Equity' row
    and extract (buy, sell, net) in Rs. Crore.
    Returns (nan, nan, nan) if extraction fails.
    """
    # Find which column holds asset class / category / route
    cat_col = _find_col(t.columns, [
        "asset class", "category", "type", "instrument",
        "debt", "equity",    # older format: column titled with category names
        "route", "investment route",
    ])

    if cat_col is None:
        # Try any column whose values contain "equity"
        for col in t.columns:
            try:
                mask = t[col].astype(str).str.lower().str.contains("equity", na=False)
                if mask.any():
                    cat_col = col
                    break
            except Exception:
                pass

    if cat_col is None:
        return np.nan, np.nan, np.nan

    # Equity row (exclude mutual fund / hybrid / AIF)
    eq_mask = (
        t[cat_col].astype(str).str.lower().str.contains("equity", na=False)
        & ~t[cat_col].astype(str).str.lower().str.contains(
            r"mutual|hybrid|aif|scheme|fund|debt|credit|corporate|etf", na=False, regex=True
        )
    )
    eq_rows = t[eq_mask]
    if eq_rows.empty:
        return np.nan, np.nan, np.nan

    row = eq_rows.iloc[0]

    buy_col  = _find_col(t.columns, ["purchase", "buy", "gross_p", "gross p"])
    sell_col = _find_col(t.columns, ["sale", "sell", "gross_s", "gross s"])
    net_col  = _find_col(t.columns, ["net invest", "net_invest", "net"])

    buy  = _to_float(row[buy_col])  if buy_col  else np.nan
    sell = _to_float(row[sell_col]) if sell_col else np.nan
    net  = _to_float(row[net_col])  if net_col  else np.nan

    if np.isnan(net) and not (np.isnan(buy) or np.isnan(sell)):
        net = buy - sell

    return buy, sell, net


def parse_equity(html: str):
    """
    3-strategy parser for NSDL archive HTML.

    Returns:
      (None, None, None)          → holiday (no data signal)
      (float, float, float)       → buy, sell, net in Rs. Crore (may be nan)
    """
    # Holiday check
    for sig in HOLIDAY_SIGNALS:
        if sig.lower() in html.lower():
            return None, None, None

    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return np.nan, np.nan, np.nan  # no tables parsed

    # ── Strategy 1: table with "investment route" column (new format) ──
    for t in tables:
        flat = _flatten_cols(t)
        if any("investment route" in c.lower() for c in flat.columns):
            buy, sell, net = _extract_equity_row(flat)
            if not (np.isnan(buy) and np.isnan(sell) and np.isnan(net)):
                return buy, sell, net

    # ── Strategy 2: table with "equity" as a VALUE in any column ──────
    for t in tables:
        flat = _flatten_cols(t)
        buy, sell, net = _extract_equity_row(flat)
        if not (np.isnan(buy) and np.isnan(sell) and np.isnan(net)):
            return buy, sell, net

    # ── Strategy 3: widest numeric table fallback ──────────────────────
    # Take the table with the most numeric columns; first row with 'equity'
    best = None
    for t in tables:
        flat = _flatten_cols(t)
        num_cols = flat.select_dtypes(include="number").shape[1]
        if best is None or num_cols > best[0]:
            best = (num_cols, flat)
    if best and best[0] >= 3:
        t = best[1]
        buy, sell, net = _extract_equity_row(t)
        if not (np.isnan(buy) and np.isnan(sell) and np.isnan(net)):
            return buy, sell, net

    return np.nan, np.nan, np.nan


# ── Trading days ─────────────────────────────────────────────────────────────
def trading_days(start: str, end: str = None):
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.today().date() if end is None else datetime.strptime(end, "%Y-%m-%d").date()
    days = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:  # Mon–Fri
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ── Load existing progress ────────────────────────────────────────────────────
def load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH, index_col=0, parse_dates=True)
        print(f"  Loaded {len(df)} existing rows from {OUT_PATH.name}")
        return df
    return pd.DataFrame(columns=["fii_buy", "fii_sell", "fii_net"])


# ── Main download loop ────────────────────────────────────────────────────────
def download_all():
    days = trading_days(START_DATE)
    print(f"Target: {len(days)} weekdays from {START_DATE} to today")

    existing = load_existing()
    done_dates = set(existing.index.strftime("%Y-%m-%d").tolist()) if not existing.empty else set()
    remaining  = [d for d in days if d.strftime("%Y-%m-%d") not in done_dates]
    print(f"Already done: {len(done_dates)} | Remaining: {len(remaining)}\n")

    if not remaining:
        print("All dates already downloaded!")
        return existing

    records  = []
    holidays = 0
    errors   = 0
    session  = make_session()
    fields   = None

    for i, dt in enumerate(remaining):
        # Refresh VIEWSTATE periodically
        if i % REFRESH_EVERY == 0:
            try:
                fields = get_viewstate(session)
                print(f"  [refresh] Got VIEWSTATE at i={i} ({dt})")
            except Exception as e:
                print(f"  [refresh ERROR] {e} — retrying with new session...")
                time.sleep(5)
                session = make_session()
                try:
                    fields = get_viewstate(session)
                    print(f"  [refresh OK] Got VIEWSTATE with new session")
                except Exception as e2:
                    print(f"  [refresh FATAL] Cannot get VIEWSTATE: {e2}")
                    break

        dt_str = dt.strftime("%Y-%m-%d")
        try:
            html = fetch_date_html(session, fields, dt)
            buy, sell, net = parse_equity(html)

            if buy is None:  # holiday
                holidays += 1
                records.append({"date": dt_str, "fii_buy": np.nan, "fii_sell": np.nan, "fii_net": np.nan, "status": "holiday"})
            else:
                records.append({"date": dt_str, "fii_buy": buy, "fii_sell": sell, "fii_net": net, "status": "ok"})

            if i % 25 == 0 or i < 5:
                tag = "HOLIDAY" if buy is None else f"net={net:.1f}" if not np.isnan(net) else "net=NaN"
                print(f"  [{i+1:4d}/{len(remaining)}] {dt_str}  {tag}")

        except requests.exceptions.HTTPError as e:
            errors += 1
            records.append({"date": dt_str, "fii_buy": np.nan, "fii_sell": np.nan, "fii_net": np.nan, "status": f"http_err_{e.response.status_code}"})
            print(f"  [{i+1:4d}/{len(remaining)}] {dt_str}  HTTP ERROR {e.response.status_code}")
            time.sleep(5)
            # Refresh session on HTTP errors
            try:
                session = make_session()
                fields = get_viewstate(session)
            except Exception:
                pass

        except Exception as e:
            errors += 1
            records.append({"date": dt_str, "fii_buy": np.nan, "fii_sell": np.nan, "fii_net": np.nan, "status": f"err: {str(e)[:60]}"})
            if i % 50 == 0:
                print(f"  [{i+1:4d}/{len(remaining)}] {dt_str}  ERROR: {e}")

        time.sleep(DELAY_SECS)

        # Incremental save
        if (i + 1) % SAVE_EVERY == 0 or (i + 1) == len(remaining):
            batch = pd.DataFrame(records)
            if not batch.empty:
                batch = batch.set_index("date")
                batch.index = pd.to_datetime(batch.index)
                combined = pd.concat([existing, batch]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.to_csv(OUT_PATH)
                n_valid = combined["fii_net"].notna().sum()
                print(f"\n  → Saved {len(combined)} rows ({n_valid} with data) → {OUT_PATH.name}")
                existing = combined
                records = []

    print(f"\n{'='*60}")
    print(f"Done! holidays={holidays} | errors={errors}")
    return existing


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FII DAILY SCRAPER — Direct NSDL Production Archive POST")
    print("=" * 60)
    print(f"Target URL: {ARCHIVE_URL}")
    print(f"Output:     {OUT_PATH}\n")

    # Sanity check: fetch one date before starting the loop
    print("--- Sanity check: 2020-10-15 ---")
    try:
        sess = make_session()
        flds = get_viewstate(sess)
        print("  ✓ VIEWSTATE obtained")
        html = fetch_date_html(sess, flds, date(2020, 10, 15))
        buy, sell, net = parse_equity(html)
        if buy is None:
            print("  → Holiday signal (or no data)")
        elif np.isnan(net):
            # Print tables for debugging
            try:
                tables = pd.read_html(StringIO(html))
                print(f"  → Parsed {len(tables)} tables; columns of each:")
                for j, t in enumerate(tables):
                    t2 = _flatten_cols(t)
                    print(f"     Table {j}: {list(t2.columns)}")
            except Exception as te:
                print(f"  → No tables parseable: {te}")
                print(f"  → HTML snippet:\n{html[500:1500]}")
        else:
            print(f"  ✓ 2020-10-15: buy={buy:.1f}  sell={sell:.1f}  net={net:.1f} Rs.Cr")
    except Exception as e:
        print(f"  ✗ Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nFix the issue above before running the full download.")
        exit(1)

    print("\n" + "=" * 60)
    ans = input("Sanity check passed. Start full download? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted.")
        exit(0)

    result = download_all()

    # Final report
    if result is not None and not result.empty:
        valid = result["fii_net"].dropna()
        print(f"\nFinal dataset: {len(result)} rows, {len(valid)} with data")
        print(f"Date range: {result.index[0].date()} → {result.index[-1].date()}")
        print(f"\nSample (non-NaN):\n{result[result['fii_net'].notna()].head(10)}")
        print(f"\n✓ Saved: {OUT_PATH}")
        print(f"\nNext step: python paper_analysis/table3_fii_leading_indicator.py")
