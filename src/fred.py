"""
step1_fred.py
=============
A guided walkthrough of fetching real macroeconomic data from the FRED API.

Run this file directly:
    python src/step1_fred.py

What you need first:
    1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
    2. Register (free, takes 2 minutes — just an email address)
    3. Copy your API key and paste it below where it says YOUR_KEY_HERE
       OR set it as an environment variable: export FRED_API_KEY="your_key"

What this file teaches:
    - What the FRED API is and how HTTP requests work
    - How to parse JSON responses into pandas DataFrames
    - What each economic series means and why we care about it
    - How to handle missing values in real data (FRED uses "." for missing)
    - How to resample daily data to quarterly frequency
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — EDIT THIS SECTION
# ─────────────────────────────────────────────────────────────────────────────

# Paste your FRED API key here, OR set the environment variable FRED_API_KEY.
# Never hardcode real API keys in code you share or submit — use env vars.
# For local development, hardcoding is fine.
FRED_API_KEY = os.getenv("FRED_API_KEY", "c8301212a7b41dd7d7ff87d6a91acde8")

# Date range for your study.
# 2018–2024 is ideal: covers pre-COVID, COVID shock, and recovery.
# This gives your model genuine variation to learn from.
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"

# Output directory — raw data gets saved here so you don't re-download
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  THE SERIES WE WANT AND WHY
# ─────────────────────────────────────────────────────────────────────────────
#
# FRED identifies every data series by a short code called a "series_id".
# You can look up any series at https://fred.stlouisfed.org — search for
# what you want and the series_id appears in the URL.
#
# Here is each series we use and its financial meaning:

FRED_SERIES = {

    # ── Credit spread series ──────────────────────────────────────────────────

    "hy_oas": "BAMLH0A0HYM2",
    # ICE BofA US High Yield Index Option-Adjusted Spread.
    # UNITS: FRED returns this in PERCENTAGE POINTS (e.g. 3.5 = 3.5% = 350 bps).
    # We multiply by 100 after fetching to convert to basis points.
    # "Option-adjusted" means it strips out the effect of callable bond
    # features to give a clean credit risk measure.
    # COVID peak (March 2020): ~8.5 on FRED = 850 bps.

    "ig_oas": "BAMLC0A4CBBBEY",
    # ICE BofA BBB US Corporate Index Effective Yield — used as IG OAS proxy.
    #
    # WHY THIS CHANGED:
    # The original series BAMLC0A0CM (IG OAS) intermittently returns HTTP 500
    # errors from FRED's server — a known instability with that specific series.
    # BAMLC0A4CBBBEY is the BBB-tier effective yield, which tracks IG spreads
    # closely and is reliably available. BBB is the largest IG rating bucket.
    # UNITS: percentage points — multiply by 100 to get bps.
    #
    # Alternative if this also fails: "BAMLC0A0CMEY" (IG effective yield)

    # ── Market fear and conditions ────────────────────────────────────────────

    "vix": "VIXCLS",
    # CBOE Volatility Index — the "fear gauge."
    # Measures the market's expectation of S&P 500 volatility over 30 days,
    # derived from options prices. VIX=20 is "normal." VIX=80 is panic.
    # Historically: VIX spikes during crises (GFC 2008: 80, COVID 2020: 82).
    # High VIX → investors demand higher premium for credit risk → wider spreads.
    # This is one of the most important macro features for CDS prediction.

    "risk_free": "DGS10",
    # 10-Year US Treasury Constant Maturity Rate (%).
    # The yield on 10-year US government bonds — considered risk-free because
    # the US government has never defaulted.
    # CDS spreads are measured on top of this: if Treasury yields 4% and
    # a BBB corporate bond yields 5.2%, the credit spread is 120 bps.
    # Rising risk-free rates can compress spreads (Treasuries become more
    # attractive, pushing bond prices down and yields up for all bonds).

    "usd_index": "DTWEXBGS",
    # Nominal Broad US Dollar Index.
    # Measures the dollar against a basket of trading partner currencies.
    # Stronger dollar → tighter financial conditions globally → can widen
    # spreads for companies with foreign-currency revenues or debts.
    # This is an optional feature but financially well-motivated.
}


# ─────────────────────────────────────────────────────────────────────────────
#  CORE FUNCTION: FETCH A SINGLE SERIES FROM FRED
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, api_key: str,
                      start: str, end: str) -> pd.Series:
    """
    Fetches one time series from the FRED API and returns a pandas Series.

    HOW THE API CALL WORKS:
    -----------------------
    We construct a URL with query parameters and send an HTTP GET request.
    The `requests` library handles the network connection.

    The FRED API returns JSON that looks like this:
    {
        "observations": [
            {"date": "2018-01-02", "value": "14.87"},
            {"date": "2018-01-03", "value": "14.61"},
            {"date": "2018-01-04", "value": "."},   ← missing value!
            ...
        ]
    }

    We parse this into a pandas Series indexed by date.

    PARAMETERS:
    -----------
    series_id : str
        The FRED series code (e.g. "VIXCLS" for VIX)
    api_key : str
        Your FRED API key
    start, end : str
        Date strings in "YYYY-MM-DD" format

    RETURNS:
    --------
    pd.Series with DatetimeIndex and float values
    """

    # The base URL for FRED observations
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Parameters are sent as query string: ?key1=val1&key2=val2
    # requests handles the formatting for you
    params = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",       # we want JSON, not XML
        "observation_start": start,
        "observation_end":   end,
    }

    print(f"    Requesting: {BASE_URL}?series_id={series_id}&...")

    # requests.get() sends the HTTP GET request and waits for a response.
    # timeout=30 means: if the server doesn't respond in 30 seconds, give up.
    response = requests.get(BASE_URL, params=params, timeout=30)

    # raise_for_status() checks the HTTP status code.
    # 200 = OK, 400 = bad request, 403 = forbidden (bad API key), 404 = not found
    # If not 200, this raises an exception with the error code — much clearer
    # than silently getting empty data.
    response.raise_for_status()

    # response.json() parses the JSON text into a Python dict.
    # We then dig into the "observations" list.
    observations = response.json()["observations"]

    print(f"    Received {len(observations)} observations")

    # Build a dict of {date_string: float_value}.
    # FRED uses "." to represent missing values (e.g. weekends, holidays).
    # We skip those with the `if obs["value"] != "."` condition.
    # Values come as strings — we cast to float.
    data_dict = {}
    for obs in observations:
        if obs["value"] != ".":
            data_dict[obs["date"]] = float(obs["value"])

    # Create a pandas Series from the dict.
    # index is the dates (as strings initially), values are floats.
    series = pd.Series(data_dict, name=series_id)

    # Convert the string dates to proper pandas Timestamp objects.
    # This enables date-based operations like resampling, slicing, etc.
    series.index = pd.to_datetime(series.index)

    # Sort by date (FRED should return them in order, but be safe)
    series = series.sort_index()

    return series


# ─────────────────────────────────────────────────────────────────────────────
#  FETCH ALL SERIES AND COMBINE INTO A DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_macro(api_key: str, start: str, end: str,
                    force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches all FRED series, combines into one DataFrame, and resamples
    to quarterly frequency.

    WHY QUARTERLY?
    --------------
    Company financial data (balance sheets, income statements) is published
    quarterly. If we kept macro data daily, we'd have a mismatch:
    - 1 row of company fundamentals (leverage ratio, ROA, etc.) per quarter
    - 65 rows of macro data (VIX, spreads) per quarter
    Combining these doesn't make sense. So we aggregate macro to quarterly
    by taking the mean over each quarter — this smooths out daily noise
    and gives us one macro row per company-quarter.

    CACHING:
    --------
    We save to CSV after downloading. On subsequent runs, we load from CSV
    rather than hitting the API again. This is important because:
    1. FRED has rate limits (they'll block you if you hammer them)
    2. Your data should be reproducible — the same CSV gives the same results
    3. It's faster — disk reads are much faster than HTTP requests

    force_refresh=True will re-download even if the cache exists.
    """
    cache_path = RAW_DIR / "macro_fred.csv"

    if cache_path.exists() and not force_refresh:
        print(f"  [CACHE] Loading macro data from {cache_path}")
        print(f"  (Pass force_refresh=True to re-download from FRED)")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print("  Fetching macro data from FRED API...")
    print(f"  Date range: {start} → {end}\n")

    # ── Fallback series: if a primary series fails, try these alternatives ────
    # FRED occasionally returns 500 errors for specific series even with a valid
    # API key — it's a server-side instability, not your fault. We define
    # fallbacks so one bad series doesn't break the whole download.
    FALLBACKS = {
        "ig_oas": ["BAMLC0A0CMEY", "BAMLC0A4CBBBEY", "BAMLC0A0CM"],
        # BAMLC0A0CMEY = IG effective yield (closely tracks OAS)
        # BAMLC0A4CBBBEY = BBB effective yield (our primary choice)
        # BAMLC0A0CM = original OAS series (intermittently unavailable)
    }

    # ── These series are in PERCENT, need ×100 to convert to basis points ────
    # FRED reports BofA spread series as percentage values:
    # e.g., HY OAS = 3.5 on FRED → 3.5% → 350 basis points
    # VIX is already in its native units (VIX points, not percent).
    # risk_free (DGS10) is in percent — we leave it as percent (conventional).
    # usd_index is an index level — leave as-is.
    MULTIPLY_BY_100 = {"hy_oas", "ig_oas"}

    all_series = {}

    for friendly_name, series_id in FRED_SERIES.items():
        print(f"  Fetching '{friendly_name}' ({series_id})...")

        # Build a list of series IDs to try: primary first, then fallbacks
        ids_to_try = [series_id] + FALLBACKS.get(friendly_name, [])
        ids_to_try = list(dict.fromkeys(ids_to_try))  # deduplicate, preserve order

        fetched = False
        for attempt_id in ids_to_try:
            if attempt_id != series_id:
                print(f"    ↳ Trying fallback: {attempt_id}")
            try:
                series = fetch_fred_series(attempt_id, api_key, start, end)

                # ── Units conversion ──────────────────────────────────────────
                # BofA spread series come in percent (e.g. 3.5 = 350 bps).
                # Multiply by 100 to work in basis points throughout.
                if friendly_name in MULTIPLY_BY_100:
                    series = series * 100
                    print(f"    (converted from % to bps: ×100)")

                all_series[friendly_name] = series
                print(f"    ✓ {len(series)} observations | "
                      f"range: {series.min():.1f} → {series.max():.1f} | "
                      f"latest: {series.index[-1].date()}")
                fetched = True
                break   # success — stop trying fallbacks

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                print(f"    ✗ HTTP {status} for {attempt_id}")
                if status == 400:
                    print(f"      Bad series ID — skipping this fallback")
                    break   # 400 = bad request, no point retrying
                elif status == 403:
                    print(f"      API key rejected. Double-check your key.")
                    break
                # 500 = server error — try next fallback
                time.sleep(1)

            except requests.exceptions.ConnectionError:
                print(f"    ✗ No internet connection.")
                break

            except Exception as e:
                print(f"    ✗ Error: {e}")

        if not fetched:
            print(f"    ⚠ Could not fetch '{friendly_name}' — "
                  f"will be missing from dataset")

        time.sleep(0.5)   # rate limiting between series

    if not all_series:
        raise RuntimeError(
            "No data was fetched. Check your API key and internet connection.\n"
            f"Your current key: '{api_key[:8]}...'"
        )

    print(f"\n  Successfully fetched: {list(all_series.keys())}")
    missing = [k for k in FRED_SERIES if k not in all_series]
    if missing:
        print(f"  Missing (will be NaN in dataset): {missing}")

    # pd.DataFrame() from a dict of Series aligns them by index (date).
    # Dates that exist in one series but not another get NaN automatically.
    df_daily = pd.DataFrame(all_series)

    print(f"\n  Combined daily DataFrame: {df_daily.shape}")
    print(f"  Date range: {df_daily.index[0].date()} → {df_daily.index[-1].date()}")
    print(f"  Missing values per column:")
    for col in df_daily.columns:
        n_missing = df_daily[col].isna().sum()
        pct = n_missing / len(df_daily) * 100
        print(f"    {col}: {n_missing} ({pct:.1f}%)")

    # ── Resample from daily to quarterly ─────────────────────────────────────
    #
    # resample("QE") groups observations by quarter-end.
    # "QE" = quarter end: March 31, June 30, September 30, December 31.
    # .mean() takes the average value over each quarter.
    #
    # Why mean and not last()?
    # Taking the last observation (e.g. VIX on Dec 31) gives you a single
    # snapshot that might be atypical. The mean over Q4 better represents
    # "what conditions were like in Q4." For stress-testing you might use
    # max() instead to capture the worst conditions in the quarter.

    df_quarterly = df_daily.resample("QE").mean()

    print(f"\n  Resampled to quarterly: {df_quarterly.shape}")
    print(f"  Quarters: {df_quarterly.index[0].date()} → {df_quarterly.index[-1].date()}")

    # Save to CSV
    df_quarterly.to_csv(cache_path)
    print(f"\n  Saved to {cache_path}")

    return df_quarterly


# ─────────────────────────────────────────────────────────────────────────────
#  INSPECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value, fmt: str, fallback: str = "N/A") -> str:
    """
    Safely formats a value that might be missing (NaN or the string 'N/A').

    WHY THIS EXISTS:
    ----------------
    When a FRED series fails to download, its column is absent from the
    DataFrame. Calling row.get('ig_oas', 'N/A') returns the STRING 'N/A',
    not a float. Trying to format a string with :.0f (a float format code)
    then crashes with: ValueError: Unknown format code 'f' for object of type 'str'

    This helper checks whether the value is actually a number before formatting.
    If it's missing or not a number, it returns the fallback string instead.

    Example:
        _fmt(3.5, ".1f")      → "3.5"
        _fmt("N/A", ".1f")    → "N/A"
        _fmt(float('nan'), ".0f") → "N/A"
    """
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return fallback
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback


def inspect_macro_data(df: pd.DataFrame) -> None:
    """
    Prints a thorough inspection of the macro DataFrame.
    Always do this before using data — surprises here cause bugs later.

    Handles missing columns gracefully (e.g. if ig_oas failed to download).
    """
    print("\n" + "="*60)
    print("MACRO DATA INSPECTION")
    print("="*60)

    print(f"\nShape: {df.shape[0]} quarters × {df.shape[1]} features")
    print(f"\nColumns: {list(df.columns)}")

    print("\n── First 4 rows (oldest data) ──")
    print(df.head(4).to_string())

    print("\n── Last 4 rows (most recent) ──")
    print(df.tail(4).to_string())

    print("\n── Descriptive statistics ──")
    print(df.describe().round(2).to_string())

    print("\n── Missing value check ──")
    missing = df.isna().sum()
    if missing.any():
        print("WARNING: Missing values found:")
        print(missing[missing > 0])
    else:
        print("✓ No missing values in quarterly data")

    # ── Units sanity check ────────────────────────────────────────────────────
    # After our ×100 conversion, spread series should be in hundreds, not single digits
    print("\n── Units sanity check ──")
    for col in ["hy_oas", "ig_oas"]:
        if col in df.columns:
            median_val = df[col].median()
            if median_val < 20:
                print(f"  ⚠ WARNING: {col} median is {median_val:.2f} — "
                      f"looks like it's still in percent, not bps!")
                print(f"    Expected ~300-500 bps for HY, ~100-150 bps for IG")
            else:
                print(f"  ✓ {col}: median {median_val:.0f} bps — looks correct")

    # ── Notable periods check ─────────────────────────────────────────────────
    # .get() on a pandas Series returns NaN if the key is missing — safe to use
    # We use our _fmt() helper to handle NaN gracefully in format strings

    def get_val(row, col):
        """Get a value from a Series row, returning NaN if column missing."""
        return row[col] if col in row.index else float("nan")

    print("\n── Notable periods ──")
    covid_q = df.loc["2020-03-31"] if "2020-03-31" in df.index else None
    if covid_q is not None:
        print("COVID shock (Q1 2020):")
        print(f"  VIX:    {_fmt(get_val(covid_q,'vix'), '.1f')} (normal ~15)")
        print(f"  HY OAS: {_fmt(get_val(covid_q,'hy_oas'), '.0f')} bps (normal ~350 bps)")
        print(f"  IG OAS: {_fmt(get_val(covid_q,'ig_oas'), '.0f')} bps (normal ~100 bps)")

    precovid_q = df.loc["2019-12-31"] if "2019-12-31" in df.index else None
    if precovid_q is not None:
        print("Pre-COVID baseline (Q4 2019):")
        print(f"  VIX:    {_fmt(get_val(precovid_q,'vix'), '.1f')}")
        print(f"  HY OAS: {_fmt(get_val(precovid_q,'hy_oas'), '.0f')} bps")
        print(f"  IG OAS: {_fmt(get_val(precovid_q,'ig_oas'), '.0f')} bps")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — runs when you execute this file directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  STEP 1: FETCHING MACRO DATA FROM FRED")
    print("="*60 + "\n")

    # Check if API key has been set
    if FRED_API_KEY == "YOUR_KEY_HERE":
        print("ERROR: You haven't set your FRED API key yet.")
        print("\nTo get a free key:")
        print("  1. Go to https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  2. Register with just an email address")
        print("  3. Copy the key and either:")
        print("     a) Paste it into this file where it says YOUR_KEY_HERE")
        print("     b) Run: export FRED_API_KEY='your_actual_key'")
        print("        Then run this script again")
        exit(1)

    # Fetch data (uses cache if already downloaded)
    macro_df = fetch_all_macro(FRED_API_KEY, START_DATE, END_DATE)

    # Inspect what we got
    inspect_macro_data(macro_df)

    print("\n✓ Step 1 complete. Run step2_equity.py next.")
    print(f"  Data saved to: {RAW_DIR / 'macro_fred.csv'}")