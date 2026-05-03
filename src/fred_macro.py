# STEP 1: FETCHING MACRO DATA FROM FRED

import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Hide API Key
FRED_API_KEY = os.getenv("FRED_API_KEY", "c8301212a7b41dd7d7ff87d6a91acde8")

# Date range
# 2018–2024 is ideal: covers pre-COVID, COVID shock, and recovery.
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"

# Output directory
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


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
    # BAMLC0A4CBBBEY is the BBB-tier effective yield, which tracks IG spreads
    # closely and is reliably available. BBB is the largest IG rating bucket.
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
}


def fetch_fred_series(
    series_id: str,
    api_key: str,
    start: str,
    end: str
    ) -> pd.Series:

    #Fetches one time series from the FRED API and returns a pandas Series.
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",       # we want JSON, not XML
        "observation_start": start,
        "observation_end":   end,
    }

    print(f"    Requesting: {BASE_URL}?series_id={series_id}&...")

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    # Parse JSON into python dict
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
    series = pd.Series(data_dict, name=series_id)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    return series


def fetch_all_macro(
    api_key: str,
    start: str,
    end: str,
    force_refresh: bool = False
    ) -> pd.DataFrame:

    cache_path = RAW_DIR / "macro_fred.csv"

    # Getting the data from cache
    if cache_path.exists() and not force_refresh:
        print(f"  [CACHE] Loading macro data from {cache_path}")
        print(f"  (Pass force_refresh=True to re-download from FRED)")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    # Fetching the data from the FRED API

    print("  Fetching macro data from FRED API...")
    print(f"  Date range: {start} → {end}\n")

    # Fallback series: if a primary series fails, try these alternatives 
    FALLBACKS = {
        "ig_oas": ["BAMLC0A0CMEY", "BAMLC0A4CBBBEY", "BAMLC0A0CM"],
        # BAMLC0A0CMEY = IG effective yield (closely tracks OAS)
        # BAMLC0A4CBBBEY = BBB effective yield (our primary choice)
        # BAMLC0A0CM = original OAS series (intermittently unavailable)
    }

    # These series are in PERCENT, need ×100 to convert to basis points
    MULTIPLY_BY_100 = {"hy_oas", "ig_oas"}


    all_series = {}

    for name, series_id in FRED_SERIES.items():

        print(f"  Fetching '{name}' ({series_id})...")

        # Build a list of series IDs to try: primary first, then fallbacks
        ids_to_try = [series_id] + FALLBACKS.get(name, [])
        ids_to_try = list(dict.fromkeys(ids_to_try))

        fetched = False
        for attempt_id in ids_to_try:
            if attempt_id != series_id:
                print(f"    ↳ Trying fallback: {attempt_id}")
            try:
                series = fetch_fred_series(attempt_id, api_key, start, end)

                if name in MULTIPLY_BY_100:
                    series = series * 100
                    print(f"    (converted from % to bps: ×100)")

                all_series[name] = series

                print(f"    ✓ {len(series)} observations | "
                      f"range: {series.min():.1f} → {series.max():.1f} | "
                      f"latest: {series.index[-1].date()}")
                fetched = True
                break

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                print(f"    ✗ HTTP {status} for {attempt_id}")
                if status == 400:
                    print(f"      Bad series ID — skipping this fallback")
                    break 
                elif status == 403:
                    print(f"      API key rejected. Double-check your key.")
                    break
                
                time.sleep(1)

            except requests.exceptions.ConnectionError:
                print(f"    ✗ No internet connection.")
                break

            except Exception as e:
                print(f"    ✗ Error: {e}")

        if not fetched:
            print(f"    ⚠ Could not fetch '{name}' — "
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

    df_daily = pd.DataFrame(all_series)

    print(f"\n  Combined daily DataFrame: {df_daily.shape}")
    print(f"  Date range: {df_daily.index[0].date()} → {df_daily.index[-1].date()}")
    print(f"  Missing values per column:")
    for col in df_daily.columns:
        n_missing = df_daily[col].isna().sum()
        pct = n_missing / len(df_daily) * 100
        print(f"    {col}: {n_missing} ({pct:.1f}%)")


    df_quarterly = df_daily.resample("QE").mean()

    print(f"\n  Resampled to quarterly: {df_quarterly.shape}")
    print(f"  Quarters: {df_quarterly.index[0].date()} → {df_quarterly.index[-1].date()}")

    # Save to CSV
    df_quarterly.to_csv(cache_path)
    print(f"\n  Saved to {cache_path}")

    return df_quarterly



def _fmt(value, fmt: str, fallback: str = "N/A") -> str:
    """
    Safely formats a value that might be missing (NaN or the string 'N/A').

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