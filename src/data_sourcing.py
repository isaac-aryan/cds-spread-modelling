"""
data_sourcing.py
================
Handles all data acquisition for the CDS Spread Modelling project.

Three sources are used:
  1. FRED API       — macro indicators (VIX proxy, credit spreads by rating bucket)
  2. yfinance       — company equity data, volatility, financial ratios
  3. Manual/CSV     — rating assignments and sector classifications

Run this file directly to download and cache all raw data:
    python src/data_sourcing.py

All raw data is saved to /data/raw/ as CSVs so you never need to
re-download unless you want fresher data.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Date range for your study.
# 2018-2024 gives you pre-COVID, COVID shock, and recovery — good variation.
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"

# Your FRED API key. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html
# Takes 30 seconds to register. Put it in an environment variable, NOT hardcoded.
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY_HERE")

# ─────────────────────────────────────────────
#  THE COMPANY UNIVERSE
# ─────────────────────────────────────────────
#
# These are real, large US companies with liquid CDS markets.
# We use their observable data as our training set.
# This list covers a range of ratings, sectors, and geographies.
#
# In a real project you would source this list from Markit CDX index
# constituents (publicly documented). Here we hand-pick a representative set.
#
# Format: ticker, company name, sector, rating (as of study start), region
#
# Rating scale we'll use:
#   1=AAA, 2=AA, 3=A, 4=BBB (Investment Grade)
#   5=BB, 6=B, 7=CCC (High Yield)

COMPANY_UNIVERSE = [
    # Investment Grade — Financials
    {"ticker": "JPM",  "name": "JPMorgan Chase",       "sector": "Financials",  "rating": 2, "region": "Northeast"},
    {"ticker": "BAC",  "name": "Bank of America",       "sector": "Financials",  "rating": 2, "region": "Southeast"},
    {"ticker": "GS",   "name": "Goldman Sachs",         "sector": "Financials",  "rating": 3, "region": "Northeast"},
    {"ticker": "MS",   "name": "Morgan Stanley",        "sector": "Financials",  "rating": 3, "region": "Northeast"},
    {"ticker": "WFC",  "name": "Wells Fargo",           "sector": "Financials",  "rating": 3, "region": "West"},
    {"ticker": "C",    "name": "Citigroup",             "sector": "Financials",  "rating": 3, "region": "Northeast"},
    {"ticker": "AIG",  "name": "AIG",                   "sector": "Financials",  "rating": 3, "region": "Northeast"},

    # Investment Grade — Technology
    {"ticker": "AAPL", "name": "Apple",                 "sector": "Technology",  "rating": 1, "region": "West"},
    {"ticker": "MSFT", "name": "Microsoft",             "sector": "Technology",  "rating": 1, "region": "West"},
    {"ticker": "GOOGL","name": "Alphabet",              "sector": "Technology",  "rating": 1, "region": "West"},
    {"ticker": "INTC", "name": "Intel",                 "sector": "Technology",  "rating": 3, "region": "West"},
    {"ticker": "IBM",  "name": "IBM",                   "sector": "Technology",  "rating": 3, "region": "Northeast"},
    {"ticker": "ORCL", "name": "Oracle",                "sector": "Technology",  "rating": 4, "region": "West"},
    {"ticker": "HPE",  "name": "Hewlett Packard Ent.",  "sector": "Technology",  "rating": 4, "region": "West"},

    # Investment Grade — Energy
    {"ticker": "XOM",  "name": "ExxonMobil",            "sector": "Energy",      "rating": 2, "region": "South"},
    {"ticker": "CVX",  "name": "Chevron",               "sector": "Energy",      "rating": 2, "region": "West"},
    {"ticker": "COP",  "name": "ConocoPhillips",        "sector": "Energy",      "rating": 3, "region": "South"},
    {"ticker": "SLB",  "name": "SLB (Schlumberger)",    "sector": "Energy",      "rating": 3, "region": "South"},
    {"ticker": "OXY",  "name": "Occidental Petroleum",  "sector": "Energy",      "rating": 4, "region": "South"},

    # Investment Grade — Healthcare
    {"ticker": "JNJ",  "name": "Johnson & Johnson",     "sector": "Healthcare",  "rating": 1, "region": "Northeast"},
    {"ticker": "PFE",  "name": "Pfizer",                "sector": "Healthcare",  "rating": 3, "region": "Northeast"},
    {"ticker": "MRK",  "name": "Merck",                 "sector": "Healthcare",  "rating": 3, "region": "Northeast"},
    {"ticker": "ABT",  "name": "Abbott Labs",           "sector": "Healthcare",  "rating": 3, "region": "Midwest"},
    {"ticker": "BMY",  "name": "Bristol-Myers Squibb",  "sector": "Healthcare",  "rating": 3, "region": "Northeast"},

    # Investment Grade — Consumer/Retail
    {"ticker": "WMT",  "name": "Walmart",               "sector": "Consumer",    "rating": 2, "region": "South"},
    {"ticker": "PG",   "name": "Procter & Gamble",      "sector": "Consumer",    "rating": 2, "region": "Midwest"},
    {"ticker": "KO",   "name": "Coca-Cola",             "sector": "Consumer",    "rating": 3, "region": "Southeast"},
    {"ticker": "MCD",  "name": "McDonald's",            "sector": "Consumer",    "rating": 4, "region": "Midwest"},
    {"ticker": "DIS",  "name": "Walt Disney",           "sector": "Consumer",    "rating": 3, "region": "West"},

    # Investment Grade — Industrials
    {"ticker": "BA",   "name": "Boeing",                "sector": "Industrials", "rating": 4, "region": "South"},
    {"ticker": "GE",   "name": "GE",                    "sector": "Industrials", "rating": 4, "region": "Northeast"},
    {"ticker": "CAT",  "name": "Caterpillar",           "sector": "Industrials", "rating": 3, "region": "Midwest"},
    {"ticker": "HON",  "name": "Honeywell",             "sector": "Industrials", "rating": 3, "region": "Southeast"},

    # High Yield — Mixed sectors
    {"ticker": "F",    "name": "Ford Motor",            "sector": "Consumer",    "rating": 5, "region": "Midwest"},
    {"ticker": "GM",   "name": "General Motors",        "sector": "Consumer",    "rating": 5, "region": "Midwest"},
    {"ticker": "T",    "name": "AT&T",                  "sector": "Telecom",     "rating": 4, "region": "South"},
    {"ticker": "VZ",   "name": "Verizon",               "sector": "Telecom",     "rating": 4, "region": "Northeast"},
    {"ticker": "CCL",  "name": "Carnival Corp",         "sector": "Consumer",    "rating": 5, "region": "Southeast"},
    {"ticker": "DAL",  "name": "Delta Air Lines",       "sector": "Industrials", "rating": 5, "region": "Southeast"},
    {"ticker": "UAL",  "name": "United Airlines",       "sector": "Industrials", "rating": 5, "region": "Midwest"},
    {"ticker": "M",    "name": "Macy's",                "sector": "Consumer",    "rating": 6, "region": "Northeast"},
    {"ticker": "AMC",  "name": "AMC Networks",          "sector": "Consumer",    "rating": 6, "region": "Northeast"},
    {"ticker": "DISH", "name": "DISH Network",          "sector": "Telecom",     "rating": 6, "region": "West"},
]


# ─────────────────────────────────────────────────────────
#  SOURCE 1: FRED — Macro Indicators
# ─────────────────────────────────────────────────────────
#
# FRED series we use:
#
#   BAMLH0A0HYM2  — ICE BofA US High Yield Index OAS (Option-Adjusted Spread)
#                   This is the average spread across ALL HY bonds. We use it
#                   as a market-wide credit risk thermometer.
#
#   BAMLC0A0CM    — ICE BofA US Corporate Index OAS (Investment Grade average)
#                   Same thing but for IG. Together, IG OAS and HY OAS capture
#                   the overall credit environment on any given date.
#
#   VIXCLS        — CBOE VIX volatility index.
#                   Higher VIX = more market fear = wider spreads generally.
#
#   DGS10         — 10-Year US Treasury yield (risk-free rate).
#                   CDS spreads widen when the risk-free rate rises in stress periods.
#
#   DTWEXBGS      — US Dollar index. Dollar strength affects multinational credit risk.
#
# We will take a snapshot per company per quarter (not daily) because
# company fundamentals are only available quarterly anyway.

FRED_SERIES = {
    "hy_oas":    "BAMLH0A0HYM2",   # HY market spread
    "ig_oas":    "BAMLC0A0CM",     # IG market spread
    "vix":       "VIXCLS",         # VIX
    "risk_free": "DGS10",          # 10Y Treasury
    "usd_index": "DTWEXBGS",       # USD index
}


def fetch_fred_series(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
    """
    Fetch a single time series from the FRED API.

    FRED's API is simple — it returns JSON with a list of {date, value} objects.
    We parse that into a pandas Series indexed by date.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",
        "observation_start": start,
        "observation_end":   end,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()  # Raises an exception for HTTP errors (404, 403, etc.)

    data = response.json()["observations"]
    series = pd.Series(
        {obs["date"]: float(obs["value"]) for obs in data if obs["value"] != "."},
        # FRED uses "." as its missing value indicator — we exclude those
        name=series_id
    )
    series.index = pd.to_datetime(series.index)
    return series


def fetch_all_macro_data(api_key: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetches all FRED macro series and returns a single DataFrame,
    resampled to quarterly frequency (end of quarter).

    We resample to quarterly because our company features (leverage ratios,
    earnings data) are only available quarterly. Using daily macro data
    with quarterly fundamentals would create a false impression of precision.
    """
    out_path = RAW_DIR / "macro_fred.csv"
    if out_path.exists():
        print(f"  [CACHE HIT] Loading macro data from {out_path}")
        return pd.read_csv(out_path, index_col=0, parse_dates=True)

    print("  Fetching macro data from FRED...")
    frames = {}
    for name, series_id in FRED_SERIES.items():
        print(f"    Fetching {name} ({series_id})...")
        try:
            frames[name] = fetch_fred_series(series_id, api_key, start, end)
            time.sleep(0.5)  # Be polite to the API — don't hammer it
        except Exception as e:
            print(f"    WARNING: Failed to fetch {series_id}: {e}")

    df = pd.DataFrame(frames)

    # Resample to quarter-end frequency.
    # 'QE' = quarter end. We take the MEAN over the quarter, not just the
    # last observation — this smooths out daily noise.
    df = df.resample("QE").mean()

    df.to_csv(out_path)
    print(f"  Saved macro data → {out_path}")
    return df


# ─────────────────────────────────────────────────────────
#  SOURCE 2: yfinance — Company Equity & Fundamental Data
# ─────────────────────────────────────────────────────────
#
# For each company we collect:
#
# FROM PRICE HISTORY (daily → aggregated quarterly):
#   - hist_volatility   : 30-day rolling std of log returns, annualised
#                         This is a proxy for asset volatility (key Merton model input)
#   - avg_volume        : average daily trading volume (proxy for liquidity)
#   - market_cap_approx : price × shares (approximated from price history)
#
# FROM QUARTERLY FINANCIALS (balance sheet + income statement):
#   - leverage_ratio    : total debt / total assets  (key default risk driver)
#   - interest_coverage : EBIT / interest expense    (ability to service debt)
#   - roa               : net income / total assets  (profitability)
#   - current_ratio     : current assets / current liabilities (short-term liquidity)
#
# IMPORTANT NOTE ON yfinance QUARTERLY DATA:
# yfinance's .quarterly_balance_sheet returns data indexed by the report date.
# The columns are dates, the rows are line items. We need to transpose and
# match each quarter's fundamental data with that quarter's spread.

def compute_historical_volatility(prices: pd.Series, window: int = 63) -> pd.Series:
    """
    Compute rolling historical volatility from a price series.

    Method:
      1. Compute log returns: ln(P_t / P_{t-1})
      2. Take rolling standard deviation over `window` trading days
         (63 days ≈ 1 quarter of trading days)
      3. Annualise by multiplying by sqrt(252)
         (252 = approximate trading days in a year)

    Returns a Series of annualised volatility values at each date.

    Why log returns instead of simple returns?
      Log returns are additive across time and more statistically well-behaved.
      This is standard in quantitative finance.
    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_vol = log_returns.rolling(window=window).std()
    annualised_vol = rolling_vol * np.sqrt(252)
    return annualised_vol


def fetch_company_equity_data(ticker: str, start: str, end: str) -> dict:
    """
    Fetches all equity and fundamental data for a single company.

    Returns a dict where keys are (ticker, quarter_end_date) and
    values are feature dictionaries. This gets assembled into the
    final panel dataset.
    """
    try:
        stock = yf.Ticker(ticker)

        # ── Price history ──────────────────────────────────────────────
        hist = stock.history(start=start, end=end, interval="1d")
        if hist.empty:
            print(f"    WARNING: No price data for {ticker}")
            return {}

        prices = hist["Close"]

        # Compute volatility and resample to quarter-end
        vol_daily = compute_historical_volatility(prices, window=63)
        vol_quarterly = vol_daily.resample("QE").mean()  # average vol over quarter

        # Market cap approximation: price × shares outstanding
        # yfinance gives shares outstanding in .info
        try:
            shares = stock.info.get("sharesOutstanding", np.nan)
        except Exception:
            shares = np.nan

        price_quarterly = prices.resample("QE").last()
        mktcap_quarterly = price_quarterly * shares  # can be NaN if shares unknown

        # Average daily volume (proxy for liquidity)
        volume_quarterly = hist["Volume"].resample("QE").mean()

        # ── Quarterly Fundamentals ──────────────────────────────────────
        # yfinance returns these as DataFrames with dates as COLUMNS
        # and financial line items as ROWS. We transpose for convenience.
        try:
            bs = stock.quarterly_balance_sheet.T   # balance sheet
            inc = stock.quarterly_income_stmt.T    # income statement
        except Exception as e:
            print(f"    WARNING: Could not get financials for {ticker}: {e}")
            bs = pd.DataFrame()
            inc = pd.DataFrame()

        # ── Assemble quarterly records ──────────────────────────────────
        quarterly_data = {}

        for qdate in vol_quarterly.index:
            record = {
                "hist_volatility": vol_quarterly.get(qdate, np.nan),
                "market_cap":      mktcap_quarterly.get(qdate, np.nan),
                "avg_volume":      volume_quarterly.get(qdate, np.nan),
            }

            # Match fundamentals to this quarter.
            # Fundamentals are reported with a slight lag (e.g. Q1 ends March 31
            # but is reported in May). We find the most recent available report
            # BEFORE this quarter end date.
            if not bs.empty:
                bs_dates = bs.index[bs.index <= qdate]
                if len(bs_dates) > 0:
                    latest_bs = bs.loc[bs_dates[-1]]

                    total_debt   = _get_bs_item(latest_bs, [
                        "Total Debt", "Long Term Debt", "LongTermDebt"
                    ])
                    total_assets = _get_bs_item(latest_bs, [
                        "Total Assets", "TotalAssets"
                    ])
                    curr_assets  = _get_bs_item(latest_bs, [
                        "Current Assets", "Total Current Assets"
                    ])
                    curr_liab    = _get_bs_item(latest_bs, [
                        "Current Liabilities", "Total Current Liabilities"
                    ])

                    record["leverage_ratio"] = (
                        total_debt / total_assets
                        if not np.isnan(total_debt) and not np.isnan(total_assets) and total_assets != 0
                        else np.nan
                    )
                    record["current_ratio"] = (
                        curr_assets / curr_liab
                        if not np.isnan(curr_assets) and not np.isnan(curr_liab) and curr_liab != 0
                        else np.nan
                    )

            if not inc.empty:
                inc_dates = inc.index[inc.index <= qdate]
                if len(inc_dates) > 0:
                    latest_inc = inc.loc[inc_dates[-1]]

                    ebit      = _get_bs_item(latest_inc, ["EBIT", "Operating Income"])
                    int_exp   = _get_bs_item(latest_inc, ["Interest Expense"])
                    net_inc   = _get_bs_item(latest_inc, ["Net Income"])
                    tot_assets_check = _get_bs_item(
                        bs.loc[bs.index[bs.index <= qdate][-1]] if not bs.empty and len(bs.index[bs.index <= qdate]) > 0 else pd.Series(),
                        ["Total Assets"]
                    )

                    record["interest_coverage"] = (
                        ebit / abs(int_exp)
                        if not np.isnan(ebit) and not np.isnan(int_exp) and int_exp != 0
                        else np.nan
                    )
                    record["roa"] = (
                        net_inc / tot_assets_check
                        if not np.isnan(net_inc) and not np.isnan(tot_assets_check) and tot_assets_check != 0
                        else np.nan
                    )

            quarterly_data[qdate] = record

        return quarterly_data

    except Exception as e:
        print(f"    ERROR fetching {ticker}: {e}")
        return {}


def _get_bs_item(series: pd.Series, possible_names: list) -> float:
    """
    Financial statement line items have inconsistent naming across companies
    in yfinance. This helper tries multiple possible names and returns the
    first one found, or NaN if none exist.

    Example: total debt might be "Total Debt" for one company and
    "Long Term Debt" for another.
    """
    for name in possible_names:
        if name in series.index:
            val = series[name]
            if pd.notna(val):
                return float(val)
    return np.nan


def fetch_all_company_equity_data(universe: list, start: str, end: str) -> pd.DataFrame:
    """
    Loops over all companies in the universe and fetches their equity/fundamental data.
    Assembles everything into a panel DataFrame with MultiIndex (ticker, date).

    Saves raw output to CSV so you don't re-download unnecessarily.
    """
    out_path = RAW_DIR / "company_equity.csv"
    if out_path.exists():
        print(f"  [CACHE HIT] Loading equity data from {out_path}")
        df = pd.read_csv(out_path, index_col=[0, 1], parse_dates=True)
        return df

    print(f"  Fetching equity data for {len(universe)} companies...")
    records = []

    for i, company in enumerate(universe):
        ticker = company["ticker"]
        print(f"    [{i+1}/{len(universe)}] {ticker} — {company['name']}")

        quarterly_data = fetch_company_equity_data(ticker, start, end)

        for qdate, features in quarterly_data.items():
            row = {"ticker": ticker, "date": qdate}
            row.update(features)
            records.append(row)

        time.sleep(1)  # Rate limiting — yfinance will block you if you spam it

    df = pd.DataFrame(records).set_index(["ticker", "date"])
    df.to_csv(out_path)
    print(f"  Saved equity data → {out_path}")
    return df


# ─────────────────────────────────────────────────────────
#  SOURCE 3: CDS Spread Proxy (Our Target Variable Y)
# ─────────────────────────────────────────────────────────
#
# HONEST DISCLAIMER:
# Real CDS spreads require Markit/Bloomberg. We construct a PROXY
# using the following empirical approach, which is academically
# defensible for coursework purposes:
#
# CDS_spread_proxy(company i, quarter t) =
#     SECTOR_BASE_SPREAD(sector, quarter t)           ← from FRED by rating bucket
#   + RATING_ADJUSTMENT(rating)                       ← fixed add-on per rating notch
#   + LEVERAGE_ADJUSTMENT(leverage_ratio)             ← higher leverage → higher spread
#   + VOLATILITY_ADJUSTMENT(hist_volatility)          ← higher vol → higher spread
#   + IDIOSYNCRATIC_NOISE                             ← small random term per company
#
# This creates a realistic synthetic target that has genuine structure
# your SVR can learn — the features causally drive the target.
#
# Importantly: this is exactly how practitioners think about CDS spreads.
# The proxy captures the main drivers even if the absolute levels aren't
# real Markit quotes.
#
# If you have WRDS access, replace this function with real data — see
# the WRDS section at the bottom of this file.

# Base spreads by rating bucket in basis points (bps)
# These are rough long-run averages — they'll be modulated by macro conditions
RATING_BASE_SPREAD = {
    1: 15,    # AAA
    2: 30,    # AA
    3: 60,    # A
    4: 120,   # BBB
    5: 280,   # BB
    6: 550,   # B
    7: 1100,  # CCC
}

# Sector adjustments in bps — some sectors are structurally riskier
SECTOR_ADJUSTMENT = {
    "Financials":  20,
    "Energy":      35,
    "Technology":  -10,
    "Healthcare":  -5,
    "Consumer":    10,
    "Industrials": 15,
    "Telecom":     25,
}


def construct_cds_spread_proxy(
    company_universe: list,
    equity_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Constructs the synthetic CDS spread proxy for each (company, quarter).

    The formula is:
        spread = base
               + sector_adj
               + macro_scale × (hy_oas or ig_oas from FRED)  ← market conditions
               + leverage_loading × leverage_ratio
               + vol_loading × hist_volatility
               + idiosyncratic_noise

    We clip to a minimum of 10 bps (spreads can't go negative in reality).
    """
    rng = np.random.default_rng(seed)
    records = []

    company_lookup = {c["ticker"]: c for c in company_universe}

    for (ticker, qdate), row in equity_df.iterrows():
        company = company_lookup.get(ticker)
        if company is None:
            continue

        rating = company["rating"]
        sector = company["sector"]

        base = RATING_BASE_SPREAD.get(rating, 200)
        sect_adj = SECTOR_ADJUSTMENT.get(sector, 0)

        # Scale by macro conditions on that date
        macro_row = macro_df.loc[macro_df.index <= qdate].iloc[-1] if len(macro_df.loc[macro_df.index <= qdate]) > 0 else pd.Series()

        # Higher VIX → all spreads wider (credit risk premium rises)
        vix = macro_row.get("vix", 20)
        vix_adj = (vix - 20) * 3   # VIX of 30 adds 30bps, VIX of 15 removes 15bps

        # Market-level spread from FRED modulates our estimate
        if rating <= 4:  # Investment grade
            market_spread = macro_row.get("ig_oas", 100)
            macro_adj = (market_spread - 100) * 0.4   # 40% pass-through
        else:             # High yield
            market_spread = macro_row.get("hy_oas", 400)
            macro_adj = (market_spread - 400) * 0.5   # 50% pass-through

        # Leverage loading: every 10pp of leverage adds ~50bps (nonlinear)
        leverage = row.get("leverage_ratio", np.nan)
        if pd.notna(leverage):
            lev_adj = leverage * 150   # e.g. leverage=0.4 → +60bps
        else:
            lev_adj = 0

        # Volatility loading: higher equity vol → higher spread
        vol = row.get("hist_volatility", np.nan)
        if pd.notna(vol):
            vol_adj = vol * 200   # e.g. vol=0.3 → +60bps
        else:
            vol_adj = 0

        # Small idiosyncratic noise per company
        # Using a fixed seed means this is reproducible — same company always
        # gets the same noise draw across runs
        noise = rng.normal(0, base * 0.08)   # ±8% of base spread

        spread = base + sect_adj + vix_adj + macro_adj + lev_adj + vol_adj + noise
        spread = max(spread, 10)   # Floor at 10bps

        records.append({
            "ticker":    ticker,
            "date":      qdate,
            "cds_spread_bps": spread
        })

    df = pd.DataFrame(records).set_index(["ticker", "date"])
    return df


# ─────────────────────────────────────────────────────────
#  WRDS / REAL DATA PATH (University Access)
# ─────────────────────────────────────────────────────────
#
# If your university has WRDS access, you can get real CDS data.
# Install the WRDS Python library: pip install wrds
#
# Then replace construct_cds_spread_proxy() with:
#
#   import wrds
#   db = wrds.Connection(wrds_username="your_username")
#
#   # Markit CDS spreads — 5-year tenor, USD, senior unsecured
#   query = """
#       SELECT ticker, datadate, parspread5y AS cds_spread_bps
#       FROM markit.cds_composites
#       WHERE currency = 'USD'
#         AND tier = 'SNRFOR'
#         AND tenor = '5Y'
#         AND datadate BETWEEN '2018-01-01' AND '2024-01-01'
#   """
#   cds_df = db.raw_sql(query)
#
# The rest of the pipeline is identical — just swap in cds_df as your target.


# ─────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────

def source_all_data():
    """
    Downloads and caches all raw data.
    Call this once to populate data/raw/.
    """
    print("\n" + "="*60)
    print("  CDS SPREAD MODELLING — DATA SOURCING")
    print("="*60 + "\n")

    print("[1/3] Fetching macro data from FRED...")
    macro_df = fetch_all_macro_data(FRED_API_KEY, START_DATE, END_DATE)
    print(f"      Shape: {macro_df.shape}  |  Date range: {macro_df.index[0].date()} → {macro_df.index[-1].date()}")

    print("\n[2/3] Fetching company equity & fundamental data from yfinance...")
    equity_df = fetch_all_company_equity_data(COMPANY_UNIVERSE, START_DATE, END_DATE)
    print(f"      Shape: {equity_df.shape}")

    print("\n[3/3] Constructing CDS spread proxy (target variable)...")
    cds_df = construct_cds_spread_proxy(COMPANY_UNIVERSE, equity_df, macro_df)
    out_path = RAW_DIR / "cds_spreads_proxy.csv"
    cds_df.to_csv(out_path)
    print(f"      Shape: {cds_df.shape}")
    print(f"      Spread range: {cds_df['cds_spread_bps'].min():.1f} — {cds_df['cds_spread_bps'].max():.1f} bps")
    print(f"      Saved → {out_path}")

    print("\n✓ All raw data sourced. Run preprocessing.py next.\n")
    return macro_df, equity_df, cds_df


if __name__ == "__main__":
    source_all_data()