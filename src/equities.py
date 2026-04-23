"""
step2_equity.py
===============
Fetches real company equity data and financial fundamentals using yfinance.

Run this file directly:
    python src/step2_equity.py

What this file teaches:
    - How yfinance works and what data it provides
    - How to compute historical (realised) volatility from price data
    - How to extract and clean quarterly financial statement data
    - How to handle the real messiness of financial data (inconsistent
      column names, reporting lags, missing entries)
    - How to build a panel dataset (multiple companies × multiple quarters)

No API key needed — yfinance uses Yahoo Finance's public data feed.
However: yfinance can be unreliable (Yahoo changes their API without warning).
If a ticker fails, that's normal — we handle it gracefully.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  COMPANY UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
#
# These are real US companies with active CDS markets.
# We've spread them across sectors and ratings to give the model
# meaningful variation to learn from.
#
# Rating scale:
#   1=AAA, 2=AA, 3=A, 4=BBB (Investment Grade — IG)
#   5=BB, 6=B, 7=CCC         (High Yield — HY)
#
# These ratings are approximate and fixed at study start (2018).
# In a real project you'd use a time-varying rating series from a
# ratings agency (Moody's, S&P, Fitch).

COMPANY_UNIVERSE = [
    # ── Investment Grade: Financials ─────────────────────────────────────────
    {"ticker": "JPM",  "name": "JPMorgan Chase",      "sector": "Financials",  "rating": 2, "region": "Northeast"},
    {"ticker": "BAC",  "name": "Bank of America",      "sector": "Financials",  "rating": 2, "region": "Southeast"},
    {"ticker": "GS",   "name": "Goldman Sachs",        "sector": "Financials",  "rating": 3, "region": "Northeast"},
    {"ticker": "MS",   "name": "Morgan Stanley",       "sector": "Financials",  "rating": 3, "region": "Northeast"},
    {"ticker": "WFC",  "name": "Wells Fargo",          "sector": "Financials",  "rating": 3, "region": "West"},
    {"ticker": "C",    "name": "Citigroup",            "sector": "Financials",  "rating": 3, "region": "Northeast"},

    # ── Investment Grade: Technology ─────────────────────────────────────────
    {"ticker": "AAPL", "name": "Apple",                "sector": "Technology",  "rating": 1, "region": "West"},
    {"ticker": "MSFT", "name": "Microsoft",            "sector": "Technology",  "rating": 1, "region": "West"},
    {"ticker": "GOOGL","name": "Alphabet",             "sector": "Technology",  "rating": 1, "region": "West"},
    {"ticker": "INTC", "name": "Intel",                "sector": "Technology",  "rating": 3, "region": "West"},
    {"ticker": "IBM",  "name": "IBM",                  "sector": "Technology",  "rating": 3, "region": "Northeast"},
    {"ticker": "ORCL", "name": "Oracle",               "sector": "Technology",  "rating": 4, "region": "West"},

    # ── Investment Grade: Energy ─────────────────────────────────────────────
    {"ticker": "XOM",  "name": "ExxonMobil",           "sector": "Energy",      "rating": 2, "region": "South"},
    {"ticker": "CVX",  "name": "Chevron",              "sector": "Energy",      "rating": 2, "region": "West"},
    {"ticker": "COP",  "name": "ConocoPhillips",       "sector": "Energy",      "rating": 3, "region": "South"},
    {"ticker": "OXY",  "name": "Occidental Petroleum", "sector": "Energy",      "rating": 4, "region": "South"},

    # ── Investment Grade: Healthcare ─────────────────────────────────────────
    {"ticker": "JNJ",  "name": "Johnson & Johnson",    "sector": "Healthcare",  "rating": 1, "region": "Northeast"},
    {"ticker": "PFE",  "name": "Pfizer",               "sector": "Healthcare",  "rating": 3, "region": "Northeast"},
    {"ticker": "MRK",  "name": "Merck",                "sector": "Healthcare",  "rating": 3, "region": "Northeast"},
    {"ticker": "ABT",  "name": "Abbott Labs",          "sector": "Healthcare",  "rating": 3, "region": "Midwest"},

    # ── Investment Grade: Consumer ────────────────────────────────────────────
    {"ticker": "WMT",  "name": "Walmart",              "sector": "Consumer",    "rating": 2, "region": "South"},
    {"ticker": "PG",   "name": "Procter & Gamble",     "sector": "Consumer",    "rating": 2, "region": "Midwest"},
    {"ticker": "KO",   "name": "Coca-Cola",            "sector": "Consumer",    "rating": 3, "region": "Southeast"},
    {"ticker": "MCD",  "name": "McDonald's",           "sector": "Consumer",    "rating": 4, "region": "Midwest"},
    {"ticker": "DIS",  "name": "Walt Disney",          "sector": "Consumer",    "rating": 3, "region": "West"},

    # ── Investment Grade: Industrials ─────────────────────────────────────────
    {"ticker": "BA",   "name": "Boeing",               "sector": "Industrials", "rating": 4, "region": "South"},
    {"ticker": "CAT",  "name": "Caterpillar",          "sector": "Industrials", "rating": 3, "region": "Midwest"},
    {"ticker": "HON",  "name": "Honeywell",            "sector": "Industrials", "rating": 3, "region": "Southeast"},

    # ── High Yield ────────────────────────────────────────────────────────────
    {"ticker": "F",    "name": "Ford Motor",           "sector": "Consumer",    "rating": 5, "region": "Midwest"},
    {"ticker": "GM",   "name": "General Motors",       "sector": "Consumer",    "rating": 5, "region": "Midwest"},
    {"ticker": "T",    "name": "AT&T",                 "sector": "Telecom",     "rating": 4, "region": "South"},
    {"ticker": "VZ",   "name": "Verizon",              "sector": "Telecom",     "rating": 4, "region": "Northeast"},
    {"ticker": "CCL",  "name": "Carnival Corp",        "sector": "Consumer",    "rating": 5, "region": "Southeast"},
    {"ticker": "DAL",  "name": "Delta Air Lines",      "sector": "Industrials", "rating": 5, "region": "Southeast"},
    {"ticker": "UAL",  "name": "United Airlines",      "sector": "Industrials", "rating": 5, "region": "Midwest"},
    {"ticker": "M",    "name": "Macy's",               "sector": "Consumer",    "rating": 6, "region": "Northeast"},
]


# ─────────────────────────────────────────────────────────────────────────────
#  VOLATILITY CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_historical_volatility(prices: pd.Series,
                                  window_days: int = 63) -> pd.Series:
    """
    Computes rolling historical (realised) volatility from daily close prices.

    WHAT IS HISTORICAL VOLATILITY?
    --------------------------------
    It measures how much a stock's price has been jumping around recently.
    High volatility = large, unpredictable daily moves = riskier company.
    This is important for CDS spreads because Merton's structural credit model
    (foundational in credit risk) says: default risk ∝ asset volatility.
    If a company's equity swings wildly, that uncertainty extends to bondholders.

    THE CALCULATION:
    ----------------
    Step 1 — Log returns:
        r_t = ln(P_t / P_{t-1})
        We use LOG returns (not simple returns = (P_t - P_{t-1})/P_{t-1}) because:
        - Log returns are additive across time: weekly return = sum of daily log returns
        - They're more normally distributed (better statistical properties)
        - They can't go below -100% (unlike simple returns in theory)

    Step 2 — Rolling standard deviation:
        std(r_t, r_{t-1}, ..., r_{t-window+1})
        This measures how dispersed the recent returns have been.
        Window of 63 trading days ≈ 1 quarter (63 ≈ 252/4).

    Step 3 — Annualise:
        vol_annual = std_daily × sqrt(252)
        252 = approximate number of trading days in a year.
        Multiplying by sqrt(252) converts daily std to annual std.
        (Variance scales linearly with time, so std scales with sqrt(time).)

    Example: if daily std = 0.015 (1.5% per day),
             annual vol = 0.015 × sqrt(252) ≈ 0.238 = 23.8%

    PARAMETERS:
    -----------
    prices      : pd.Series of daily closing prices with DatetimeIndex
    window_days : rolling window in trading days (63 = one quarter)

    RETURNS:
    --------
    pd.Series of annualised volatility values (one per trading day)
    """

    # Step 1: log returns
    # shift(1) shifts the series forward by 1 day, so prices/prices.shift(1)
    # gives today's price divided by yesterday's price for each day
    log_returns = np.log(prices / prices.shift(1))

    # Step 2: rolling standard deviation
    # rolling(window=63) creates a window that slides forward one day at a time
    # .std() computes standard deviation within each window
    # The first 62 values will be NaN (not enough data to fill the window)
    rolling_std = log_returns.rolling(window=window_days).std()

    # Step 3: annualise
    annualised_vol = rolling_std * np.sqrt(252)

    return annualised_vol


# ─────────────────────────────────────────────────────────────────────────────
#  FINANCIAL STATEMENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(series: pd.Series, possible_names: list,
             label: str = "") -> float:
    """
    Retrieves a value from a financial statement row (pd.Series) by trying
    multiple possible column names.

    WHY THIS IS NECESSARY:
    ----------------------
    yfinance extracts financial data from Yahoo Finance, which standardises
    column names — but not perfectly across all companies. For example:
    - JPMorgan might report "Total Debt" as "Total Debt"
    - Citigroup might report it as "Long Term Debt And Capital Lease Obligation"
    - A smaller company might have it as "LongTermDebt" with no spaces

    Rather than hardcoding one name and getting NaN for half our companies,
    we try a list of synonyms and take the first one that has a real value.

    PARAMETERS:
    -----------
    series         : one row of a financial statement (already transposed)
    possible_names : list of column names to try, in order of preference
    label          : human-readable name for debug messages

    RETURNS:
    --------
    float value, or np.nan if none of the names were found
    """
    for name in possible_names:
        if name in series.index:
            val = series[name]
            if pd.notna(val) and val != 0:
                return float(val)
    return np.nan


def get_financial_ratios(bs_row: pd.Series,
                         inc_row: pd.Series) -> dict:
    """
    Computes the financial ratios we need from balance sheet and income
    statement rows.

    FINANCIAL JARGON EXPLAINED:
    ----------------------------

    LEVERAGE RATIO = Total Debt / Total Assets
        What fraction of a company's assets are funded by debt?
        Range: 0 (no debt) to 1+ (more debt than assets = technically insolvent)
        A leveraged buyout target might be 0.7-0.8. Apple is around 0.3.
        Higher leverage → more interest to pay → higher default risk → wider CDS.

    INTEREST COVERAGE = EBIT / Interest Expense
        EBIT = Earnings Before Interest and Taxes (operating profit)
        If EBIT = £500m and interest expense = £100m, coverage = 5×.
        This means operating earnings could fall 80% before the company
        can't cover its interest payments.
        Coverage < 1× = danger zone (company can't pay interest from operations)
        Coverage 3-5× = comfortable
        Coverage 10×+ = very strong

    ROA = Net Income / Total Assets
        Return on Assets measures how efficiently the company uses its
        assets to generate profit.
        -0.05 to 0.20 is the typical range.
        Negative ROA = losing money.

    CURRENT RATIO = Current Assets / Current Liabilities
        Current assets = cash + assets expected to convert to cash within 1 year
        Current liabilities = debts due within 1 year
        Current ratio < 1 means the company can't cover its short-term debts
        with its short-term assets — a liquidity warning sign.
        Typical healthy range: 1.2 - 2.5

    DEBT-TO-EBITDA
        EBITDA = Earnings Before Interest, Taxes, Depreciation and Amortisation
        (a rough proxy for operating cash flow)
        Debt/EBITDA = how many years of operating earnings to repay all debt
        Debt/EBITDA > 5× is considered highly leveraged by most lenders
        Covenant triggers in loan agreements are often set at Debt/EBITDA = 6×
    """
    ratios = {}

    # ── Balance sheet items ───────────────────────────────────────────────────
    total_debt = safe_get(bs_row, [
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "LongTermDebt",
        "Long Term Debt",
    ], "total_debt")

    total_assets = safe_get(bs_row, [
        "Total Assets",
        "TotalAssets",
    ], "total_assets")

    current_assets = safe_get(bs_row, [
        "Current Assets",
        "Total Current Assets",
        "CurrentAssets",
    ], "current_assets")

    current_liab = safe_get(bs_row, [
        "Current Liabilities",
        "Total Current Liabilities",
        "CurrentLiabilities",
    ], "current_liab")

    # ── Income statement items ────────────────────────────────────────────────
    ebit = safe_get(inc_row, [
        "EBIT",
        "Operating Income",
        "OperatingIncome",
    ], "ebit")

    ebitda = safe_get(inc_row, [
        "EBITDA",
        "Normalized EBITDA",
    ], "ebitda")

    interest_expense = safe_get(inc_row, [
        "Interest Expense",
        "InterestExpense",
        "Interest Expense Non Operating",
    ], "interest_expense")

    net_income = safe_get(inc_row, [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
    ], "net_income")

    # ── Compute ratios ────────────────────────────────────────────────────────

    # Leverage ratio: debt / assets
    if not np.isnan(total_debt) and not np.isnan(total_assets) and total_assets > 0:
        ratios["leverage_ratio"] = total_debt / total_assets
    else:
        ratios["leverage_ratio"] = np.nan

    # Interest coverage: EBIT / |interest expense|
    # We use abs() because interest expense is sometimes reported negative
    if not np.isnan(ebit) and not np.isnan(interest_expense) and interest_expense != 0:
        ratios["interest_coverage"] = ebit / abs(interest_expense)
    else:
        ratios["interest_coverage"] = np.nan

    # ROA: net income / total assets
    if not np.isnan(net_income) and not np.isnan(total_assets) and total_assets > 0:
        ratios["roa"] = net_income / total_assets
    else:
        ratios["roa"] = np.nan

    # Current ratio: current assets / current liabilities
    if not np.isnan(current_assets) and not np.isnan(current_liab) and current_liab > 0:
        ratios["current_ratio"] = current_assets / current_liab
    else:
        ratios["current_ratio"] = np.nan

    # Debt-to-EBITDA
    if not np.isnan(total_debt) and not np.isnan(ebitda) and ebitda > 0:
        ratios["debt_to_ebitda"] = total_debt / ebitda
    else:
        ratios["debt_to_ebitda"] = np.nan

    return ratios


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN FETCHER: ONE COMPANY
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one_company(ticker: str, start: str, end: str,
                      verbose: bool = True) -> dict:
    """
    Fetches all equity and fundamental data for a single company.
    """
    stock = yf.Ticker(ticker)
    quarterly_records = {}

    # ── Step 1: Price history → volatility ───────────────────────────────────
    if verbose:
        print(f"    Fetching price history...")

    try:
        hist = stock.history(start=start, end=end, interval="1d",
                             auto_adjust=True)

        if hist.empty:
            if verbose:
                print(f"    ✗ No price data returned")
            return {}

        prices = hist["Close"]

        if verbose:
            print(f"    ✓ {len(prices)} daily prices "
                  f"({prices.index[0].date()} → {prices.index[-1].date()})")

        # Daily volatility → quarterly average
        vol_daily    = compute_historical_volatility(prices, window_days=63)
        vol_q        = vol_daily.resample("QE").mean()

        # Market cap: price × shares outstanding
        try:
            info   = stock.info
            shares = info.get("sharesOutstanding", np.nan)
        except Exception:
            shares = np.nan

        price_q  = prices.resample("QE").last()   # last price of each quarter
        mktcap_q = price_q * shares if not np.isnan(shares) else price_q * np.nan

        volume_q = hist["Volume"].resample("QE").mean()

    except Exception as e:
        if verbose:
            print(f"    ✗ Price data error: {e}")
        return {}

    # ── Step 2: Financial statements ──────────────────────────────────────────
    if verbose:
        print(f"    Fetching financial statements...")

    try:
        # Transpose so rows=dates, columns=line items
        bs  = stock.quarterly_balance_sheet.T
        inc = stock.quarterly_income_stmt.T

        # Sort by date (oldest first)
        bs  = bs.sort_index()
        inc = inc.sort_index()
        
        # FIX: Convert index to datetime and ensure timezone-naive
        if not bs.empty:
            bs.index = pd.to_datetime(bs.index).tz_localize(None)
        if not inc.empty:
            inc.index = pd.to_datetime(inc.index).tz_localize(None)

        if verbose:
            n_bs  = len(bs)  if not bs.empty  else 0
            n_inc = len(inc) if not inc.empty else 0
            print(f"    ✓ Balance sheet: {n_bs} quarters | "
                  f"Income stmt: {n_inc} quarters")

    except Exception as e:
        if verbose:
            print(f"    ✗ Financial statements error: {e}")
        bs  = pd.DataFrame()
        inc = pd.DataFrame()

    # ── Step 3: Assemble quarterly records ───────────────────────────────────
    
    # FIX: Ensure vol_q index is also timezone-naive for consistent comparisons
    vol_q.index = pd.to_datetime(vol_q.index).tz_localize(None)
    
    for qdate in vol_q.index:
        record = {
            "hist_volatility": vol_q.get(qdate, np.nan),
            "market_cap":      mktcap_q.get(qdate, np.nan),
            "avg_volume":      volume_q.get(qdate, np.nan),
        }

        # Find the most recent balance sheet filed on or before this quarter
        if not bs.empty:
            # FIX: Ensure qdate is timezone-naive for comparison
            qdate_naive = pd.to_datetime(qdate).tz_localize(None)
            
            # FIX: Use .values for comparison to avoid dtype issues
            available_bs = bs[bs.index.values <= qdate_naive]
            if len(available_bs) > 0:
                bs_row = available_bs.iloc[-1]

                # Find the most recent income statement similarly
                if not inc.empty:
                    available_inc = inc[inc.index.values <= qdate_naive]
                    inc_row = available_inc.iloc[-1] if len(available_inc) > 0 else pd.Series()
                else:
                    inc_row = pd.Series()

                ratios = get_financial_ratios(bs_row, inc_row)
                record.update(ratios)

        quarterly_records[qdate] = record

    return quarterly_records


# ─────────────────────────────────────────────────────────────────────────────
#  FETCH ALL COMPANIES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_equity(universe: list, start: str, end: str,
                     force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches equity and fundamental data for all companies in the universe.
    Assembles into a panel DataFrame with MultiIndex (ticker, date).

    A PANEL DATASET is a combination of cross-sectional data (many companies)
    and time-series data (many quarters per company). Also called "longitudinal"
    data. The MultiIndex lets you slice by either ticker or date:

        df.loc["AAPL"]           → all quarters for Apple
        df.loc[pd.IndexSlice[:, "2020-03-31"], :] → all companies in Q1 2020
    """
    cache_path = RAW_DIR / "company_equity.csv"

    if cache_path.exists() and not force_refresh:
        print(f"  [CACHE] Loading equity data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=[0, 1], parse_dates=True)
        return df

    print(f"  Fetching equity data for {len(universe)} companies...")
    print(f"  This will take several minutes (rate limiting between requests)\n")

    all_records = []
    failed = []

    for i, company in enumerate(universe):
        ticker = company["ticker"]
        print(f"  [{i+1:2d}/{len(universe)}] {ticker:6s} — {company['name']}")

        try:
            quarterly_data = fetch_one_company(ticker, start, end, verbose=True)

            if not quarterly_data:
                failed.append(ticker)
                print(f"    ✗ No data returned\n")
                continue

            for qdate, features in quarterly_data.items():
                row = {
                    "ticker": ticker,
                    "date":   qdate,
                    **features   # unpack all feature key-value pairs
                }
                all_records.append(row)

            n_quarters = len(quarterly_data)
            print(f"    → {n_quarters} quarters of data assembled\n")

        except Exception as e:
            failed.append(ticker)
            print(f"    ✗ Unexpected error: {e}\n")

        # Rate limiting: yfinance will throttle or block you without this
        time.sleep(1.5)

    if not all_records:
        raise RuntimeError("No data was assembled for any company.")

    df = pd.DataFrame(all_records)

    # Set MultiIndex: (ticker, date) — the standard format for panel data
    df = df.set_index(["ticker", "date"]).sort_index()

    # Report what we got
    n_companies = df.index.get_level_values("ticker").nunique()
    n_rows      = len(df)
    print(f"\n  ── Assembly complete ──")
    print(f"  {n_companies} companies × {n_rows // n_companies:.0f} avg quarters "
          f"= {n_rows} total rows")
    print(f"  Features: {list(df.columns)}")
    if failed:
        print(f"  Failed tickers: {failed}")

    df.to_csv(cache_path)
    print(f"\n  Saved to {cache_path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

def inspect_equity_data(df: pd.DataFrame) -> None:
    """Thorough inspection of the assembled equity panel."""
    print("\n" + "="*60)
    print("EQUITY DATA INSPECTION")
    print("="*60)

    tickers = df.index.get_level_values("ticker").unique()
    dates   = df.index.get_level_values("date").unique()

    print(f"\nCompanies: {len(tickers)}")
    print(f"Quarters:  {len(dates)} ({dates.min().date()} → {dates.max().date()})")
    print(f"Total rows: {len(df)}")
    print(f"Features:  {list(df.columns)}")

    print("\n── Missing values (%) ──")
    for col in df.columns:
        pct_missing = df[col].isna().mean() * 100
        bar = "█" * int(pct_missing / 5) + "░" * (20 - int(pct_missing / 5))
        print(f"  {col:25s} [{bar}] {pct_missing:5.1f}%")

    print("\n── Feature ranges ──")
    print(df.describe().round(3).to_string())

    print("\n── Sample: Apple (AAPL) first 4 quarters ──")
    if "AAPL" in tickers:
        print(df.loc["AAPL"].head(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 2: FETCHING COMPANY EQUITY DATA")
    print("="*60 + "\n")

    equity_df = fetch_all_equity(COMPANY_UNIVERSE, START_DATE, END_DATE)
    inspect_equity_data(equity_df)

    print("\n✓ Step 2 complete. Run step3_target.py next.")
    print(f"  Data saved to: {RAW_DIR / 'company_equity.csv'}")