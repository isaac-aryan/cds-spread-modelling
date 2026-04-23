"""
step3_build_panel.py
====================
Combines macro data (FRED) + equity data (yfinance) + the CDS spread proxy
into a single, model-ready panel dataset.

Run this file directly:
    python src/step3_build_panel.py

What this file teaches:
    - How to merge panel datasets on multiple keys (ticker + date)
    - How to construct a financially-grounded synthetic target variable
    - How to do a final data quality check before modelling
    - What the final feature matrix looks like

IMPORTANT NOTE ON THE TARGET VARIABLE:
    Real CDS spreads require Markit (Bloomberg/WRDS).
    We construct a proxy driven by REAL macro data (FRED) and REAL fundamentals
    (yfinance), with a formula grounded in credit risk theory.
    This is the same approach used in your mock.py, but now the inputs are real.
    The spread levels won't exactly match Markit quotes, but the DRIVERS are
    the same — which means the SVR can learn genuine relationships.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SPREAD CONSTRUCTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
#
# These parameters are based on long-run empirical averages in credit markets.
# Sources: BAML/ICE credit spread indices, academic CDS pricing literature.

# Base CDS spread in bps by rating (approximate long-run average 5Y CDS)
RATING_BASE_SPREAD = {
    1:  15,    # AAA — almost no default risk (only a handful of companies)
    2:  30,    # AA  — very low risk (US agencies, top-tier banks)
    3:  60,    # A   — low risk (strong investment grade)
    4: 120,    # BBB — moderate risk (the bulk of IG corporate bonds)
    5: 280,    # BB  — elevated risk, speculative grade
    6: 550,    # B   — high risk, highly speculative
    7:1100,    # CCC — very high risk, near distress
}

# Sector risk premium in bps — some sectors are structurally riskier
# Financials: exposed to systemic risk, mark-to-market losses
# Energy: commodity price volatility, capex-heavy
# Technology: low physical assets, but strong cash generation
SECTOR_ADJUSTMENT = {
    "Financials":  20,
    "Energy":      35,
    "Technology": -10,
    "Healthcare":  -5,
    "Consumer":    10,
    "Industrials": 15,
    "Telecom":     25,
}


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD THE CDS SPREAD PROXY
# ─────────────────────────────────────────────────────────────────────────────

def build_cds_proxy(company_universe: list,
                    equity_df: pd.DataFrame,
                    macro_df: pd.DataFrame,
                    seed: int = 42) -> pd.DataFrame:
    """
    Constructs a CDS spread proxy for each (company, quarter) observation.

    THE FORMULA:
    ------------
    spread = base_rating_spread
           + sector_adjustment
           + vix_adjustment          ← market fear → all spreads wider
           + market_spread_passthru  ← IG/HY market level from FRED
           + leverage_loading        ← more debt → wider spread
           + volatility_loading      ← higher eq vol → wider spread
           + idiosyncratic_noise     ← company-specific variation

    Each component has a direct financial interpretation and is calibrated
    to produce realistic spread levels.

    WHAT "PASS-THROUGH" MEANS:
    --------------------------
    The market OAS from FRED captures the average spread level across all
    IG or HY companies on a given date. Our company's spread moves with
    the market but not 1:1 — we use a 40-50% pass-through rate.
    This reflects the empirical finding that idiosyncratic factors explain
    roughly half of CDS spread variation, with systematic (market) factors
    explaining the other half.
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

        # Base spread from rating
        base = RATING_BASE_SPREAD.get(rating, 200)

        # Sector adjustment
        sect_adj = SECTOR_ADJUSTMENT.get(sector, 0)

        # ── Macro conditions on this date ────────────────────────────────────
        # We need the macro row for THIS quarter.
        # macro_df is indexed by quarter-end dates, so we find the
        # closest available macro date that is <= this quarter-end date.
        available_macro = macro_df[macro_df.index <= qdate]
        if len(available_macro) == 0:
            continue   # no macro data available yet — skip
        macro_row = available_macro.iloc[-1]

        # VIX adjustment: VIX=20 is "neutral" (no adjustment)
        # Each point above 20 adds 3 bps to all spreads
        # Each point below 20 subtracts 3 bps
        vix = macro_row.get("vix", 20)
        vix_adj = (vix - 20) * 3   # e.g. VIX=40 (COVID shock) → +60 bps

        # Market spread pass-through
        # IG companies track IG OAS; HY companies track HY OAS
        if rating <= 4:   # Investment Grade
            market_oas = macro_row.get("ig_oas", 100)
            # 100 bps is approx long-run average IG OAS
            # If market is tighter than average, our company's spread narrows
            market_adj = (market_oas - 100) * 0.40   # 40% pass-through
        else:             # High Yield
            market_oas = macro_row.get("hy_oas", 400)
            # 400 bps is approx long-run average HY OAS
            market_adj = (market_oas - 400) * 0.50   # 50% pass-through

        # ── Company-specific adjustments ─────────────────────────────────────

        leverage = row.get("leverage_ratio", np.nan)
        if pd.notna(leverage):
            # Leverage ratio 0.4 → +60 bps, 0.7 → +105 bps
            lev_adj = leverage * 150
        else:
            lev_adj = 0   # use zero if missing (mean-ish)

        vol = row.get("hist_volatility", np.nan)
        if pd.notna(vol):
            # Vol 0.30 (30%) → +60 bps, Vol 0.60 → +120 bps
            vol_adj = vol * 200
        else:
            vol_adj = 0

        # Small idiosyncratic noise: ±8% of base spread
        # Fixed seed ensures the same company always gets the same noise
        # so results are reproducible
        noise = rng.normal(0, base * 0.08)

        # ── Assemble final spread ─────────────────────────────────────────────
        spread = base + sect_adj + vix_adj + market_adj + lev_adj + vol_adj + noise
        spread = max(spread, 10)   # floor at 10 bps (spreads can't go negative)

        records.append({
            "ticker":         ticker,
            "date":           qdate,
            "cds_spread_bps": spread,
            # Store component breakdown for your report (useful to analyse)
            "_base":          base,
            "_sect_adj":      sect_adj,
            "_vix_adj":       vix_adj,
            "_market_adj":    market_adj,
            "_lev_adj":       lev_adj,
            "_vol_adj":       vol_adj,
        })

    df = pd.DataFrame(records).set_index(["ticker", "date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE EVERYTHING INTO ONE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def build_full_panel(company_universe: list,
                     equity_df: pd.DataFrame,
                     macro_df: pd.DataFrame,
                     cds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges equity features, macro features, static company info, and
    the CDS target into one flat DataFrame ready for the model.

    HOW THE MERGE WORKS:
    --------------------
    equity_df   → indexed by (ticker, date)
    macro_df    → indexed by (date) only — same macro applies to all companies
    cds_df      → indexed by (ticker, date)
    static info → from company_universe list (rating, sector, region per ticker)

    We merge them all onto the (ticker, date) MultiIndex.

    For macro: we need to "broadcast" macro data across all companies.
    For each (ticker, date) row in equity_df, we look up the macro values
    for that date. This is done with a merge on the date level.
    """

    # Step 1: add static company info (rating, sector, region)
    company_info_df = pd.DataFrame(company_universe).set_index("ticker")[
        ["rating", "sector", "region"]
    ]

    # equity_df has MultiIndex (ticker, date)
    # We merge company_info on the ticker level
    panel = equity_df.copy()

    # Reset index to merge, then re-set
    panel = panel.reset_index()
    panel = panel.merge(company_info_df.reset_index(), on="ticker", how="left")

    # Step 2: merge macro data on date
    # macro_df is quarterly, equity_df is also quarterly — dates should align
    macro_reset = macro_df.reset_index().rename(columns={"index": "date"})
    if "date" not in macro_reset.columns:
        # macro_df's index is named differently — find it
        macro_reset = macro_df.copy()
        macro_reset.index.name = "date"
        macro_reset = macro_reset.reset_index()

    panel = panel.merge(macro_reset, on="date", how="left")

    # Step 3: merge the CDS spread target
    cds_reset = cds_df.reset_index()[["ticker", "date", "cds_spread_bps"]]
    panel = panel.merge(cds_reset, on=["ticker", "date"], how="left")

    # Step 4: restore MultiIndex
    panel = panel.set_index(["ticker", "date"]).sort_index()

    # Drop spread component breakdown columns (prefixed with _)
    # These were just for analysis — not features for the model
    debug_cols = [c for c in panel.columns if c.startswith("_")]
    panel = panel.drop(columns=debug_cols)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL DATA QUALITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def final_quality_check(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a final quality check and produces the clean, model-ready panel.

    Checks:
    1. Missing values in target (cds_spread_bps) — rows with no target are useless
    2. Missing rate per feature — flag anything above 40%
    3. Sanity check on spread distribution
    4. Panel balance — are all companies represented in all quarters?
    """
    print("\n" + "="*60)
    print("FINAL DATA QUALITY CHECK")
    print("="*60)

    n_raw = len(panel)
    print(f"\nRaw panel: {n_raw} rows")

    # Drop rows with no target variable
    n_no_target = panel["cds_spread_bps"].isna().sum()
    panel = panel.dropna(subset=["cds_spread_bps"])
    print(f"Dropped {n_no_target} rows with missing target → {len(panel)} rows remain")

    # Missing values per feature
    print("\n── Missing rates per feature ──")
    threshold = 0.40
    cols_to_drop = []
    for col in panel.columns:
        pct = panel[col].isna().mean()
        flag = " ← DROP (>40% missing)" if pct > threshold else ""
        print(f"  {col:30s} {pct*100:5.1f}%{flag}")
        if pct > threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"\nDropping {len(cols_to_drop)} features with >40% missing: {cols_to_drop}")
        panel = panel.drop(columns=cols_to_drop)

    # Sanity check: spread distribution
    spreads = panel["cds_spread_bps"]
    print(f"\n── Spread distribution ──")
    print(f"  Min:    {spreads.min():.1f} bps")
    print(f"  Median: {spreads.median():.1f} bps")
    print(f"  Mean:   {spreads.mean():.1f} bps")
    print(f"  Max:    {spreads.max():.1f} bps")
    print(f"  Std:    {spreads.std():.1f} bps")

    # Count by rating
    print(f"\n── Observations by rating ──")
    rating_counts = panel.groupby(panel["rating"])["cds_spread_bps"].agg(["count", "mean"])
    rating_labels = {1:"AAA", 2:"AA", 3:"A", 4:"BBB", 5:"BB", 6:"B", 7:"CCC"}
    for rating, row in rating_counts.iterrows():
        label = rating_labels.get(int(rating), str(rating))
        print(f"  Rating {int(rating)} ({label}): {int(row['count']):3d} obs | "
              f"avg spread {row['mean']:.0f} bps")

    # Final dimensions
    tickers = panel.index.get_level_values("ticker").unique()
    dates   = panel.index.get_level_values("date").unique()
    features = [c for c in panel.columns if c != "cds_spread_bps"]

    print(f"\n── FINAL PANEL DIMENSIONS ──")
    print(f"  N (observations): {len(panel)}")
    print(f"  Companies:        {len(tickers)}")
    print(f"  Quarters:         {len(dates)}")
    print(f"  D (features):     {len(features)}")
    print(f"  Features:         {features}")

    return panel


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from equities import COMPANY_UNIVERSE

    print("\n" + "="*60)
    print("  STEP 3: BUILDING THE FULL PANEL DATASET")
    print("="*60 + "\n")

    # Load cached data from steps 1 and 2
    print("[1/4] Loading macro data...")
    macro_df = pd.read_csv(RAW_DIR / "macro_fred.csv", index_col=0, parse_dates=True)
    print(f"      {macro_df.shape[0]} quarters, {macro_df.shape[1]} macro features")

    print("\n[2/4] Loading equity data...")
    equity_df = pd.read_csv(RAW_DIR / "company_equity.csv",
                             index_col=[0, 1], parse_dates=True)
    print(f"      {len(equity_df)} company-quarter observations")

    print("\n[3/4] Building CDS spread proxy...")
    cds_df = build_cds_proxy(COMPANY_UNIVERSE, equity_df, macro_df)
    spread_path = RAW_DIR / "cds_spreads_proxy.csv"
    cds_df.to_csv(spread_path)
    print(f"      {len(cds_df)} spread observations")
    print(f"      Range: {cds_df['cds_spread_bps'].min():.0f} – "
          f"{cds_df['cds_spread_bps'].max():.0f} bps")
    print(f"      Saved to {spread_path}")

    print("\n[4/4] Merging into full panel...")
    panel = build_full_panel(COMPANY_UNIVERSE, equity_df, macro_df, cds_df)

    # Quality check and clean
    panel = final_quality_check(panel)

    # Save the final model-ready panel
    out_path = PROCESSED_DIR / "panel_final.csv"
    panel.to_csv(out_path)
    print(f"\n✓ Final panel saved to {out_path}")
    print(f"\nNext step: run step4_preprocess.py to encode and scale for the model.")