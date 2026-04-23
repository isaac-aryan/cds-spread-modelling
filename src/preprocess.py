"""
step4_preprocess.py
===================
Takes the raw panel from step3 and produces a model-ready dataset.

Run after step3_build_panel.py:
    python src/step4_preprocess.py

This file covers:
    - Why each preprocessing step matters for SVR specifically
    - How to do a temporal train/test split (the correct way for panel data)
    - Imputation strategy with justification
    - Scaling with a clear explanation of what it does mathematically
    - Target transformation and why it helps
    - How to save the preprocessed data so your model file stays clean

Every choice here needs to be justified in your Tier 1 report.
This file gives you the justification for each.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: CATEGORICAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical variables for use in SVR.

    SVR (and most ML models) require all inputs to be numeric.
    We have three categorical-ish columns:

    RATING — Ordinal encoding (keep as number 1-7)
    -------
    Rating has a genuine order: AAA=1 is better than AA=2, which is
    better than A=3, etc. The numeric encoding 1-7 preserves this ordering.
    If we used one-hot encoding, we'd create 7 binary columns and lose the
    information that BBB (4) sits between A (3) and BB (5) in risk terms.
    The spec explicitly asks you to justify this choice — this is your answer.

    SECTOR — One-hot encoding
    ------
    Sectors have no natural ordering. Financials is not "greater than"
    Technology in any meaningful sense. If we encoded them as integers
    (Financials=1, Energy=2, Technology=3...), SVR's kernel would compute
    distances treating Technology as "twice as far" from Financials as Energy
    is. That's meaningless. One-hot creates a binary column per category —
    the model treats each sector as genuinely independent.

    REGION — One-hot encoding
    ------
    Same reasoning as sector. Geographic regions have no ordering.

    DROP_FIRST=True:
    ----------------
    With 5 sectors, we create 4 dummy columns (dropping one).
    The dropped category becomes the "baseline." If ALL 5 columns were
    included, knowing 4 of them perfectly predicts the 5th — this causes
    perfect multicollinearity, which can destabilise model fitting.
    Dropping one removes this without losing any information.
    """
    df = df.copy()

    print("  Encoding categoricals:")
    print(f"    rating: kept as ordinal numeric (1=AAA → 7=CCC)")

    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=["sector", "region"], drop_first=True)
    after_cols = df.shape[1]

    new_cols = [c for c in df.columns if c.startswith("sector_") or c.startswith("region_")]
    print(f"    sector, region: one-hot encoded → {len(new_cols)} new binary columns")
    print(f"    Total: {before_cols} → {after_cols} columns")

    # Convert bool columns (from get_dummies) to int (0/1)
    # SVR works with floats, not booleans
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def temporal_train_test_split(panel: pd.DataFrame,
                              cutoff_date: str = "2022-06-30"
                              ) -> tuple:
    """
    Splits data into train (pre-cutoff) and test (post-cutoff) sets.

    WHY TEMPORAL SPLIT, NOT RANDOM SPLIT:
    ---------------------------------------
    Your data has a time dimension — each row corresponds to a quarter.
    A random split might put Q3 2020 in the training set and Q1 2020
    in the test set. This means the model trained on data from AFTER
    the test period. That's look-ahead bias — in real deployment, you
    can't know the future.

    Temporal split ensures:
    - Training set: everything before the cutoff date
    - Test set: everything from the cutoff date onward
    This simulates real-world deployment: train on history, predict the future.

    CUTOFF CHOICE (2022-06-30):
    ----------------------------
    This gives ~4.5 years of training (2018-2022) and ~1.5 years of testing
    (2022-2024). Roughly 75/25 by time. The split falls after the COVID
    recovery period, so the test set is "normal" conditions rather than
    the extreme stress of 2020 — which is appropriate for baseline evaluation.

    For Tier 2, you'll replace this with walk-forward validation, which rolls
    the cutoff date forward quarter by quarter.

    PARAMETERS:
    -----------
    panel       : the full panel DataFrame with (ticker, date) MultiIndex
    cutoff_date : everything before this date goes into train
    """
    dates = panel.index.get_level_values("date")
    cutoff = pd.Timestamp(cutoff_date)

    train = panel[dates < cutoff].copy()
    test  = panel[dates >= cutoff].copy()

    n_train_quarters = train.index.get_level_values("date").nunique()
    n_test_quarters  = test.index.get_level_values("date").nunique()

    print(f"\n  Temporal train/test split (cutoff: {cutoff_date}):")
    print(f"    Train: {len(train):4d} obs | "
          f"{n_train_quarters} quarters | "
          f"{train.index.get_level_values('date').min().date()} → "
          f"{train.index.get_level_values('date').max().date()}")
    print(f"    Test:  {len(test):4d} obs | "
          f"{n_test_quarters} quarters | "
          f"{test.index.get_level_values('date').min().date()} → "
          f"{test.index.get_level_values('date').max().date()}")

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   strategy: str = "median") -> tuple:
    """
    Fills in missing values in feature matrices.

    WHY NOT JUST DROP ROWS?
    ------------------------
    If we drop every row with any missing value, we might lose 30-40% of
    our data. Worse, the rows we'd drop aren't random — companies that don't
    report certain ratios tend to be smaller or non-US, so dropping them
    introduces selection bias. The model would be trained only on
    well-reporting companies and fail on others.

    WHY MEDIAN, NOT MEAN?
    ----------------------
    Financial ratios are right-skewed. For example, interest_coverage:
    most companies have coverage of 3-8×, but a handful have 50×+.
    The MEAN is pulled up by those outliers. The MEDIAN is the middle value —
    it's robust to extreme observations and better represents "typical."
    Replacing a missing coverage ratio with the median coverage (say, 5×) is
    more reasonable than replacing it with a mean inflated to 12× by outliers.

    WHY FIT ONLY ON TRAINING DATA?
    --------------------------------
    If we computed the median using both train AND test data, then information
    from the test set has leaked into our preprocessing. The test set is meant
    to simulate unseen future data. In practice, you wouldn't have access to
    future data when computing your imputation medians. Fitting on train only
    keeps the test set truly "unseen."

    PARAMETERS:
    -----------
    X_train, X_test : feature matrices (DataFrames)
    strategy        : "median" (default) or "mean"

    RETURNS:
    --------
    X_train_imp, X_test_imp : imputed DataFrames
    imputer                 : fitted imputer object (save this for deployment)
    """
    print(f"\n  Imputation strategy: {strategy} (fit on train only)")

    n_missing_train = X_train.isna().sum().sum()
    n_missing_test  = X_test.isna().sum().sum()
    print(f"    Missing before: {n_missing_train} in train, {n_missing_test} in test")

    imputer = SimpleImputer(strategy=strategy)

    # fit() computes the median for each column from training data only
    imputer.fit(X_train)

    # transform() fills NaN values using the computed medians
    # Applied to BOTH train and test, but using train's medians for both
    X_train_imp = pd.DataFrame(
        imputer.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"    Missing after:  {X_train_imp.isna().sum().sum()} in train, "
          f"{X_test_imp.isna().sum().sum()} in test")

    return X_train_imp, X_test_imp, imputer


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: SCALING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    """
    Standardises continuous features to zero mean and unit variance.

    THE MATHEMATICAL TRANSFORMATION:
    ---------------------------------
    For each continuous feature x:
        x_scaled = (x - mean(x)) / std(x)

    After scaling:
        mean(x_scaled) = 0
        std(x_scaled) = 1

    WHY THIS IS ESSENTIAL FOR SVR:
    --------------------------------
    SVR uses a kernel function to compute "similarity" between data points.
    For the RBF kernel:
        K(xᵢ, xⱼ) = exp(-gamma × ||xᵢ - xⱼ||²)

    ||xᵢ - xⱼ||² is the squared Euclidean distance — the sum of squared
    differences across all features.

    Example WITHOUT scaling:
        hist_volatility: ranges 0.1 to 0.8    → max difference ≈ 0.7
        vix:             ranges 12 to 45       → max difference ≈ 33
        leverage_ratio:  ranges 0.1 to 0.9     → max difference ≈ 0.8

    In the distance calculation, VIX contributes 33² = 1,089 to the distance,
    while volatility contributes only 0.7² = 0.49.
    The model effectively IGNORES volatility and leverage — they're drowned
    out by VIX. This is mathematically wrong: all features should get equal
    opportunity to influence the kernel.

    Example WITH scaling:
        All features have std=1, so max differences are comparable.
        A 1-std difference in VIX has the same weight as a 1-std difference
        in volatility. The model can actually use all features.

    WHAT WE DON'T SCALE:
    --------------------
    Binary (dummy) columns from one-hot encoding (0 or 1).
    These are already on a bounded, comparable scale. Scaling them would
    change their interpretation (a "1" would no longer mean "yes, this sector").
    We identify dummy columns as those with only 2 unique values.
    """
    print(f"\n  Feature scaling (StandardScaler, fit on train only):")

    # Identify which columns are dummies (binary) vs continuous
    dummy_cols = [c for c in X_train.columns if X_train[c].nunique() <= 2]
    scale_cols = [c for c in X_train.columns if c not in dummy_cols]

    print(f"    Scaling {len(scale_cols)} continuous features")
    print(f"    Leaving {len(dummy_cols)} binary features unscaled")
    print(f"    Continuous: {scale_cols}")

    scaler = StandardScaler()
    scaler.fit(X_train[scale_cols])

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test_scaled[scale_cols]  = scaler.transform(X_test[scale_cols])

    # Print what the scaler learned from training data
    print(f"\n    Scaler parameters (from training data):")
    print(f"    {'Feature':25s} {'Mean':>10s} {'Std':>10s}")
    print(f"    {'-'*47}")
    for feature, mean, std in zip(scale_cols, scaler.mean_, scaler.scale_):
        print(f"    {feature:25s} {mean:10.3f} {std:10.3f}")

    return X_train_scaled, X_test_scaled, scaler, scale_cols


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5: TARGET TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def transform_target(y_train: pd.Series,
                     y_test: pd.Series) -> tuple:
    """
    Log-transforms the CDS spread target variable.

    THE DISTRIBUTION PROBLEM:
    --------------------------
    CDS spreads are heavily right-skewed:
        AAA: 15-30 bps    (most common)
        AA:  25-50 bps
        A:   50-100 bps
        BBB: 100-200 bps
        BB:  200-400 bps   (moderately common)
        B:   400-800 bps
        CCC: 800-2000 bps  (rare, but extreme)

    SVR minimises prediction error UNIFORMLY across all training points.
    Without transformation, a single CCC company with spread=1500 bps
    contributes as much to the loss as 50 BBB companies at 120 bps.
    The model wastes capacity fitting that one extreme case at the expense
    of accuracy for the typical company.

    AFTER LOG TRANSFORMATION:
    --------------------------
    log(15)  ≈ 2.7    log(30)  ≈ 3.4
    log(120) ≈ 4.8    log(400) ≈ 6.0
    log(1500)≈ 7.3

    The distribution becomes much more symmetric. The model can now fit
    all rating buckets with approximately equal precision.

    WHY log1p (log(1+x)) INSTEAD OF log(x)?
    -----------------------------------------
    log1p is defined for x=0, whereas log(0) = -infinity.
    For small spread values (e.g. 10 bps), log(10) ≈ 2.3 and log1p(10) ≈ 2.4
    — nearly identical, but log1p is safe for values close to zero.

    BACK-TRANSFORMATION:
    --------------------
    Predictions come out in log-space. To get bps, apply expm1:
        bps = expm1(log_pred) = exp(log_pred) - 1

    IMPORTANT: Always report RMSE and MAE in BASIS POINTS (back-transformed).
    Log-space RMSE is mathematically convenient but meaningless to a reader.
    """
    print(f"\n  Target transformation: log1p(cds_spread_bps)")
    print(f"    Train spread range: {y_train.min():.0f} → {y_train.max():.0f} bps")
    print(f"    Train spread median: {y_train.median():.0f} bps")

    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    print(f"    After log1p: {y_train_log.min():.2f} → {y_train_log.max():.2f}")

    # Return the back-transform function alongside the transformed series
    back_transform = np.expm1

    return y_train_log, y_test_log, y_train, y_test, back_transform


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(panel: pd.DataFrame,
                      cutoff_date: str = "2022-06-30") -> dict:
    """
    Runs the complete preprocessing pipeline and returns all components.

    Saves processed train/test CSVs so the model file doesn't need to
    re-run preprocessing on every run.
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)

    # Separate features and target
    target_col   = "cds_spread_bps"
    feature_cols = [c for c in panel.columns if c != target_col]

    print(f"\n  Target: {target_col}")
    print(f"  Features: {len(feature_cols)} columns")

    # Step 1: encode categoricals
    print("\n[1/5] Encoding categoricals...")
    panel_encoded = encode_categoricals(panel)

    # Re-separate after encoding (new dummy columns added)
    feature_cols_encoded = [c for c in panel_encoded.columns if c != target_col]
    X_all = panel_encoded[feature_cols_encoded]
    y_all = panel_encoded[target_col]

    # Step 2: temporal split
    print("\n[2/5] Temporal train/test split...")
    # Split the full encoded panel first
    dates = panel_encoded.index.get_level_values("date")
    cutoff = pd.Timestamp(cutoff_date)
    train_mask = dates < cutoff
    test_mask  = dates >= cutoff

    X_train_raw = X_all[train_mask]
    X_test_raw  = X_all[test_mask]
    y_train_raw = y_all[train_mask]
    y_test_raw  = y_all[test_mask]

    n_train_q = X_train_raw.index.get_level_values("date").nunique()
    n_test_q  = X_test_raw.index.get_level_values("date").nunique()
    print(f"    Train: {len(X_train_raw)} obs ({n_train_q} quarters)")
    print(f"    Test:  {len(X_test_raw)} obs ({n_test_q} quarters)")

    # Step 3: impute
    print("\n[3/5] Imputing missing values...")
    X_train_imp, X_test_imp, imputer = impute_missing(X_train_raw, X_test_raw)

    # Step 4: scale
    print("\n[4/5] Scaling features...")
    X_train_scaled, X_test_scaled, scaler, scale_cols = scale_features(
        X_train_imp, X_test_imp
    )

    # Step 5: transform target
    print("\n[5/5] Transforming target...")
    y_train, y_test, y_train_bps, y_test_bps, back_transform = transform_target(
        y_train_raw, y_test_raw
    )

    # Save processed data
    X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv")
    X_test_scaled.to_csv(PROCESSED_DIR / "X_test.csv")
    y_train.to_csv(PROCESSED_DIR / "y_train.csv")
    y_test.to_csv(PROCESSED_DIR / "y_test.csv")
    y_train_bps.to_csv(PROCESSED_DIR / "y_train_bps.csv")
    y_test_bps.to_csv(PROCESSED_DIR / "y_test_bps.csv")

    print(f"\n  All processed data saved to {PROCESSED_DIR}/")

    return {
        "X_train":       X_train_scaled,
        "X_test":        X_test_scaled,
        "y_train":       y_train,         # log-transformed
        "y_test":        y_test,           # log-transformed
        "y_train_bps":   y_train_bps,     # original bps
        "y_test_bps":    y_test_bps,      # original bps
        "imputer":       imputer,
        "scaler":        scaler,
        "scale_cols":    scale_cols,
        "feature_names": list(X_train_scaled.columns),
        "back_transform":back_transform,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STEP 4: PREPROCESSING")
    print("="*60 + "\n")

    panel = pd.read_csv(
        PROCESSED_DIR / "panel_final.csv",
        index_col=[0, 1], parse_dates=True
    )
    print(f"Loaded panel: {panel.shape}")

    data = run_preprocessing(panel)

    print(f"\n── FINAL PREPROCESSED DIMENSIONS ──")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_test:  {data['X_test'].shape}")
    print(f"  Features: {data['feature_names']}")

    print("\n✓ Step 4 complete. Run step5_model.py to train SVR.")