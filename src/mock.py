"""
tier1_svr.py
============
A complete, self-contained Tier 1 SVR implementation for CDS Spread Modelling.

Run this single file. It will:
  1. Build a synthetic but realistic dataset (no API keys needed)
  2. Preprocess it (encoding, imputation, scaling)
  3. Train an SVR with cross-validated hyperparameter tuning
  4. Evaluate it (RMSE, MAE, R²)
  5. Produce all required plots
  6. Analyse support vectors vs epsilon

No external data sources required. Replace the data section with
real data once you have it — the rest of the pipeline is identical.

Requirements:
    pip install scikit-learn numpy pandas matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
BLUE   = "#1F3864"
ORANGE = "#C9372C"
GREEN  = "#1A6B3A"
GREY   = "#888888"


# =============================================================================
#  SECTION 1: DATA
#  Build a synthetic but realistic dataset.
#  Each row = one company. Features drive the target in a financially
#  sensible way so the SVR has real structure to learn.
# =============================================================================

def build_dataset(n_companies: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic panel of companies with realistic CDS spread targets.

    Features:
        rating          : 1 (AAA) to 7 (CCC) — ordinal
        sector          : categorical (5 sectors)
        region          : categorical (4 regions)
        leverage_ratio  : total debt / total assets (0 to 1)
        hist_volatility : annualised equity vol (0.1 to 0.8)
        interest_coverage: EBIT / interest expense (0.5 to 15)
        roa             : return on assets (-0.05 to 0.20)
        vix             : market fear index (10 to 45)
        risk_free       : 10-year treasury yield % (1.5 to 5.0)

    Target:
        cds_spread_bps  : CDS spread in basis points
                          Constructed from features + noise so SVR can learn it.

    Some values are set to NaN intentionally to practise imputation.
    """
    rng = np.random.default_rng(seed)
    N = n_companies

    # ── Static features ───────────────────────────────────────────────────────
    rating = rng.integers(1, 8, size=N)          # 1=AAA ... 7=CCC
    sector = rng.choice(
        ["Financials", "Energy", "Technology", "Healthcare", "Consumer"],
        size=N
    )
    region = rng.choice(
        ["Northeast", "South", "Midwest", "West"],
        size=N
    )

    # ── Continuous features ───────────────────────────────────────────────────
    # Leverage increases with rating (higher rating number = worse credit)
    leverage = np.clip(
        0.15 + (rating - 1) * 0.06 + rng.normal(0, 0.08, N),
        0.05, 0.95
    )

    # Volatility also increases with rating
    vol = np.clip(
        0.12 + (rating - 1) * 0.04 + rng.normal(0, 0.05, N),
        0.08, 0.80
    )

    # Interest coverage decreases as credit quality worsens
    coverage = np.clip(
        12 - (rating - 1) * 1.5 + rng.normal(0, 1.5, N),
        0.3, 20.0
    )

    # ROA: profitable companies have better credit
    roa = np.clip(
        0.10 - (rating - 1) * 0.015 + rng.normal(0, 0.03, N),
        -0.08, 0.25
    )

    # Macro — same "market conditions" snapshot for all companies
    # In a real panel this would vary by date; here it's a cross-sectional slice
    vix       = rng.uniform(12, 38, N)
    risk_free = rng.uniform(1.5, 5.0, N)

    # ── Target: CDS spread in bps ─────────────────────────────────────────────
    # This formula mirrors the Tier 2 proxy construction but simplified.
    # Each component has a clear financial interpretation.
    rating_base = {1: 15, 2: 30, 3: 60, 4: 120, 5: 280, 6: 550, 7: 1100}
    base = np.array([rating_base[r] for r in rating], dtype=float)

    sector_adj = np.array([
        {"Financials": 20, "Energy": 35, "Technology": -10,
         "Healthcare": -5, "Consumer": 10}[s]
        for s in sector
    ], dtype=float)

    lev_component  = leverage * 150       # leverage_ratio 0.4 → +60bps
    vol_component  = vol * 200            # vol 0.30 → +60bps
    vix_component  = (vix - 20) * 3      # VIX 30 → +30bps
    cov_component  = -coverage * 8       # high coverage → tighter spread
    roa_component  = -roa * 300          # high ROA → tighter spread

    noise = rng.normal(0, base * 0.07)   # 7% idiosyncratic noise

    spread = (base + sector_adj + lev_component + vol_component +
              vix_component + cov_component + roa_component + noise)
    spread = np.clip(spread, 10, None)   # floor at 10bps

    # ── Introduce realistic missing values ────────────────────────────────────
    # ~12% of leverage_ratio missing (common with small/foreign companies)
    miss_lev = rng.choice(N, size=int(N * 0.12), replace=False)
    leverage[miss_lev] = np.nan

    # ~18% of interest_coverage missing
    miss_cov = rng.choice(N, size=int(N * 0.18), replace=False)
    coverage[miss_cov] = np.nan

    # ~8% of roa missing
    miss_roa = rng.choice(N, size=int(N * 0.08), replace=False)
    roa[miss_roa] = np.nan

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "rating":           rating.astype(float),
        "sector":           sector,
        "region":           region,
        "leverage_ratio":   leverage,
        "hist_volatility":  vol,
        "interest_coverage":coverage,
        "roa":              roa,
        "vix":              vix,
        "risk_free":        risk_free,
        "cds_spread_bps":   spread,
    })

    print(f"Dataset built: {df.shape[0]} companies, {df.shape[1]} columns")
    print(f"Spread range: {spread.min():.0f} — {spread.max():.0f} bps  "
          f"| Median: {np.median(spread):.0f} bps")
    print(f"Missing values: {df.isnull().sum().sum()} total across all features\n")
    return df


# =============================================================================
#  SECTION 2: PREPROCESSING
#  Clean the data and prepare it for SVR.
#  Every choice is documented with a justification.
# =============================================================================

def preprocess(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline.

    Returns:
        X_train_scaled, X_test_scaled : model-ready feature matrices
        y_train, y_test               : target vectors (log-transformed)
        scaler                        : fitted StandardScaler
        feature_names                 : list of column names in X
        back_transform                : function to convert predictions back to bps
    """
    print("=" * 55)
    print("PREPROCESSING")
    print("=" * 55)

    # ── Separate features and target ──────────────────────────────────────────
    y_raw = df["cds_spread_bps"].copy()
    X_raw = df.drop(columns=["cds_spread_bps"]).copy()

    # ── Step 1: Encode categorical variables ──────────────────────────────────
    #
    # rating: already numeric and ORDINAL — leave it as-is.
    #   A rating of 4 (BBB) genuinely sits between 3 (A) and 5 (BB).
    #   One-hot would destroy this ordering.
    #
    # sector, region: NOMINAL — no ordering. Use one-hot encoding.
    #   If we encoded sector as 1=Financials, 2=Energy, 3=Technology,
    #   the model would think Technology is "twice as far" from Financials
    #   as Energy is — which is meaningless.
    #   drop_first=True drops one dummy per group to avoid perfect
    #   multicollinearity (the dropped category becomes the baseline).

    X_encoded = pd.get_dummies(X_raw, columns=["sector", "region"], drop_first=True)
    feature_names = list(X_encoded.columns)

    print(f"\n[1/4] Categorical encoding:")
    print(f"      {len(feature_names)} features after one-hot encoding")
    print(f"      Features: {feature_names}")

    # ── Step 2: Train / test split ────────────────────────────────────────────
    #
    # 80/20 random split. For Tier 1 this is acceptable.
    # Tier 2 will replace this with a temporal split.
    # random_state fixes the split for reproducibility.

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_encoded, y_raw,
        test_size=0.2,
        random_state=42
    )
    print(f"\n[2/4] Train/test split (80/20):")
    print(f"      Train: {len(X_train_raw)} companies")
    print(f"      Test:  {len(X_test_raw)} companies")

    # ── Step 3: Impute missing values ─────────────────────────────────────────
    #
    # Strategy: median imputation for continuous features.
    # We use MEDIAN not MEAN because CDS spreads and financial ratios are
    # right-skewed. The mean is pulled up by extremes; the median is robust.
    #
    # IMPORTANT: fit the imputer on TRAIN only, apply to both.
    # If you fit on all data, the test set's values have influenced the
    # imputation — this is data leakage.
    #
    # Boolean/dummy columns will never be NaN, so imputation leaves them unchanged.

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw)
    X_train_imp = pd.DataFrame(
        imputer.transform(X_train_raw),
        columns=feature_names,
        index=X_train_raw.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=feature_names,
        index=X_test_raw.index
    )

    n_missing_before = X_train_raw.isnull().sum().sum() + X_test_raw.isnull().sum().sum()
    print(f"\n[3/4] Missing value imputation (median, fit on train only):")
    print(f"      Imputed {n_missing_before} missing values")
    print(f"      Remaining NaNs: {X_train_imp.isnull().sum().sum()}")

    # ── Step 4: Scale features ────────────────────────────────────────────────
    #
    # SVR uses ||xᵢ - xⱼ||² (Euclidean distance) inside its kernel.
    # Without scaling: leverage_ratio ranges 0-1, vix ranges 12-40.
    # The vix dominates all distance calculations — leverage ratio becomes
    # invisible to the model.
    #
    # StandardScaler: x_scaled = (x - mean) / std
    # After this, every feature has mean=0 and std=1.
    #
    # We DON'T scale the dummy variables (0/1 columns) — they're already
    # on a bounded, comparable scale. Scaling them would distort them.
    # We identify dummies as columns with only two unique values.
    #
    # Again: fit on TRAIN only.

    dummy_cols = [c for c in feature_names if X_train_imp[c].nunique() <= 2]
    scale_cols = [c for c in feature_names if c not in dummy_cols]

    scaler = StandardScaler()
    scaler.fit(X_train_imp[scale_cols])

    X_train_scaled = X_train_imp.copy()
    X_test_scaled  = X_test_imp.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train_imp[scale_cols])
    X_test_scaled[scale_cols]  = scaler.transform(X_test_imp[scale_cols])

    print(f"\n[4/4] StandardScaler (fit on train only):")
    print(f"      Scaled {len(scale_cols)} continuous features")
    print(f"      Left {len(dummy_cols)} dummy variables unscaled")

    # ── Target transformation ─────────────────────────────────────────────────
    #
    # CDS spreads are right-skewed: most 50-200bps, a few at 800-1500bps.
    # SVR tries to minimise prediction error uniformly — without log transform,
    # it wastes capacity fitting the rare extreme observations.
    #
    # log1p = log(1 + x) — handles values close to zero safely.
    # Back-transform: expm1(y_pred) = exp(y_pred) - 1.

    y_train = np.log1p(y_train_raw)
    y_test  = np.log1p(y_test_raw)
    back_transform = np.expm1

    print(f"\nTarget log-transformed.")
    print(f"  Raw spread range:  [{y_train_raw.min():.0f}, {y_train_raw.max():.0f}] bps")
    print(f"  Log-space range:   [{y_train.min():.2f}, {y_train.max():.2f}]")

    return (X_train_scaled, X_test_scaled,
            y_train, y_test,
            y_train_raw, y_test_raw,
            scaler, feature_names, back_transform)


# =============================================================================
#  SECTION 3: HYPERPARAMETER TUNING
#  Find the best C, epsilon, and gamma via cross-validated grid search.
# =============================================================================

def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series) -> dict:
    """
    Finds the best SVR hyperparameters using GridSearchCV with 5-fold CV.

    The three hyperparameters we tune:

    C — Regularisation strength.
        Controls the tradeoff between keeping ||w||² small (simple model)
        and minimising training errors.
        Small C → simpler model, allows more errors.
        Large C → fits training data harder, risks overfitting.

    epsilon (ε) — The tube half-width.
        Errors within ε are completely ignored (zero loss).
        Large ε → more points inside tube → fewer support vectors → simpler model.
        Small ε → tight tube → nearly every point is a support vector.

    gamma — Controls the RBF kernel's "reach".
        RBF kernel: K(xᵢ, xⱼ) = exp(-gamma × ||xᵢ - xⱼ||²)
        Large gamma → kernel drops off fast → each point only influences
                      its immediate neighbours → can overfit.
        Small gamma → smooth, wide-reaching kernel → simpler decision surface.

    We use negative MSE as the scoring metric (GridSearchCV maximises,
    so we negate MSE to get it to prefer lower errors).
    """
    print("\n" + "=" * 55)
    print("HYPERPARAMETER TUNING (5-fold cross-validation)")
    print("=" * 55)

    param_grid = {
        "C":       [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "gamma":   ["scale", 0.01, 0.1, 1.0],
    }

    n_combinations = (len(param_grid["C"]) *
                      len(param_grid["epsilon"]) *
                      len(param_grid["gamma"]))
    print(f"\nSearching {n_combinations} combinations × 5 folds = "
          f"{n_combinations * 5} SVR fits...")
    print("(This takes ~30-60 seconds on a typical laptop)")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    svr = SVR(kernel="rbf", max_iter=10000)
    grid_search = GridSearchCV(
        svr,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,       # Use all CPU cores
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)   # RMSE in log-space

    print(f"\nBest parameters found:")
    print(f"  C       = {best['C']}")
    print(f"  epsilon = {best['epsilon']}")
    print(f"  gamma   = {best['gamma']}")
    print(f"  CV RMSE (log-space) = {best_score:.4f}")

    return best, grid_search


# =============================================================================
#  SECTION 4: TRAIN FINAL MODEL & EVALUATE
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       y_train_raw, y_test_raw,
                       best_params, back_transform) -> dict:
    """
    Trains the final SVR with best params and evaluates on the test set.

    We report metrics in TWO spaces:
      - Log space: what the model actually optimises
      - BPS space: what matters financially (back-transformed)

    Always report in the original units in your report — log-space RMSE
    is meaningless to a reader who thinks in basis points.
    """
    print("\n" + "=" * 55)
    print("TRAINING FINAL MODEL & EVALUATION")
    print("=" * 55)

    model = SVR(kernel="rbf", **best_params, max_iter=10000)
    model.fit(X_train, y_train)

    # Predictions in log-space
    y_pred_log_train = model.predict(X_train)
    y_pred_log_test  = model.predict(X_test)

    # Back-transform to basis points
    y_pred_train_bps = back_transform(y_pred_log_train)
    y_pred_test_bps  = back_transform(y_pred_log_test)

    # ── Metrics in bps (report these in your report) ──────────────────────────
    rmse_train = np.sqrt(mean_squared_error(y_train_raw, y_pred_train_bps))
    rmse_test  = np.sqrt(mean_squared_error(y_test_raw,  y_pred_test_bps))
    mae_train  = mean_absolute_error(y_train_raw, y_pred_train_bps)
    mae_test   = mean_absolute_error(y_test_raw,  y_pred_test_bps)
    r2_train   = r2_score(y_train_raw, y_pred_train_bps)
    r2_test    = r2_score(y_test_raw,  y_pred_test_bps)

    print(f"\n  {'Metric':<10} {'Train':>10} {'Test':>10}")
    print(f"  {'-'*32}")
    print(f"  {'RMSE':<10} {rmse_train:>9.1f}  {rmse_test:>9.1f}  bps")
    print(f"  {'MAE':<10} {mae_train:>9.1f}  {mae_test:>9.1f}  bps")
    print(f"  {'R²':<10} {r2_train:>9.3f}  {r2_test:>9.3f}")

    # ── Support vector info ───────────────────────────────────────────────────
    n_sv = model.n_support_[0]   # SVR has only one class, so index [0]
    pct_sv = n_sv / len(X_train) * 100
    print(f"\n  Support vectors: {n_sv} / {len(X_train)} training points ({pct_sv:.1f}%)")
    print(f"  (With epsilon={best_params['epsilon']}, {100-pct_sv:.1f}% of points "
          f"are inside the ε-tube and contribute nothing to the model)")

    results = {
        "model":             model,
        "y_pred_test_bps":   y_pred_test_bps,
        "y_pred_train_bps":  y_pred_train_bps,
        "rmse_test":         rmse_test,
        "mae_test":          mae_test,
        "r2_test":           r2_test,
        "n_sv":              n_sv,
        "pct_sv":            pct_sv,
    }
    return results


# =============================================================================
#  SECTION 5: PLOTS
#  All three required plots with proper labelling and interpretation hints.
# =============================================================================

def plot_predicted_vs_actual(y_true, y_pred, title_suffix="Test Set"):
    """
    Plot 1: Predicted vs Actual CDS Spreads.

    A perfect model would have all points on the 45-degree line.
    Systematic deviation above the line = model underpredicts those companies.
    Systematic deviation below = model overpredicts.
    Spread of points around the line = random error (irreducible or model noise).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.7, color=BLUE, edgecolors="white",
               linewidth=0.5, s=60, label="Companies")

    # 45-degree reference line (perfect predictions)
    lims = [min(y_true.min(), y_pred.min()) * 0.9,
            max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, color=ORANGE, linewidth=2, linestyle="--",
            label="Perfect prediction (45°)")

    # Annotate R²
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.93, f"R² = {r2:.3f}\nRMSE = {rmse:.1f} bps",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual CDS Spread (bps)", fontsize=12)
    ax.set_ylabel("Predicted CDS Spread (bps)", fontsize=12)
    ax.set_title(f"Predicted vs Actual CDS Spreads — {title_suffix}", fontsize=13, color=BLUE)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    path = OUTPUT_DIR / "plot1_pred_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


def plot_residuals(y_true, y_pred):
    """
    Plot 2: Residuals vs Predicted Values.

    Residual = actual - predicted.
    Positive residual = model UNDERPREDICTED (actual was higher).
    Negative residual = model OVERPREDICTED.

    What to look for:
      - Random scatter around zero = good, no systematic bias.
      - Fan shape (residuals grow with predicted value) = heteroscedasticity.
        This would mean the model is less accurate for high-spread companies.
      - Curved pattern = the model is missing a nonlinear relationship.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: residuals vs predicted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.7, color=BLUE,
               edgecolors="white", linewidth=0.5, s=60)
    ax.axhline(0, color=ORANGE, linewidth=2, linestyle="--", label="Zero error")
    ax.set_xlabel("Predicted CDS Spread (bps)", fontsize=12)
    ax.set_ylabel("Residual: Actual − Predicted (bps)", fontsize=12)
    ax.set_title("Residuals vs Predicted Values", fontsize=13, color=BLUE)
    ax.legend()

    # Annotate mean and std of residuals
    ax.text(0.05, 0.93,
            f"Mean residual: {residuals.mean():.1f} bps\nStd: {residuals.std():.1f} bps",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # Right: distribution of residuals
    ax2 = axes[1]
    ax2.hist(residuals, bins=20, color=BLUE, alpha=0.7, edgecolor="white")
    ax2.axvline(0, color=ORANGE, linewidth=2, linestyle="--")
    ax2.axvline(residuals.mean(), color=GREEN, linewidth=2, linestyle="-",
                label=f"Mean = {residuals.mean():.1f}")
    ax2.set_xlabel("Residual (bps)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Residuals", fontsize=13, color=BLUE)
    ax2.legend()

    plt.suptitle("Residual Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "plot2_residuals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


def plot_epsilon_vs_support_vectors(X_train, y_train, best_params):
    """
    Plot 3: How the number of support vectors changes with epsilon (ε).

    This is the core theoretical insight from lectures:
      - Small ε → tight tube → most points fall OUTSIDE the tube
        → most points become support vectors
        → model is complex, uses lots of training points
      - Large ε → wide tube → most points INSIDE the tube, ignored
        → few support vectors → simpler model

    This plot makes the ε-tube concept tangible with your own data.
    You should see a clear decreasing relationship.
    """
    print("\n  Computing support vector counts for different ε values...")

    epsilons = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    sv_counts = []
    sv_pcts   = []

    for eps in epsilons:
        m = SVR(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"],
                epsilon=eps, max_iter=10000)
        m.fit(X_train, y_train)
        n_sv = m.n_support_[0]
        sv_counts.append(n_sv)
        sv_pcts.append(n_sv / len(X_train) * 100)
        print(f"    ε={eps:.3f}  →  {n_sv} SVs ({n_sv/len(X_train)*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute count
    ax = axes[0]
    ax.plot(epsilons, sv_counts, "o-", color=BLUE, linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Number of Support Vectors", fontsize=12)
    ax.set_title("Support Vectors vs ε", fontsize=13, color=BLUE)
    ax.set_xscale("log")

    # Mark best epsilon
    best_eps = best_params["epsilon"]
    best_idx = min(range(len(epsilons)), key=lambda i: abs(epsilons[i] - best_eps))
    ax.axvline(best_eps, color=ORANGE, linestyle="--", linewidth=2,
               label=f"Best ε = {best_eps}")
    ax.scatter([best_eps], [sv_counts[best_idx]], color=ORANGE, s=120, zorder=5)
    ax.legend()

    # Right: percentage
    ax2 = axes[1]
    ax2.fill_between(epsilons, sv_pcts, alpha=0.2, color=BLUE)
    ax2.plot(epsilons, sv_pcts, "o-", color=BLUE, linewidth=2.5,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax2.axhline(100, color=GREY, linestyle=":", linewidth=1.5, label="100% (all points)")
    ax2.axvline(best_eps, color=ORANGE, linestyle="--", linewidth=2,
                label=f"Best ε = {best_eps}")
    ax2.set_xlabel("Epsilon (ε)", fontsize=12)
    ax2.set_ylabel("Support Vectors as % of Training Data", fontsize=12)
    ax2.set_title("% of Training Points Used by Model", fontsize=13, color=BLUE)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 110)
    ax2.legend()

    plt.suptitle(
        "Effect of ε-tube Width on Model Complexity\n"
        "Wider tube → fewer support vectors → simpler model",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "plot3_epsilon_vs_sv.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


def plot_support_vector_analysis(model, X_train, y_train_raw, df_train):
    """
    Bonus Plot: Who are the support vectors?

    Breaks down support vectors by rating bucket and sector.
    This connects to a key question in your report: do support vectors
    correspond to particular types of companies?

    The theory says: support vectors are the companies ON or OUTSIDE
    the ε-tube boundary. You'd expect these to be the "unusual" companies
    — those whose spread can't be well-predicted by the smooth model.
    """
    sv_indices = model.support_         # Indices into the training array
    is_sv = np.zeros(len(X_train), dtype=bool)
    is_sv[sv_indices] = True

    df_analysis = df_train.copy()
    df_analysis["is_support_vector"] = is_sv
    df_analysis["cds_spread_bps"] = y_train_raw.values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── By rating ─────────────────────────────────────────────────────────────
    ax = axes[0]
    sv_by_rating    = df_analysis.groupby("rating")["is_support_vector"].mean() * 100
    total_by_rating = df_analysis.groupby("rating")["is_support_vector"].count()

    colors = [GREEN if pct < 60 else ORANGE if pct < 80 else ORANGE
              for pct in sv_by_rating]
    bars = ax.bar(sv_by_rating.index, sv_by_rating.values, color=BLUE, alpha=0.75,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(is_sv.mean() * 100, color=ORANGE, linestyle="--", linewidth=2,
               label=f"Overall avg ({is_sv.mean()*100:.1f}%)")
    ax.set_xlabel("Credit Rating (1=AAA → 7=CCC)", fontsize=12)
    ax.set_ylabel("% that are Support Vectors", fontsize=12)
    ax.set_title("Support Vector Rate by Rating", fontsize=13, color=BLUE)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(["AAA\n(1)", "AA\n(2)", "A\n(3)", "BBB\n(4)",
                         "BB\n(5)", "B\n(6)", "CCC\n(7)"])
    ax.legend()
    ax.set_ylim(0, 110)

    # ── Spread distribution: SVs vs non-SVs ───────────────────────────────────
    ax2 = axes[1]
    sv_spreads    = df_analysis.loc[df_analysis["is_support_vector"], "cds_spread_bps"]
    nonsv_spreads = df_analysis.loc[~df_analysis["is_support_vector"], "cds_spread_bps"]

    ax2.hist(nonsv_spreads, bins=20, alpha=0.6, color=BLUE, label="Non-support vectors",
             edgecolor="white")
    ax2.hist(sv_spreads, bins=20, alpha=0.6, color=ORANGE, label="Support vectors",
             edgecolor="white")
    ax2.set_xlabel("CDS Spread (bps)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Spread Distribution: Support Vectors vs Others", fontsize=13, color=BLUE)
    ax2.legend(fontsize=10)

    plt.suptitle("Who Are the Support Vectors?", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "plot4_sv_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


# =============================================================================
#  SECTION 6: SUMMARY TABLE
# =============================================================================

def print_summary(best_params, results, feature_names):
    """Prints a clean final summary suitable for copying into your report."""
    print("\n" + "=" * 55)
    print("FINAL SUMMARY")
    print("=" * 55)
    print(f"\nModel:   SVR with RBF kernel")
    print(f"Dataset: {len(feature_names)} features, 120 companies")
    print(f"\nBest hyperparameters:")
    print(f"  C       = {best_params['C']}")
    print(f"  ε       = {best_params['epsilon']}")
    print(f"  gamma   = {best_params['gamma']}")
    print(f"\nTest set performance:")
    print(f"  RMSE = {results['rmse_test']:.1f} bps")
    print(f"  MAE  = {results['mae_test']:.1f} bps")
    print(f"  R²   = {results['r2_test']:.3f}")
    print(f"\nModel complexity:")
    print(f"  Support vectors: {results['n_sv']} / 96 training points "
          f"({results['pct_sv']:.1f}%)")
    print(f"  → {100 - results['pct_sv']:.1f}% of points are inside the ε-tube "
          f"and contribute zero to the prediction function")
    print(f"\nAll plots saved to: {OUTPUT_DIR.resolve()}")
    print(f"\nWhat to write in your report:")
    print(f"  - Quote RMSE and MAE in bps (not log-space)")
    print(f"  - Explain why R²={results['r2_test']:.2f} means what it means")
    print(f"  - Discuss the support vector percentage and what it reveals")
    print(f"  - Reference Plot 3 when discussing the ε-tube theory")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  TIER 1 — CDS SPREAD MODELLING WITH SVR")
    print("=" * 55 + "\n")

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    df = build_dataset(n_companies=120, seed=42)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    (X_train, X_test,
     y_train, y_test,
     y_train_raw, y_test_raw,
     scaler, feature_names, back_transform) = preprocess(df)

    # ── 3. Tune hyperparameters ───────────────────────────────────────────────
    best_params, grid_search = tune_hyperparameters(X_train, y_train)

    # ── 4. Train final model & evaluate ──────────────────────────────────────
    results = train_and_evaluate(
        X_train, X_test,
        y_train, y_test,
        y_train_raw, y_test_raw,
        best_params, back_transform
    )

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("GENERATING PLOTS")
    print("=" * 55)

    print("\nPlot 1: Predicted vs Actual")
    plot_predicted_vs_actual(y_test_raw, results["y_pred_test_bps"])

    print("\nPlot 2: Residuals")
    plot_residuals(y_test_raw, results["y_pred_test_bps"])

    print("\nPlot 3: Epsilon vs Support Vectors")
    plot_epsilon_vs_support_vectors(X_train, y_train, best_params)

    print("\nPlot 4 (Bonus): Support Vector Analysis")
    # Reconstruct df_train for the bonus plot (original features, unscaled)
    train_idx = y_train_raw.index
    df_train_orig = df.loc[train_idx, ["rating", "sector", "region"]]
    plot_support_vector_analysis(
        results["model"], X_train, y_train_raw, df_train_orig
    )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print_summary(best_params, results, feature_names)