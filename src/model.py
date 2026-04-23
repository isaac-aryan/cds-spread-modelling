"""
step5_model.py
==============
Tier 1 SVR model for CDS Spread prediction.

Loads preprocessed data from data/processed/, trains an SVR with
cross-validated hyperparameter tuning, evaluates it, and produces
all four required plots.

Run after step4_preprocess.py:
    python src/step5_model.py

Outputs (all saved to outputs/):
    plot1_pred_vs_actual.png   — predicted vs actual scatter
    plot2_residuals.png        — residual analysis
    plot3_epsilon_sv.png       — epsilon vs support vector count
    plot4_sv_analysis.png      — who are the support vectors?
    model_summary.txt          — printable summary for your report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR    = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
#  Using a consistent palette throughout all plots looks professional.
#  These are accessible colours (distinguishable with colour-blindness).
# ─────────────────────────────────────────────────────────────────────────────

NAVY   = "#1F3864"
RED    = "#C0392B"
GREEN  = "#1A6B3A"
AMBER  = "#E67E22"
GREY   = "#7F8C8D"
LIGHT  = "#D6E4F0"


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    """
    Loads the preprocessed CSVs produced by step4_preprocess.py.

    WHY SEPARATE LOADING FROM MODELLING:
    -------------------------------------
    Keeping data loading in its own function means:
    - Easy to swap in different data (real vs synthetic) without touching model code
    - The model functions are testable independently
    - Errors in loading are caught immediately with clear messages

    FILE SUMMARY:
    -------------
    X_train.csv     — feature matrix, training set, scaled, imputed, encoded
    X_test.csv      — feature matrix, test set (same transformations as train)
    y_train.csv     — log1p(cds_spread_bps), training targets
    y_test.csv      — log1p(cds_spread_bps), test targets
    y_train_bps.csv — original basis-point spreads, training (for reporting)
    y_test_bps.csv  — original basis-point spreads, test (for reporting)
    panel_final.csv — full panel with metadata (used for support vector analysis)

    The MultiIndex (ticker, date) is restored by specifying index_col=[0,1].
    parse_dates=True converts the date strings back to pandas Timestamps.
    """
    print("Loading preprocessed data...")

    def load_csv(filename, index_col, squeeze=False):
        path = PROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"\n  Could not find: {path}"
                f"\n  Have you run step4_preprocess.py first?"
            )
        df = pd.read_csv(path, index_col=index_col, parse_dates=True)
        # If it's a single-column CSV (a Series saved as CSV), return as Series
        if squeeze and df.shape[1] == 1:
            return df.iloc[:, 0]
        return df

    X_train    = load_csv("X_train.csv",    index_col=[0, 1])
    X_test     = load_csv("X_test.csv",     index_col=[0, 1])
    y_train    = load_csv("y_train.csv",    index_col=[0, 1], squeeze=True)
    y_test     = load_csv("y_test.csv",     index_col=[0, 1], squeeze=True)
    y_train_bps= load_csv("y_train_bps.csv",index_col=[0, 1], squeeze=True)
    y_test_bps = load_csv("y_test_bps.csv", index_col=[0, 1], squeeze=True)
    panel      = load_csv("panel_final.csv", index_col=[0, 1])

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert len(X_train) == len(y_train), "X_train and y_train row counts don't match"
    assert len(X_test)  == len(y_test),  "X_test and y_test row counts don't match"
    assert X_train.columns.tolist() == X_test.columns.tolist(), \
        "Train and test have different feature columns"

    n_train = len(X_train)
    n_test  = len(X_test)
    n_feat  = X_train.shape[1]

    # Report the dates covered by each split
    train_dates = X_train.index.get_level_values("date")
    test_dates  = X_test.index.get_level_values("date")

    print(f"  ✓ Training set:  {n_train:4d} observations "
          f"({train_dates.min().date()} → {train_dates.max().date()})")
    print(f"  ✓ Test set:      {n_test:4d} observations "
          f"({test_dates.min().date()} → {test_dates.max().date()})")
    print(f"  ✓ Features:      {n_feat} columns")
    print(f"  ✓ Features:      {list(X_train.columns)}")
    print(f"  ✓ Spread range:  {y_train_bps.min():.0f} – {y_train_bps.max():.0f} bps (train) | "
          f"{y_test_bps.min():.0f} – {y_test_bps.max():.0f} bps (test)")

    return {
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train,       # log-space (what SVR optimises)
        "y_test":      y_test,        # log-space
        "y_train_bps": y_train_bps,   # original bps (what we report)
        "y_test_bps":  y_test_bps,    # original bps
        "panel":       panel,
        "feature_names": list(X_train.columns),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparameters(X_train: pd.DataFrame,
                         y_train: pd.Series) -> tuple:
    """
    Finds the best SVR hyperparameters using 5-fold cross-validated grid search.

    THE THREE HYPERPARAMETERS:
    --------------------------

    C — Regularisation parameter
        Controls the penalty for training errors. In the SVR optimisation:
            minimise: (1/2)||w||² + C × Σ(ξᵢ + ξᵢ*)
        where ξᵢ are the slack variables (errors beyond the ε-tube).
        High C → model pays a lot to avoid errors → fits training data closely
                → risks overfitting (poor generalisation to new companies)
        Low C  → model accepts more errors → simpler decision boundary
                → risks underfitting (missing real patterns)
        Search range [0.1, 1, 10, 100, 500] covers 4 orders of magnitude.

    epsilon (ε) — The tube half-width
        The ε-tube is the core concept distinguishing SVR from regular regression.
        Predictions within ε of the true value incur ZERO loss. Only errors
        larger than ε contribute to the objective function.
        Large ε → wide tube → most points inside → few support vectors → simple model
        Small ε → narrow tube → most points outside → many support vectors → complex model
        In financial terms: ε is the "acceptable error" in bps (in log-space).
        Search range [0.01, 0.05, 0.1, 0.2, 0.5] explores tight to loose tolerances.

    gamma — RBF kernel bandwidth
        The RBF kernel: K(xᵢ, xⱼ) = exp(-gamma × ||xᵢ - xⱼ||²)
        Controls how quickly similarity drops off with distance.
        High gamma → sharp drop-off → each training point has local influence
                  → model is "jagged," potentially overfit
        Low gamma  → slow drop-off → each point has broad influence
                  → smooth predictions, potentially underfit
        'scale' is a data-driven default: gamma = 1 / (n_features × var(X))
        This is usually a good starting point, so we include it.

    CROSS-VALIDATION STRATEGY:
    --------------------------
    We use KFold(n_splits=5) — the training data is divided into 5 equal chunks.
    For each combination of (C, ε, gamma):
      - Train on 4 chunks, evaluate on the 5th
      - Repeat 5 times (each chunk takes a turn as the validation set)
      - Average the 5 validation scores → this is the CV score for that combination
    The combination with the best average CV score becomes our best_params.

    WHY NOT JUST USE THE BEST TRAINING SCORE?
    ------------------------------------------
    Training score measures how well the model fits data IT ALREADY SAW.
    A sufficiently complex model can memorise training data perfectly (R²=1.0)
    while being useless on new data. CV score estimates performance on
    UNSEEN data, which is what we actually care about.

    Note: we shuffle=True because our data has a time ordering — without
    shuffling, folds would be temporally consecutive and bleed information
    between adjacent time periods (similar to look-ahead bias).
    For Tier 2, walk-forward CV would replace this. For Tier 1, KFold is fine.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)

    param_grid = {
        "C":       [0.1, 1, 10, 100, 500],
        "epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "gamma":   ["scale", 0.01, 0.05, 0.1, 1.0],
    }

    n_combos = (len(param_grid["C"]) *
                len(param_grid["epsilon"]) *
                len(param_grid["gamma"]))

    print(f"\n  Grid: {param_grid}")
    print(f"  Combinations: {n_combos} × 5 folds = {n_combos*5} SVR fits")
    print(f"  Scoring metric: neg_mean_squared_error (in log-space)")
    print(f"  (This typically takes 1–4 minutes)\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    svr = SVR(kernel="rbf", max_iter=50000)

    # GridSearchCV tries every combination and picks the best by CV score.
    # n_jobs=-1 uses all available CPU cores in parallel (much faster).
    # return_train_score=True lets us check for overfitting (train >> CV score).
    gs = GridSearchCV(
        svr,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)

    best = gs.best_params_
    cv_rmse = np.sqrt(-gs.best_score_)   # convert neg MSE → RMSE (log-space)

    # Also get training RMSE to check for overfitting
    # If train RMSE << CV RMSE, the model is overfitting
    best_idx   = gs.best_index_
    train_rmse = np.sqrt(-gs.cv_results_["mean_train_score"][best_idx])

    print(f"\n  Best hyperparameters:")
    print(f"    C       = {best['C']}")
    print(f"    epsilon = {best['epsilon']}")
    print(f"    gamma   = {best['gamma']}")
    print(f"\n  Performance (log-space RMSE — lower is better):")
    print(f"    CV RMSE (validation):  {cv_rmse:.4f}")
    print(f"    Training RMSE:         {train_rmse:.4f}")

    gap = train_rmse / cv_rmse if cv_rmse > 0 else 1
    if gap < 0.7:
        print(f"    ⚠ Large train/CV gap ({gap:.2f}) — possible overfitting."
              f" Consider increasing C penalty or epsilon.")
    else:
        print(f"    ✓ Train/CV gap reasonable ({gap:.2f}) — no major overfitting.")

    return best, gs


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: TRAIN FINAL MODEL AND EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test,
                       y_train, y_test,
                       y_train_bps, y_test_bps,
                       best_params: dict) -> dict:
    """
    Trains the final SVR with best_params and evaluates on the held-out test set.

    IMPORTANT DISTINCTION — LOG-SPACE vs BPS:
    ------------------------------------------
    The SVR optimises in LOG-SPACE (we trained on log1p(spread)).
    Predictions come out in log-space: y_pred_log.
    We back-transform with expm1() to get basis points: y_pred_bps.

    We report metrics in BOTH spaces:
    - Log-space: what the model actually minimises (useful for debugging)
    - BPS: what matters financially (use these in your report)

    METRICS EXPLAINED:
    ------------------
    RMSE (Root Mean Squared Error):
        √(mean((y_true - y_pred)²))
        Average error in bps, but large errors are penalised more heavily.
        An RMSE of 50 bps doesn't mean you're always wrong by 50 —
        it means you might be right by 20 bps most of the time but
        occasionally very wrong (e.g. 200 bps off), and those big misses
        drag RMSE up. More sensitive to outliers than MAE.

    MAE (Mean Absolute Error):
        mean(|y_true - y_pred|)
        Simpler interpretation: on average, you're X bps wrong.
        More robust to occasional large errors than RMSE.
        If MAE << RMSE, your errors are concentrated in a few extreme cases.

    R² (Coefficient of Determination):
        1 - SS_res / SS_tot
        What fraction of the spread variation does the model explain?
        R²=1.0: perfect prediction.
        R²=0.0: model is no better than predicting the mean for everyone.
        R²<0.0: model is WORSE than just predicting the mean (a bad sign).
        For cross-sectional credit models, R²=0.7–0.9 is typical.
    """
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    print(f"\n  Parameters: C={best_params['C']}, "
          f"ε={best_params['epsilon']}, "
          f"gamma={best_params['gamma']}")

    model = SVR(kernel="rbf", max_iter=50000, **best_params)
    model.fit(X_train, y_train)
    print(f"  ✓ Model fitted")

    # ── Predictions ───────────────────────────────────────────────────────────
    # Predict in log-space, then back-transform to bps
    pred_train_log = model.predict(X_train)
    pred_test_log  = model.predict(X_test)

    pred_train_bps = np.expm1(pred_train_log)   # expm1(x) = exp(x) - 1
    pred_test_bps  = np.expm1(pred_test_log)

    # Clip negatives (shouldn't happen with log transform but be defensive)
    pred_train_bps = np.clip(pred_train_bps, 0, None)
    pred_test_bps  = np.clip(pred_test_bps,  0, None)

    # ── Metrics ───────────────────────────────────────────────────────────────
    def metrics(y_true, y_pred, label):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return {"label": label, "rmse": rmse, "mae": mae, "r2": r2}

    m_train = metrics(y_train_bps, pred_train_bps, "Train")
    m_test  = metrics(y_test_bps,  pred_test_bps,  "Test")

    print(f"\n  {'Metric':<8}  {'Train':>10}  {'Test':>10}")
    print(f"  {'─'*32}")
    print(f"  {'RMSE':<8}  {m_train['rmse']:>9.1f}  {m_test['rmse']:>9.1f}  bps")
    print(f"  {'MAE':<8}  {m_train['mae']:>9.1f}  {m_test['mae']:>9.1f}  bps")
    print(f"  {'R²':<8}  {m_train['r2']:>9.3f}  {m_test['r2']:>9.3f}")

    # ── Support vector analysis ───────────────────────────────────────────────
    # model.n_support_ is an array with one entry per class.
    # For regression (SVR), there is one "class," so we take index [0].
    n_sv  = model.n_support_[0]
    pct_sv = n_sv / len(X_train) * 100

    print(f"\n  Support vectors: {n_sv} / {len(X_train)} training points "
          f"({pct_sv:.1f}%)")
    print(f"  → {100 - pct_sv:.1f}% of training points sit INSIDE the ε-tube")
    print(f"    and contribute nothing to the prediction function.")
    print(f"  → Only the {pct_sv:.1f}% that are support vectors determine")
    print(f"    the shape of the regression surface.")

    # Interpretation guidance
    if pct_sv > 80:
        print(f"\n  ⚠ Very high SV count ({pct_sv:.0f}%) — model may be overfit.")
        print(f"    Try: larger ε, smaller C, or check for noisy features.")
    elif pct_sv < 20:
        print(f"\n  ⚠ Very low SV count ({pct_sv:.0f}%) — model may be underfit.")
        print(f"    Try: smaller ε or larger C.")
    else:
        print(f"\n  ✓ SV count in reasonable range.")

    return {
        "model":           model,
        "pred_train_bps":  pred_train_bps,
        "pred_test_bps":   pred_test_bps,
        "pred_train_log":  pred_train_log,
        "pred_test_log":   pred_test_log,
        "m_train":         m_train,
        "m_test":          m_test,
        "n_sv":            n_sv,
        "pct_sv":          pct_sv,
        "sv_indices":      model.support_,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4: PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot1_predicted_vs_actual(y_true: pd.Series,
                              y_pred: np.ndarray,
                              m_test: dict) -> None:
    """
    PLOT 1: Predicted vs Actual CDS Spreads (scatter with 45° line)

    HOW TO READ THIS PLOT:
    ----------------------
    - Each point is one (company, quarter) observation in the test set.
    - The diagonal dashed line is the "perfect prediction" line: y_pred = y_true.
    - Points ABOVE the line: model underpredicted (actual was higher than predicted).
    - Points BELOW the line: model overpredicted.
    - Tight cluster around the line = good model.
    - Systematic deviation (e.g. all HY points above the line) = model bias
      for that type of company.

    We colour-code by rating bucket so you can spot whether the model
    performs differently across the credit quality spectrum.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Colour-code by rating (stored in y_true's index or panel — we approximate
    # by spread level as a proxy for rating here)
    # Bin into IG (spread < 200 bps) and HY (spread >= 200 bps)
    y_true_arr = np.array(y_true)
    is_hy = y_true_arr >= 200

    ax.scatter(y_true_arr[~is_hy], y_pred[~is_hy],
               alpha=0.7, color=NAVY, edgecolors="white",
               linewidth=0.5, s=55, label="Investment Grade (spread < 200 bps)")
    ax.scatter(y_true_arr[is_hy], y_pred[is_hy],
               alpha=0.7, color=RED, edgecolors="white",
               linewidth=0.5, s=55, label="High Yield (spread ≥ 200 bps)")

    # 45-degree line
    lo = min(y_true_arr.min(), y_pred.min()) * 0.9
    hi = max(y_true_arr.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=AMBER, linewidth=2.5,
            linestyle="--", label="Perfect prediction (45°)", zorder=3)

    # Metrics annotation box
    box_text = (f"R²   = {m_test['r2']:.3f}\n"
                f"RMSE = {m_test['rmse']:.1f} bps\n"
                f"MAE  = {m_test['mae']:.1f} bps")
    ax.text(0.04, 0.95, box_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=GREY, alpha=0.9))

    ax.set_xlabel("Actual CDS Spread (bps)", fontsize=12)
    ax.set_ylabel("Predicted CDS Spread (bps)", fontsize=12)
    ax.set_title("Predicted vs Actual CDS Spreads — Test Set",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = OUTPUT_DIR / "plot1_pred_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot2_residuals(y_true: pd.Series, y_pred: np.ndarray) -> None:
    """
    PLOT 2: Residual Analysis (two-panel)

    WHAT IS A RESIDUAL?
    -------------------
    residual = actual − predicted
    Positive residual: model UNDERPREDICTED (actual spread was higher than predicted)
    Negative residual: model OVERPREDICTED (predicted spread was higher than actual)

    LEFT PANEL — Residuals vs Predicted Values:
    What to look for:
      1. Random scatter around zero = good. The model has no systematic bias.
      2. Fan shape (residuals grow as predicted spread grows) = heteroscedasticity.
         This means the model is less accurate for high-spread (HY) companies.
         Common in credit models — HY spreads are more volatile and harder to predict.
      3. Curved pattern = the model misses a nonlinear relationship.
         Could mean a feature transformation would help (e.g. log(leverage)).

    RIGHT PANEL — Distribution of Residuals:
    Should be approximately bell-shaped and centred near zero.
    A skewed distribution means the model has directional bias
    (consistently over- or under-predicting).
    """
    residuals = np.array(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: residuals vs predicted ─────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.65, color=NAVY,
               edgecolors="white", linewidth=0.4, s=50)
    ax.axhline(0, color=RED, linewidth=2, linestyle="--", label="Zero residual")

    # Add a LOESS-style smoothed line to show any trend in residuals
    # We use a rolling median as a simple alternative (no extra libraries needed)
    sort_idx = np.argsort(y_pred)
    y_pred_sorted   = y_pred[sort_idx]
    residuals_sorted = residuals[sort_idx]
    window = max(5, len(residuals) // 10)
    smoothed = pd.Series(residuals_sorted).rolling(window, center=True,
                                                    min_periods=1).mean()
    ax.plot(y_pred_sorted, smoothed, color=AMBER, linewidth=2,
            linestyle="-", label="Trend (rolling mean)", alpha=0.8)

    mean_res = residuals.mean()
    std_res  = residuals.std()
    ax.text(0.04, 0.96,
            f"Mean residual: {mean_res:+.1f} bps\nStd: {std_res:.1f} bps",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GREY, alpha=0.9))

    ax.set_xlabel("Predicted CDS Spread (bps)", fontsize=12)
    ax.set_ylabel("Residual: Actual − Predicted (bps)", fontsize=12)
    ax.set_title("Residuals vs Predicted Values", fontsize=12,
                 fontweight="bold", color=NAVY)
    ax.legend(fontsize=9)

    # ── Right: distribution of residuals ─────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(residuals, bins=25, color=NAVY, alpha=0.75, edgecolor="white",
             linewidth=0.7, label="Residuals")
    ax2.axvline(0,          color=RED,   linewidth=2.5, linestyle="--",
                label="Zero")
    ax2.axvline(mean_res,   color=AMBER, linewidth=2,   linestyle="-",
                label=f"Mean = {mean_res:+.1f} bps")
    ax2.axvline(mean_res + std_res, color=GREY, linewidth=1.5, linestyle=":",
                label=f"±1 SD ({std_res:.1f} bps)")
    ax2.axvline(mean_res - std_res, color=GREY, linewidth=1.5, linestyle=":")

    ax2.set_xlabel("Residual (bps)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Residuals", fontsize=12,
                  fontweight="bold", color=NAVY)
    ax2.legend(fontsize=9)

    fig.suptitle("Residual Analysis — Test Set",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / "plot2_residuals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot3_epsilon_vs_sv(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        best_params: dict) -> None:
    """
    PLOT 3: Epsilon vs Number of Support Vectors

    THIS IS THE CORE THEORETICAL PLOT FOR TIER 1.
    It makes the ε-tube concept tangible with your actual data.

    THE THEORY (from lectures):
    ---------------------------
    SVR defines an ε-insensitive loss function:
        L(y, f(x)) = max(0, |y − f(x)| − ε)
    Errors within ε contribute zero to the objective.
    Only points OUTSIDE the tube (errors > ε) become support vectors.

    WHAT THIS PLOT SHOWS:
    ---------------------
    As ε increases, the tube gets wider, more training points fall inside,
    fewer become support vectors, and the model gets simpler (lower complexity).
    As ε decreases toward 0, almost every point falls outside the tube,
    almost every point becomes a support vector, and the model essentially
    memorises the training data.

    HOW TO DESCRIBE THIS IN YOUR REPORT:
    -------------------------------------
    "As ε increases from 0.005 to 1.0, the number of support vectors decreases
    monotonically from X to Y, representing a reduction from X% to Y% of the
    training data. This demonstrates the ε-tube's role as a complexity control
    mechanism: wider tolerance → fewer support vectors → simpler model.
    The optimal ε=[value] balances predictive accuracy with model simplicity."

    We keep C and gamma fixed at their best values so the only thing changing
    is the tube width, making the effect of ε isolated and interpretable.
    """
    print("\n  Computing support vector counts across ε values...")
    print(f"  (Fitting SVR {12} times — takes ~1–2 minutes)\n")

    epsilons  = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    sv_counts = []
    sv_pcts   = []

    n_train = len(X_train)

    for eps in epsilons:
        m = SVR(kernel="rbf",
                C=best_params["C"],
                gamma=best_params["gamma"],
                epsilon=eps,
                max_iter=50000)
        m.fit(X_train, y_train)
        n = m.n_support_[0]
        sv_counts.append(n)
        sv_pcts.append(n / n_train * 100)
        print(f"    ε = {eps:5.3f}  →  {n:4d} support vectors ({n/n_train*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    best_eps = best_params["epsilon"]
    # Find the index of the epsilon closest to our best (for annotation)
    best_idx = min(range(len(epsilons)), key=lambda i: abs(epsilons[i] - best_eps))

    # ── Left: absolute count ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epsilons, sv_counts, "o-", color=NAVY, linewidth=2.5,
            markersize=8, markerfacecolor="white",
            markeredgecolor=NAVY, markeredgewidth=2)
    ax.axvline(best_eps, color=RED, linestyle="--", linewidth=2,
               label=f"Optimal ε = {best_eps}", alpha=0.8)
    ax.scatter([best_eps], [sv_counts[best_idx]],
               color=RED, s=140, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (ε) — log scale", fontsize=12)
    ax.set_ylabel("Number of Support Vectors", fontsize=12)
    ax.set_title("Support Vectors vs ε", fontsize=12,
                 fontweight="bold", color=NAVY)
    ax.legend(fontsize=10)

    # Annotate the two extremes
    ax.annotate(f"{sv_counts[0]} SVs\n({sv_pcts[0]:.0f}% of train)",
                xy=(epsilons[0], sv_counts[0]),
                xytext=(epsilons[0]*1.5, sv_counts[0]*0.92),
                fontsize=8, color=GREY)
    ax.annotate(f"{sv_counts[-1]} SVs\n({sv_pcts[-1]:.0f}% of train)",
                xy=(epsilons[-1], sv_counts[-1]),
                xytext=(epsilons[-1]*0.4, sv_counts[-1]*1.05),
                fontsize=8, color=GREY)

    # ── Right: percentage ─────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(epsilons, sv_pcts, alpha=0.15, color=NAVY)
    ax2.plot(epsilons, sv_pcts, "o-", color=NAVY, linewidth=2.5,
             markersize=8, markerfacecolor="white",
             markeredgecolor=NAVY, markeredgewidth=2)
    ax2.axhline(100, color=GREY, linestyle=":", linewidth=1.5,
                label="100% (all training points)")
    ax2.axvline(best_eps, color=RED, linestyle="--", linewidth=2,
                label=f"Optimal ε = {best_eps}", alpha=0.8)
    ax2.scatter([best_eps], [sv_pcts[best_idx]],
                color=RED, s=140, zorder=5,
                label=f"{sv_pcts[best_idx]:.1f}% SVs at optimal ε")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("Epsilon (ε) — log scale", fontsize=12)
    ax2.set_ylabel("Support Vectors as % of Training Data", fontsize=12)
    ax2.set_title("Model Complexity vs ε", fontsize=12,
                  fontweight="bold", color=NAVY)
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Effect of ε-tube Width on Model Complexity\n"
        "Wider tube → fewer support vectors → simpler model (lower variance, higher bias)",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    path = OUTPUT_DIR / "plot3_epsilon_sv.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {path}")


def plot4_sv_analysis(model, X_train: pd.DataFrame,
                      y_train_bps: pd.Series,
                      panel: pd.DataFrame) -> None:
    """
    PLOT 4: Who Are the Support Vectors?

    This plot answers the question the spec explicitly asks:
    "Do the support vectors correspond to particular types of companies?"

    WHAT ARE SUPPORT VECTORS IN CREDIT TERMS?
    ------------------------------------------
    Support vectors are the training observations that the model found
    HARDEST to predict within the ε-tolerance. They are the "edge cases"
    that define the boundary of the prediction surface.

    In credit risk, you'd expect support vectors to be concentrated among:
    - Companies with volatile spreads (harder to predict)
    - Companies at rating boundaries (e.g. BBB vs BB — close to IG/HY transition)
    - Stressed periods (quarters with unusual macro conditions)
    - Sectors with idiosyncratic risk (airlines, energy — COVID hit these asymmetrically)

    If support vectors are uniformly distributed across the data, it means
    the model struggles equally everywhere — no systematic blind spots.
    If they cluster in HY, it means HY spreads are harder to model
    (probably true — HY is more volatile and more company-specific).

    LEFT PANEL — Support vector rate by rating:
    Bar chart: what % of each rating's training observations are support vectors?

    RIGHT PANEL — Spread distributions:
    Histograms comparing spread distributions for SVs vs non-SVs.
    If SV spreads are more extreme, that confirms they're the "hard" cases.
    """
    sv_idx = model.support_   # integer indices into X_train

    # Build a boolean array: True where observation is a support vector
    is_sv = np.zeros(len(X_train), dtype=bool)
    is_sv[sv_idx] = True

    # Try to attach rating metadata from the panel
    train_index = X_train.index
    analysis = pd.DataFrame({
        "is_sv":    is_sv,
        "spread_bps": np.array(y_train_bps),
    }, index=train_index)

    # Attach rating from panel if available
    if "rating" in panel.columns:
        analysis = analysis.join(panel[["rating"]], how="left")
    elif "rating" in X_train.columns:
        analysis["rating"] = X_train["rating"].values
    else:
        analysis["rating"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: SV rate by rating bucket ───────────────────────────────────────
    ax = axes[0]

    rating_labels = {1:"AAA", 2:"AA", 3:"A", 4:"BBB", 5:"BB", 6:"B", 7:"CCC"}
    overall_pct = is_sv.mean() * 100

    if analysis["rating"].notna().any():
        # Round rating to nearest integer (it's stored as float)
        analysis["rating_int"] = analysis["rating"].round().astype("Int64")
        sv_by_rating = (analysis.groupby("rating_int")["is_sv"]
                        .agg(["sum", "count"]))
        sv_by_rating["pct"] = sv_by_rating["sum"] / sv_by_rating["count"] * 100

        colors = [RED if row["pct"] > overall_pct * 1.2 else NAVY
                  for _, row in sv_by_rating.iterrows()]
        bars = ax.bar(sv_by_rating.index, sv_by_rating["pct"],
                      color=colors, alpha=0.8, edgecolor="white",
                      linewidth=0.8)
        ax.axhline(overall_pct, color=AMBER, linestyle="--", linewidth=2,
                   label=f"Overall avg: {overall_pct:.1f}%")

        # Add count labels on bars
        for bar, (idx, row) in zip(bars, sv_by_rating.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"n={int(row['count'])}", ha="center",
                    fontsize=8, color=GREY)

        ax.set_xticks(list(sv_by_rating.index))
        ax.set_xticklabels([rating_labels.get(int(r), str(r))
                             for r in sv_by_rating.index], fontsize=10)
        ax.set_xlabel("Credit Rating", fontsize=12)
        ax.set_ylabel("% of Observations that are Support Vectors", fontsize=12)
        ax.set_title("Support Vector Rate by Credit Rating",
                     fontsize=12, fontweight="bold", color=NAVY)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 110)

        # Highlight IG/HY boundary
        ax.axvline(4.5, color=GREY, linestyle=":", linewidth=1.5, alpha=0.6)
        ax.text(4.6, 105, "HY →", fontsize=8, color=GREY, va="top")
        ax.text(4.3, 105, "← IG", fontsize=8, color=GREY, va="top", ha="right")
    else:
        ax.text(0.5, 0.5, "Rating data not available",
                transform=ax.transAxes, ha="center")

    # ── Right: spread distribution — SVs vs non-SVs ───────────────────────────
    ax2 = axes[1]

    sv_spreads    = analysis.loc[analysis["is_sv"],  "spread_bps"]
    nonsv_spreads = analysis.loc[~analysis["is_sv"], "spread_bps"]

    bins = np.linspace(analysis["spread_bps"].min(),
                       analysis["spread_bps"].max(), 30)

    ax2.hist(nonsv_spreads, bins=bins, alpha=0.65, color=NAVY,
             label=f"Non-support vectors (n={len(nonsv_spreads)})",
             edgecolor="white", linewidth=0.7)
    ax2.hist(sv_spreads, bins=bins, alpha=0.65, color=RED,
             label=f"Support vectors (n={len(sv_spreads)})",
             edgecolor="white", linewidth=0.7)

    ax2.axvline(sv_spreads.median(), color=RED, linewidth=2, linestyle="-",
                label=f"SV median: {sv_spreads.median():.0f} bps")
    ax2.axvline(nonsv_spreads.median(), color=NAVY, linewidth=2, linestyle="-",
                label=f"Non-SV median: {nonsv_spreads.median():.0f} bps")

    ax2.set_xlabel("CDS Spread (bps)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Spread Distribution: SVs vs Non-SVs",
                  fontsize=12, fontweight="bold", color=NAVY)
    ax2.legend(fontsize=9)

    fig.suptitle("Who Are the Support Vectors?",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / "plot4_sv_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: WRITTEN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(best_params: dict, results: dict,
                  data: dict, gs) -> None:
    """
    Writes a structured text summary to outputs/model_summary.txt.

    This is designed to be directly useful for writing your Tier 1 report.
    Every number you need to quote is in here, with context.
    """
    lines = []
    a = lines.append   # shorthand

    a("=" * 65)
    a("TIER 1 SVR MODEL SUMMARY")
    a("CDS Spread Modelling — Support Vector Regression")
    a("=" * 65)

    train_dates = data["X_train"].index.get_level_values("date")
    test_dates  = data["X_test"].index.get_level_values("date")

    a(f"\nDATA")
    a(f"  Training period : {train_dates.min().date()} → {train_dates.max().date()}")
    a(f"  Test period     : {test_dates.min().date()}  → {test_dates.max().date()}")
    a(f"  Train obs (N)   : {len(data['X_train'])}")
    a(f"  Test obs        : {len(data['X_test'])}")
    a(f"  Features (D)    : {len(data['feature_names'])}")
    a(f"  Feature list    : {data['feature_names']}")
    a(f"  Train spread    : {data['y_train_bps'].min():.0f} – {data['y_train_bps'].max():.0f} bps")
    a(f"  Test spread     : {data['y_test_bps'].min():.0f} – {data['y_test_bps'].max():.0f} bps")

    a(f"\nMODEL SPECIFICATION")
    a(f"  Algorithm       : Support Vector Regression (SVR)")
    a(f"  Kernel          : RBF (Radial Basis Function)")
    a(f"  Kernel formula  : K(xᵢ,xⱼ) = exp(-γ||xᵢ-xⱼ||²)")
    a(f"  Implementation  : scikit-learn SVR")

    a(f"\nHYPERPARAMETER TUNING")
    a(f"  Method          : GridSearchCV, 5-fold KFold")
    a(f"  Scoring         : neg_mean_squared_error (log-space)")
    a(f"  Grid searched   : C ∈ [0.1,1,10,100,500]")
    a(f"                    ε ∈ [0.01,0.05,0.1,0.2,0.5]")
    a(f"                    γ ∈ ['scale',0.01,0.05,0.1,1.0]")
    n_combos = 5 * 5 * 5
    a(f"  Total fits      : {n_combos} combinations × 5 folds = {n_combos*5}")

    a(f"\nBEST HYPERPARAMETERS")
    a(f"  C               = {best_params['C']}")
    a(f"  epsilon (ε)     = {best_params['epsilon']}")
    a(f"  gamma (γ)       = {best_params['gamma']}")
    cv_rmse = np.sqrt(-gs.best_score_)
    a(f"  CV RMSE         = {cv_rmse:.4f} (log-space)")

    m = results["m_test"]
    mt = results["m_train"]
    a(f"\nPERFORMANCE METRICS (in basis points — back-transformed)")
    a(f"  {'Metric':<8}  {'Train':>10}  {'Test':>10}")
    a(f"  {'─'*32}")
    a(f"  {'RMSE':<8}  {mt['rmse']:>9.1f}  {m['rmse']:>9.1f}  bps")
    a(f"  {'MAE':<8}  {mt['mae']:>9.1f}  {m['mae']:>9.1f}  bps")
    a(f"  {'R²':<8}  {mt['r2']:>9.3f}  {m['r2']:>9.3f}")

    n_sv  = results["n_sv"]
    pct   = results["pct_sv"]
    n_tr  = len(data["X_train"])
    a(f"\nSUPPORT VECTOR ANALYSIS")
    a(f"  Total training points  : {n_tr}")
    a(f"  Support vectors        : {n_sv} ({pct:.1f}%)")
    a(f"  Inside ε-tube          : {n_tr - n_sv} ({100-pct:.1f}%)")
    a(f"  Interpretation         : {100-pct:.1f}% of training points sit within")
    a(f"                           the ε={best_params['epsilon']} tube and contribute")
    a(f"                           zero to the prediction function.")

    a(f"\nOUTPUT FILES")
    a(f"  outputs/plot1_pred_vs_actual.png — predicted vs actual scatter")
    a(f"  outputs/plot2_residuals.png      — residual analysis (2-panel)")
    a(f"  outputs/plot3_epsilon_sv.png     — ε vs support vector count")
    a(f"  outputs/plot4_sv_analysis.png    — support vector breakdown")

    summary = "\n".join(lines)
    print("\n" + summary)

    path = OUTPUT_DIR / "model_summary.txt"
    with open(path, "w") as f:
        f.write(summary)
    print(f"\n  Summary saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    print("\n" + "="*60)
    print("  STEP 5 — TIER 1 SVR MODEL")
    print("="*60 + "\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    data = load_data()

    X_train    = data["X_train"]
    X_test     = data["X_test"]
    y_train    = data["y_train"]
    y_test     = data["y_test"]
    y_train_bps= data["y_train_bps"]
    y_test_bps = data["y_test_bps"]
    panel      = data["panel"]

    # ── 2. Tune ───────────────────────────────────────────────────────────────
    best_params, gs = tune_hyperparameters(X_train, y_train)

    # ── 3. Train and evaluate ─────────────────────────────────────────────────
    results = train_and_evaluate(
        X_train, X_test,
        y_train, y_test,
        y_train_bps, y_test_bps,
        best_params,
    )

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    print("\n  Plot 1: Predicted vs Actual")
    plot1_predicted_vs_actual(y_test_bps, results["pred_test_bps"],
                              results["m_test"])

    print("  Plot 2: Residual Analysis")
    plot2_residuals(y_test_bps, results["pred_test_bps"])

    print("  Plot 3: Epsilon vs Support Vectors")
    plot3_epsilon_vs_sv(X_train, y_train, best_params)

    print("  Plot 4: Support Vector Analysis")
    plot4_sv_analysis(results["model"], X_train, y_train_bps, panel)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    write_summary(best_params, results, data, gs)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR.resolve()}")