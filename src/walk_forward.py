"""
step6b_walkforward.py
=====================
Tier 2: Walk-Forward Validation

Replaces the single temporal split with a rolling window evaluation
that simulates how the model would actually be deployed in practice.

Run after step6a_altman_zscore.py:
    python src/step6b_walkforward.py

What this covers:
    - Why random splits are wrong for financial time-series data
    - What walk-forward validation is and how it's implemented
    - How to compare performance across different market regimes
    - How to report results honestly when they differ from naive splits

Output:
    outputs/walkforward_results.png
    outputs/walkforward_summary.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR    = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

NAVY  = "#1F3864"
RED   = "#C0392B"
GREEN = "#1A6B3A"
AMBER = "#E67E22"
GREY  = "#7F8C8D"
LIGHT = "#AED6F1"

# ─────────────────────────────────────────────────────────────────────────────
#  WALK-FORWARD VALIDATION: THE CONCEPT
# ─────────────────────────────────────────────────────────────────────────────
#
# THE PROBLEM WITH A SINGLE TEMPORAL SPLIT:
# -----------------------------------------
# Even when we split on a date (train before 2022, test after), we still
# have only ONE estimate of model performance. That estimate might be
# unusually good or bad depending on which time periods happen to fall
# in each half.
#
# Example: if our training period (2018-2022) covers stable markets and
# our test period (2022-2024) covers rising rates, we've tested generalisation
# across one specific regime change — but we don't know how the model would
# perform if the test period were 2020-2021 (COVID stress) or 2018-2019
# (pre-COVID stability).
#
# THE SOLUTION — EXPANDING WINDOW WALK-FORWARD:
# ---------------------------------------------
# We simulate the model being deployed in real time, quarter by quarter:
#
#   Step 1: Train on Q1 2018 → Q4 2019, predict Q1 2020
#   Step 2: Train on Q1 2018 → Q1 2020, predict Q2 2020
#   Step 3: Train on Q1 2018 → Q2 2020, predict Q3 2020
#   ...and so on until the end of the data.
#
# This is called an "expanding window" because the training set grows.
# An alternative is a "rolling window" (fixed size — drop old quarters
# as you add new ones). Expanding is more common in credit modelling
# because older data is still informative (default cycles are long).
#
# Each step produces ONE out-of-sample prediction for that quarter.
# Aggregating all these predictions gives a more robust performance estimate.
#
# KEY INSIGHT: The walk-forward RMSE is typically higher than the single-split
# RMSE because it tests harder scenarios (e.g. predicting COVID quarters
# using only pre-COVID training data). This is the honest number.

PARAM_GRID = {
    "C":       [1, 10, 100],
    "epsilon": [0.01, 0.05, 0.1],
    "gamma":   ["scale", 0.01, 0.1],
}
# Smaller grid than step5 because we're fitting it many times (once per fold)


def run_single_fold(X_train_fold: pd.DataFrame,
                    X_test_fold:  pd.DataFrame,
                    y_train_bps:  pd.Series,
                    y_test_bps:   pd.Series) -> tuple:
    """
    Runs the complete SVR pipeline for one walk-forward fold.
    Returns (predictions_bps, best_params, n_sv).

    Each fold has its own imputer, scaler, and tuning — no information
    from future folds leaks backward.
    """

        # ── Fix: encode any remaining string columns before imputing ─────────────
    # sector and region may still be raw strings if panel was saved pre-encoding.
    # We one-hot encode here, then align columns so train and test match exactly.
    string_cols = X_train_fold.select_dtypes(include="object").columns.tolist()
    if string_cols:
        X_train_fold = pd.get_dummies(X_train_fold, columns=string_cols, drop_first=True)
        X_test_fold  = pd.get_dummies(X_test_fold,  columns=string_cols, drop_first=True)
        # Align: test may be missing a dummy column if a category only appears in train
        X_train_fold, X_test_fold = X_train_fold.align(
            X_test_fold, join="left", axis=1, fill_value=0
        )

        
    # Impute
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train_fold)
    Xtr = pd.DataFrame(imp.transform(X_train_fold),
                       columns=X_train_fold.columns, index=X_train_fold.index)
    Xte = pd.DataFrame(imp.transform(X_test_fold),
                       columns=X_test_fold.columns, index=X_test_fold.index)

    # Scale
    dummy_cols = [c for c in Xtr.columns if Xtr[c].nunique() <= 2]
    scale_cols = [c for c in Xtr.columns if c not in dummy_cols]
    sc = StandardScaler()
    sc.fit(Xtr[scale_cols])
    Xtr[scale_cols] = sc.transform(Xtr[scale_cols])
    Xte[scale_cols] = sc.transform(Xte[scale_cols])

    # Log-transform target
    ytr_log = np.log1p(y_train_bps)

    # Tune (3-fold CV within training fold — smaller because training set is smaller)
    n_splits = min(3, len(Xtr) // 10)  # adaptive: at least 10 obs per CV fold
    n_splits = max(n_splits, 2)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    gs = GridSearchCV(SVR(kernel="rbf", max_iter=50000), PARAM_GRID,
                      cv=cv, scoring="neg_mean_squared_error",
                      n_jobs=-1, verbose=0)
    gs.fit(Xtr, ytr_log)

    # Train final
    model = SVR(kernel="rbf", max_iter=50000, **gs.best_params_)
    model.fit(Xtr, ytr_log)

    pred_bps = np.clip(np.expm1(model.predict(Xte)), 0, None)
    return pred_bps, gs.best_params_, model.n_support_[0]


def walk_forward_validation(panel: pd.DataFrame,
                            feature_cols: list,
                            target_col: str = "cds_spread_bps",
                            min_train_quarters: int = 8) -> pd.DataFrame:
    """
    Runs expanding-window walk-forward validation.

    Parameters:
    -----------
    panel               : full panel DataFrame with (ticker, date) MultiIndex
    feature_cols        : which columns to use as features
    target_col          : the target column name
    min_train_quarters  : minimum quarters needed before we start predicting
                          (8 quarters = 2 years minimum training history)

    Returns:
    --------
    results_df : DataFrame with one row per (ticker, quarter) in the test region,
                 containing actual spread, predicted spread, and fold metadata.
    """
    all_quarters = (panel.index.get_level_values("date")
                    .unique().sort_values())

    print(f"  Total quarters: {len(all_quarters)}")
    print(f"  Minimum training quarters: {min_train_quarters}")
    print(f"  Walk-forward will predict: {len(all_quarters) - min_train_quarters} quarters")
    print()

    all_results = []
    fold_summaries = []

    for i, test_quarter in enumerate(all_quarters[min_train_quarters:], 1):
        # Training data: everything strictly before this quarter
        train_dates = all_quarters[all_quarters < test_quarter]
        train_mask  = panel.index.get_level_values("date").isin(train_dates)
        test_mask   = panel.index.get_level_values("date") == test_quarter

        train_data = panel[train_mask]
        test_data  = panel[test_mask]

        if len(test_data) == 0:
            continue

        X_train_f = train_data[feature_cols]
        X_test_f  = test_data[feature_cols]
        y_train_f = train_data[target_col]
        y_test_f  = test_data[target_col]

        n_train_q = len(train_dates)
        n_train_obs = len(train_data)

        print(f"  Fold {i:2d} | Test: {test_quarter.date()} | "
              f"Train: {n_train_q} quarters ({n_train_obs} obs) | "
              f"Test obs: {len(test_data)}", end="")

        try:
            pred_bps, best_params, n_sv = run_single_fold(
                X_train_f, X_test_f, y_train_f, y_test_f
            )

            fold_rmse = np.sqrt(mean_squared_error(y_test_f, pred_bps))
            fold_r2   = r2_score(y_test_f, pred_bps)
            print(f" → RMSE={fold_rmse:.1f} bps, R²={fold_r2:.3f}")

            # Store per-observation results
            for j, (idx, true_val) in enumerate(y_test_f.items()):
                all_results.append({
                    "ticker":    idx[0],
                    "date":      idx[1],
                    "actual":    true_val,
                    "predicted": pred_bps[j],
                    "residual":  true_val - pred_bps[j],
                    "fold":      i,
                    "n_train_q": n_train_q,
                })

            fold_summaries.append({
                "fold":          i,
                "test_quarter":  test_quarter,
                "n_train_q":     n_train_q,
                "n_test_obs":    len(test_data),
                "rmse":          fold_rmse,
                "r2":            fold_r2,
                "best_C":        best_params["C"],
                "best_epsilon":  best_params["epsilon"],
                "best_gamma":    best_params["gamma"],
                "n_sv":          n_sv,
            })

        except Exception as e:
            print(f" → ERROR: {e}")

    results_df = pd.DataFrame(all_results)
    folds_df   = pd.DataFrame(fold_summaries)

    return results_df, folds_df


def plot_walkforward(results_df: pd.DataFrame,
                     folds_df: pd.DataFrame,
                     single_split_rmse: float,
                     single_split_r2: float) -> None:
    """
    Four-panel walk-forward results plot.
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: RMSE per quarter with regime annotations ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    quarters  = folds_df["test_quarter"]
    fold_rmse = folds_df["rmse"]
    colors    = []
    for q in quarters:
        if q <= pd.Timestamp("2020-06-30"):
            colors.append(AMBER)   # COVID stress
        elif q <= pd.Timestamp("2021-12-31"):
            colors.append(GREEN)   # Recovery
        else:
            colors.append(NAVY)    # Rate hiking cycle

    bars = ax1.bar(range(len(quarters)), fold_rmse, color=colors,
                   alpha=0.8, edgecolor="white", linewidth=0.7)
    ax1.axhline(single_split_rmse, color=RED, linestyle="--", linewidth=2,
                label=f"Single-split RMSE: {single_split_rmse:.1f} bps")
    wf_mean = fold_rmse.mean()
    ax1.axhline(wf_mean, color=GREY, linestyle=":", linewidth=1.5,
                label=f"Walk-forward mean: {wf_mean:.1f} bps")

    ax1.set_xticks(range(len(quarters)))
    ax1.set_xticklabels([q.strftime("%Y-Q%q") if hasattr(q, 'strftime') else str(q)[:7]
                          for q in quarters], rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("RMSE (bps)", fontsize=11)
    ax1.set_title("RMSE per Quarter (Walk-Forward)",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax1.legend(fontsize=8)

    # Regime legend
    patches = [
        mpatches.Patch(color=AMBER, label="COVID stress (2020)"),
        mpatches.Patch(color=GREEN, label="Recovery (2021)"),
        mpatches.Patch(color=NAVY,  label="Rate hike cycle (2022+)"),
    ]
    ax1.legend(handles=patches + [
        plt.Line2D([0],[0], color=RED, linestyle="--", label=f"Single-split: {single_split_rmse:.1f} bps"),
        plt.Line2D([0],[0], color=GREY, linestyle=":", label=f"WF mean: {wf_mean:.1f} bps"),
    ], fontsize=7, loc="upper left")

    # ── Panel 2: R² per quarter ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(range(len(quarters)), folds_df["r2"], "o-",
             color=NAVY, linewidth=2, markersize=6,
             markerfacecolor="white", markeredgewidth=2)
    ax2.axhline(single_split_r2, color=RED, linestyle="--", linewidth=2,
                label=f"Single-split R²: {single_split_r2:.3f}")
    ax2.axhline(folds_df["r2"].mean(), color=GREY, linestyle=":", linewidth=1.5,
                label=f"WF mean R²: {folds_df['r2'].mean():.3f}")
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.3)
    ax2.fill_between(range(len(quarters)), folds_df["r2"], 0,
                     where=folds_df["r2"] > 0, alpha=0.1, color=GREEN)
    ax2.fill_between(range(len(quarters)), folds_df["r2"], 0,
                     where=folds_df["r2"] < 0, alpha=0.1, color=RED)

    ax2.set_xticks(range(len(quarters)))
    ax2.set_xticklabels([q.strftime("%Y-Q%q") if hasattr(q,'strftime') else str(q)[:7]
                          for q in quarters], rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("R²", fontsize=11)
    ax2.set_title("R² per Quarter (Walk-Forward)",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax2.legend(fontsize=8)

    # ── Panel 3: Aggregated predicted vs actual ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.scatter(results_df["actual"], results_df["predicted"],
                alpha=0.5, s=30, color=NAVY, edgecolors="white", linewidth=0.3)
    lo = min(results_df["actual"].min(), results_df["predicted"].min()) * 0.95
    hi = max(results_df["actual"].max(), results_df["predicted"].max()) * 1.05
    ax3.plot([lo, hi], [lo, hi], "--", color=AMBER, linewidth=2,
             label="Perfect prediction")

    overall_rmse = np.sqrt(mean_squared_error(results_df["actual"],
                                               results_df["predicted"]))
    overall_r2   = r2_score(results_df["actual"], results_df["predicted"])
    ax3.text(0.04, 0.95,
             f"All folds combined:\nR² = {overall_r2:.3f}\nRMSE = {overall_rmse:.1f} bps",
             transform=ax3.transAxes, fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=GREY, alpha=0.9))

    ax3.set_xlabel("Actual CDS Spread (bps)", fontsize=11)
    ax3.set_ylabel("Predicted CDS Spread (bps)", fontsize=11)
    ax3.set_title("Walk-Forward: Predicted vs Actual (All Folds)",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax3.legend(fontsize=8)

    # ── Panel 4: Residuals over time ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])

    # Aggregate residuals by quarter
    q_residuals = (results_df.groupby("date")["residual"]
                   .agg(["mean", "std"]).reset_index())

    ax4.fill_between(range(len(q_residuals)),
                     q_residuals["mean"] - q_residuals["std"],
                     q_residuals["mean"] + q_residuals["std"],
                     alpha=0.2, color=NAVY, label="±1 SD")
    ax4.plot(range(len(q_residuals)), q_residuals["mean"],
             "o-", color=NAVY, linewidth=2, markersize=5,
             markerfacecolor="white", markeredgewidth=1.5,
             label="Mean residual per quarter")
    ax4.axhline(0, color=RED, linestyle="--", linewidth=1.5, label="Zero")

    # Shade COVID period
    covid_start = q_residuals[q_residuals["date"] >= "2020-01-01"].index.min()
    covid_end   = q_residuals[q_residuals["date"] <= "2020-12-31"].index.max()
    if pd.notna(covid_start) and pd.notna(covid_end):
        ax4.axvspan(covid_start, covid_end, alpha=0.1, color=RED,
                    label="COVID period")

    ax4.set_xticks(range(len(q_residuals)))
    ax4.set_xticklabels([str(d)[:7] for d in q_residuals["date"]],
                         rotation=45, ha="right", fontsize=7)
    ax4.set_ylabel("Residual (bps)", fontsize=11)
    ax4.set_title("Mean Residual Over Time",
                  fontsize=11, fontweight="bold", color=NAVY)
    ax4.legend(fontsize=8)

    fig.suptitle(
        "Walk-Forward Validation Results\n"
        "Expanding window — each quarter predicted using only prior data",
        fontsize=13, fontweight="bold", y=1.01
    )
    path = OUTPUT_DIR / "walkforward_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")

    print("\n" + "="*60)
    print("  STEP 6b — WALK-FORWARD VALIDATION")
    print("="*60 + "\n")

    # Load augmented panel (with Altman Z-Score)
    print("[1/3] Loading data...")
    panel = pd.read_csv(PROCESSED_DIR / "panel_augmented.csv",
                        index_col=[0, 1], parse_dates=True)

    print(f"  Panel: {panel.shape[0]} obs, {panel.shape[1]} columns")
    all_quarters = panel.index.get_level_values("date").unique().sort_values()
    print(f"  Quarters: {len(all_quarters)} ({all_quarters[0].date()} → {all_quarters[-1].date()})")

    # Feature columns: everything except the target
    feature_cols = [c for c in panel.columns if c != "cds_spread_bps"]
    print(f"  Features: {feature_cols}")

    # Single-split reference metrics (from step5)
    SINGLE_SPLIT_RMSE = 33.3   # from your model_summary.txt
    SINGLE_SPLIT_R2   = 0.849  # from your model_summary.txt

    # [2] Run walk-forward
    print("\n[2/3] Running walk-forward validation...")
    print("  (This fits one SVR per quarter — ~8-12 minutes total)\n")

    results_df, folds_df = walk_forward_validation(
        panel,
        feature_cols=feature_cols,
        target_col="cds_spread_bps",
        min_train_quarters=8,   # need 2 years before first prediction
    )

    # [3] Summarise
    print("\n[3/3] Generating results...\n")

    overall_rmse = np.sqrt(mean_squared_error(results_df["actual"],
                                               results_df["predicted"]))
    overall_r2   = r2_score(results_df["actual"], results_df["predicted"])
    overall_mae  = mean_absolute_error(results_df["actual"], results_df["predicted"])

    print("="*60)
    print("WALK-FORWARD SUMMARY")
    print("="*60)
    print(f"\n  Folds evaluated:       {len(folds_df)}")
    print(f"  Total predictions:     {len(results_df)}")
    print(f"\n  OVERALL WALK-FORWARD METRICS (all folds combined):")
    print(f"    RMSE: {overall_rmse:.1f} bps")
    print(f"    MAE:  {overall_mae:.1f} bps")
    print(f"    R²:   {overall_r2:.3f}")
    print(f"\n  COMPARISON WITH SINGLE TEMPORAL SPLIT:")
    print(f"    Single-split RMSE: {SINGLE_SPLIT_RMSE:.1f} bps")
    print(f"    Walk-forward RMSE: {overall_rmse:.1f} bps")
    rmse_diff = overall_rmse - SINGLE_SPLIT_RMSE
    print(f"    Difference: {rmse_diff:+.1f} bps "
          f"({'walk-forward is harder' if rmse_diff > 0 else 'similar'})")

    print(f"\n  PER-FOLD RMSE range: {folds_df['rmse'].min():.1f} → "
          f"{folds_df['rmse'].max():.1f} bps")
    print(f"  Best quarter:  {folds_df.loc[folds_df['rmse'].idxmin(), 'test_quarter'].date()} "
          f"(RMSE={folds_df['rmse'].min():.1f})")
    print(f"  Worst quarter: {folds_df.loc[folds_df['rmse'].idxmax(), 'test_quarter'].date()} "
          f"(RMSE={folds_df['rmse'].max():.1f})")

    # COVID vs non-COVID comparison
    folds_df["is_covid"] = folds_df["test_quarter"].between(
        "2020-01-01", "2020-12-31"
    )
    if folds_df["is_covid"].any():
        covid_rmse    = folds_df[folds_df["is_covid"]]["rmse"].mean()
        noncovid_rmse = folds_df[~folds_df["is_covid"]]["rmse"].mean()
        print(f"\n  COVID quarters (2020) avg RMSE:     {covid_rmse:.1f} bps")
        print(f"  Non-COVID quarters avg RMSE:        {noncovid_rmse:.1f} bps")
        print(f"  Performance degradation in stress:  {covid_rmse - noncovid_rmse:+.1f} bps")

    plot_walkforward(results_df, folds_df,
                     SINGLE_SPLIT_RMSE, SINGLE_SPLIT_R2)

    # Save fold-level results
    folds_df.to_csv(OUTPUT_DIR / "walkforward_folds.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "walkforward_predictions.csv", index=False)

    # Write summary
    summary = [
        "WALK-FORWARD VALIDATION SUMMARY",
        "=" * 50,
        "",
        "METHOD: Expanding window, one fold per quarter",
        f"Minimum training quarters: 8 (2 years)",
        f"Total folds: {len(folds_df)}",
        f"Total predictions: {len(results_df)}",
        "",
        "OVERALL METRICS (all folds combined)",
        f"  RMSE: {overall_rmse:.1f} bps",
        f"  MAE:  {overall_mae:.1f} bps",
        f"  R²:   {overall_r2:.3f}",
        "",
        "COMPARISON WITH SINGLE TEMPORAL SPLIT",
        f"  Single-split RMSE: {SINGLE_SPLIT_RMSE:.1f} bps",
        f"  Walk-forward RMSE: {overall_rmse:.1f} bps",
        f"  Difference: {rmse_diff:+.1f} bps",
        "",
        "PER-QUARTER RESULTS",
    ]
    for _, row in folds_df.iterrows():
        summary.append(
            f"  {str(row['test_quarter'])[:10]}  RMSE={row['rmse']:5.1f}  "
            f"R²={row['r2']:6.3f}  C={row['best_C']}  ε={row['best_epsilon']}"
        )

    path = OUTPUT_DIR / "walkforward_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(summary))
    print(f"\n  Summary saved → {path}")

    print(f"\n✓ Walk-forward validation complete.")
    print(f"  Results saved to: {OUTPUT_DIR}")