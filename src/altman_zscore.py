"""
step6a_altman_zscore.py
=======================
Tier 2 Feature Engineering: Altman Z-Score

Adds the Altman Z-Score as a financially-motivated feature and runs
a proper ablation study — does it actually improve the model?

Run after step5_model.py:
    python src/step6a_altman_zscore.py

What this covers:
    - What the Altman Z-Score is and why it belongs in a CDS model
    - How to construct it from available financial ratios
    - What "ablation study" means and how to run one correctly
    - How to interpret whether a new feature genuinely helps

Output:
    outputs/altman_zscore_ablation.png
    outputs/altman_summary.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


# ─────────────────────────────────────────────────────────────────────────────
#  THE ALTMAN Z-SCORE: WHAT IT IS AND WHY IT BELONGS HERE
# ─────────────────────────────────────────────────────────────────────────────
#
# Edward Altman (1968) developed the Z-Score to predict corporate bankruptcy
# using five financial ratios. It remains one of the most widely cited
# default-prediction models in academic and practitioner finance.
#
# The original formula (for publicly traded manufacturers):
#
#   Z = 1.2 × X1  +  1.4 × X2  +  3.3 × X3  +  0.6 × X4  +  1.0 × X5
#
# Where:
#   X1 = Working Capital / Total Assets
#        Working capital = current assets - current liabilities
#        Measures short-term liquidity relative to asset base.
#        Negative = company can't cover near-term obligations from operations.
#
#   X2 = Retained Earnings / Total Assets
#        Measures cumulative profitability relative to assets.
#        Young or loss-making companies have low/negative retained earnings.
#        We APPROXIMATE this with ROA × 5 (a defensible proxy —
#        consistently profitable companies build retained earnings over time).
#
#   X3 = EBIT / Total Assets
#        Pure operating profitability — how efficiently assets generate earnings.
#        We APPROXIMATE with ROA (net income / assets) since we lack separate EBIT.
#        ROA and EBIT/Assets are highly correlated in practice.
#
#   X4 = Market Cap / Total Liabilities
#        Equity cushion relative to debt. How much equity value exists to absorb
#        losses before creditors are impaired?
#        We approximate: Total Liabilities ≈ Total Assets × Leverage Ratio
#        So X4 ≈ Market Cap / (Market Cap + Debt) × (1/Leverage)
#
#   X5 = Sales / Total Assets (Asset Turnover)
#        Revenue generated per unit of assets. High turnover = efficient use of assets.
#        We approximate using 1/Debt-to-EBITDA as a revenue-productivity proxy.
#
# INTERPRETATION:
#   Z > 2.99  : "Safe Zone"  — low default probability
#   1.81–2.99 : "Grey Zone"  — uncertain
#   Z < 1.81  : "Distress Zone" — high default probability
#
# WHY IT BELONGS IN A CDS MODEL:
#   CDS spreads price default risk. The Z-Score is a direct measure of default
#   probability derived from accounting data. Adding it as a feature gives the
#   SVR access to a compressed, academically validated summary of financial
#   health — one that's been shown to predict default 1–2 years ahead.
#   The spec explicitly recommends it as an alternative to Moody's EDF.

def compute_altman_z(panel: pd.DataFrame) -> pd.Series:
    """
    Computes a modified Altman Z-Score from available features.

    Because we don't have every original component (we lack total assets
    and sales directly), we use financially motivated approximations from
    what yfinance gave us. We document every approximation so the report
    is honest about this.

    APPROXIMATIONS MADE:
    --------------------
    X1 (Working Capital / Assets):
        current_ratio = current_assets / current_liabilities
        Working Capital / Assets ≈ (current_ratio - 1) / (1 + 1/leverage_ratio)
        Rationale: if current_ratio=2, current_assets are twice current_liabilities,
        so WC = current_liabilities × 1 = half of current assets.
        We scale by leverage to get to a fraction of total assets.

    X2 (Retained Earnings / Assets):
        Approximated as roa × 5
        Rationale: A company with 10% ROA consistently over 5 years accumulates
        approximately 50% of assets as retained earnings. This is a rough but
        directionally correct proxy.

    X3 (EBIT / Assets):
        Approximated as roa (net income / assets)
        EBIT is pre-interest, pre-tax; net income is post.
        The two are strongly correlated across companies.
        We note this approximation leads to slight downward bias in X3.

    X4 (Market Cap / Total Liabilities):
        total_liabilities ≈ total_assets × leverage_ratio
        total_assets ≈ market_cap / (roa) when roa>0 (book value proxy)
        Simplification: X4 ≈ (1 - leverage_ratio) / leverage_ratio
        = the equity fraction divided by the debt fraction
        This preserves the key signal: high leverage → low X4 → higher distress.

    X5 (Sales / Assets — asset turnover):
        Approximated as 1 / debt_to_ebitda (when debt_to_ebitda > 0)
        EBITDA is a proxy for earnings power; debt/EBITDA inversely relates
        to how efficiently assets generate revenue.
    """
    z = pd.Series(np.nan, index=panel.index, name="altman_z")

    for idx, row in panel.iterrows():
        try:
            # ── Extract components ────────────────────────────────────────────
            cr  = row.get("current_ratio",    np.nan)   # current assets / current liab
            lev = row.get("leverage_ratio",   np.nan)   # debt / assets
            roa = row.get("roa",              np.nan)   # net income / assets
            mkt = row.get("market_cap",       np.nan)   # USD
            d2e = row.get("debt_to_ebitda",   np.nan)   # debt / ebitda

            # ── X1: Working Capital / Total Assets ────────────────────────────
            # WC = current_assets - current_liabilities
            # WC/TA ≈ (cr - 1) × current_liabilities / total_assets
            # Since leverage = debt/assets, and current liab ≈ leverage × assets / 2
            # We simplify: WC/TA ≈ (cr - 1) × leverage × 0.3
            # (0.3 is an empirical fraction of total debt that is short-term)
            if not np.isnan(cr) and not np.isnan(lev) and lev > 0:
                x1 = np.clip((cr - 1) * lev * 0.3, -0.5, 0.5)
            else:
                x1 = 0.0

            # ── X2: Retained Earnings / Total Assets ─────────────────────────
            if not np.isnan(roa):
                x2 = np.clip(roa * 5, -0.5, 0.8)
            else:
                x2 = 0.0

            # ── X3: EBIT / Total Assets ───────────────────────────────────────
            if not np.isnan(roa):
                x3 = np.clip(roa, -0.2, 0.3)
            else:
                x3 = 0.0

            # ── X4: Market Cap / Total Liabilities ────────────────────────────
            if not np.isnan(lev) and lev > 0:
                # Equity fraction / Debt fraction
                equity_frac = 1 - lev
                x4 = np.clip(equity_frac / lev, 0, 10)
            elif not np.isnan(lev):
                x4 = 5.0   # very low leverage → high X4 (safe)
            else:
                x4 = 1.0

            # ── X5: Sales / Total Assets (Asset Turnover) ────────────────────
            if not np.isnan(d2e) and d2e > 0:
                x5 = np.clip(1 / d2e, 0.05, 3.0)
            else:
                x5 = 0.5

            # ── Altman Formula ────────────────────────────────────────────────
            z_score = (1.2 * x1 +
                       1.4 * x2 +
                       3.3 * x3 +
                       0.6 * x4 +
                       1.0 * x5)

            z.loc[idx] = z_score

        except Exception:
            continue   # leave as NaN for this observation

    return z


# ─────────────────────────────────────────────────────────────────────────────
#  ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IS AN ABLATION STUDY?
# --------------------------
# Borrowed from neuroscience (where you remove brain regions to study function),
# in ML "ablation" means systematically removing or adding features to measure
# their individual contribution.
#
# For each new feature, we answer: does adding it improve test-set performance?
# If R² goes from 0.849 to 0.861, the feature helped.
# If R² stays at 0.849 or drops, the feature is redundant or harmful.
#
# CORRECT PROCEDURE:
# 1. Train and evaluate WITHOUT the new feature (baseline)
# 2. Train and evaluate WITH the new feature (augmented)
# 3. Compare test-set metrics (RMSE, MAE, R²)
# 4. Both models tuned independently with GridSearchCV
#
# Common mistake: students add a feature and train only with it, then compare
# to a different baseline. The split, tuning, and evaluation must be identical.

PARAM_GRID = {
    "C":       [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.05, 0.1, 0.2],
    "gamma":   ["scale", 0.01, 0.1],
}

def run_svr_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series,
                     y_train_bps: pd.Series, y_test_bps: pd.Series,
                     label: str) -> dict:
    """
    Full SVR pipeline: impute → scale → tune → train → evaluate.
    Returns metrics dict. Used for fair comparison in ablation.
    """
    # Impute (fit on train only)
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    Xtr = pd.DataFrame(imp.transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(imp.transform(X_test),  columns=X_test.columns,  index=X_test.index)

    # Scale continuous features only (not dummies)
    dummy_cols = [c for c in Xtr.columns if Xtr[c].nunique() <= 2]
    scale_cols = [c for c in Xtr.columns if c not in dummy_cols]
    sc = StandardScaler()
    sc.fit(Xtr[scale_cols])
    Xtr[scale_cols] = sc.transform(Xtr[scale_cols])
    Xte[scale_cols] = sc.transform(Xte[scale_cols])

    # Log-transform target
    ytr_log = np.log1p(y_train_bps)
    yte_log = np.log1p(y_test_bps)

    # Tune
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(SVR(kernel="rbf", max_iter=50000), PARAM_GRID,
                      cv=cv, scoring="neg_mean_squared_error",
                      n_jobs=-1, verbose=0)
    gs.fit(Xtr, ytr_log)
    best = gs.best_params_

    # Train final model
    model = SVR(kernel="rbf", max_iter=50000, **best)
    model.fit(Xtr, ytr_log)

    # Predict + back-transform
    pred_bps = np.expm1(model.predict(Xte))
    pred_bps = np.clip(pred_bps, 0, None)

    metrics = {
        "label":   label,
        "rmse":    np.sqrt(mean_squared_error(y_test_bps, pred_bps)),
        "mae":     mean_absolute_error(y_test_bps, pred_bps),
        "r2":      r2_score(y_test_bps, pred_bps),
        "best_params": best,
        "n_features": X_train.shape[1],
        "n_sv":    model.n_support_[0],
        "pct_sv":  model.n_support_[0] / len(X_train) * 100,
        "pred_bps": pred_bps,
        "y_true":  np.array(y_test_bps),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(results: list) -> None:
    """
    Two-panel ablation plot:
    Left  — bar chart of RMSE and R² for each model configuration
    Right — scatter overlay: baseline vs. Z-Score model predictions
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [r["label"] for r in results]
    rmses  = [r["rmse"]  for r in results]
    r2s    = [r["r2"]    for r in results]
    colors = [NAVY, GREEN, AMBER, RED][:len(results)]

    # ── Left: metric comparison bar chart ────────────────────────────────────
    ax = axes[0]
    x    = np.arange(len(labels))
    w    = 0.35
    ax2  = ax.twinx()   # second y-axis for R²

    bars1 = ax.bar(x - w/2, rmses, w, color=colors, alpha=0.8,
                   edgecolor="white", label="RMSE (bps)")
    bars2 = ax2.bar(x + w/2, r2s, w, color=colors, alpha=0.4,
                    edgecolor="white", hatch="//", label="R²")

    # Value labels on bars
    for bar, val in zip(bars1, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", fontsize=9, color=GREY)

    ax.set_ylabel("RMSE (bps) — lower is better", fontsize=11)
    ax2.set_ylabel("R² — higher is better", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Ablation Study: Feature Contribution", fontsize=12,
                 fontweight="bold", color=NAVY)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines = [plt.Rectangle((0,0),1,1, color=NAVY, alpha=0.8),
             plt.Rectangle((0,0),1,1, color=NAVY, alpha=0.4, hatch="//")]
    ax.legend(lines, ["RMSE (bps)", "R²"], fontsize=9, loc="upper left")

    # ── Right: residual comparison — baseline vs best model ──────────────────
    ax3 = axes[1]
    baseline = results[0]
    best_res = min(results, key=lambda r: r["rmse"])

    ax3.scatter(baseline["y_true"], baseline["pred_bps"],
                alpha=0.5, s=40, color=NAVY, label=f"Baseline (R²={baseline['r2']:.3f})")

    if best_res["label"] != baseline["label"]:
        ax3.scatter(best_res["y_true"], best_res["pred_bps"],
                    alpha=0.5, s=40, color=GREEN, marker="^",
                    label=f"{best_res['label']} (R²={best_res['r2']:.3f})")

    lo = min(baseline["y_true"].min(), baseline["pred_bps"].min()) * 0.95
    hi = max(baseline["y_true"].max(), baseline["pred_bps"].max()) * 1.05
    ax3.plot([lo, hi], [lo, hi], "--", color=AMBER, linewidth=2, label="Perfect prediction")
    ax3.set_xlabel("Actual CDS Spread (bps)", fontsize=11)
    ax3.set_ylabel("Predicted CDS Spread (bps)", fontsize=11)
    ax3.set_title("Baseline vs Best Model: Predicted vs Actual",
                  fontsize=12, fontweight="bold", color=NAVY)
    ax3.legend(fontsize=9)
    ax3.set_xlim(lo, hi)
    ax3.set_ylim(lo, hi)

    plt.tight_layout()
    path = OUTPUT_DIR / "altman_zscore_ablation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")

    print("\n" + "="*60)
    print("  STEP 6a — ALTMAN Z-SCORE FEATURE ENGINEERING")
    print("="*60 + "\n")

    # Load panel and existing split
    print("[1/4] Loading data...")
    panel = pd.read_csv(PROCESSED_DIR / "panel_final.csv",
                        index_col=[0,1], parse_dates=True)
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv",
                          index_col=[0,1], parse_dates=True)
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv",
                          index_col=[0,1], parse_dates=True)
    y_train_bps = pd.read_csv(PROCESSED_DIR / "y_train_bps.csv",
                              index_col=[0,1], parse_dates=True).iloc[:,0]
    y_test_bps  = pd.read_csv(PROCESSED_DIR / "y_test_bps.csv",
                              index_col=[0,1], parse_dates=True).iloc[:,0]

    # Align y with X (index may differ slightly)
    y_train_bps = y_train_bps.reindex(X_train.index)
    y_test_bps  = y_test_bps.reindex(X_test.index)

    print(f"  Train: {len(X_train)} obs | Test: {len(X_test)} obs")
    print(f"  Baseline features: {list(X_train.columns)}")

    # [2] Compute Altman Z-Score for whole panel
    print("\n[2/4] Computing Altman Z-Score...")
    z_scores = compute_altman_z(panel)

    # Print distribution summary
    z_clean = z_scores.dropna()
    print(f"  Z-Scores computed for {len(z_clean)} / {len(panel)} observations")
    print(f"  Range:  {z_clean.min():.2f} → {z_clean.max():.2f}")
    print(f"  Mean:   {z_clean.mean():.2f} | Median: {z_clean.median():.2f}")
    print(f"\n  Distress zone  (Z < 1.81): {(z_clean < 1.81).mean()*100:.0f}% of obs")
    print(f"  Grey zone  (1.81 ≤ Z < 2.99): {((z_clean >= 1.81) & (z_clean < 2.99)).mean()*100:.0f}% of obs")
    print(f"  Safe zone      (Z ≥ 2.99): {(z_clean >= 2.99).mean()*100:.0f}% of obs")

    # Check correlation with CDS spread (should be negative — higher Z = safer)
    spread_and_z = pd.concat([panel["cds_spread_bps"], z_scores], axis=1).dropna()
    corr = spread_and_z.corr().iloc[0, 1]
    print(f"\n  Correlation with CDS spread: {corr:.3f}")
    if corr < -0.1:
        print(f"  ✓ Negative correlation — higher Z-Score → lower spread (as expected)")
    else:
        print(f"  ⚠ Weak/positive correlation — Z-Score approximation may be noisy")

    # Attach Z-Score to train and test sets
    X_train_z = X_train.copy()
    X_test_z  = X_test.copy()
    X_train_z["altman_z"] = z_scores.reindex(X_train.index)
    X_test_z["altman_z"]  = z_scores.reindex(X_test.index)

    print(f"\n  Missing Z-Scores in train: {X_train_z['altman_z'].isna().sum()} "
          f"({X_train_z['altman_z'].isna().mean()*100:.0f}%)")
    print(f"  Missing Z-Scores in test:  {X_test_z['altman_z'].isna().sum()} "
          f"({X_test_z['altman_z'].isna().mean()*100:.0f}%)")

    # [3] Ablation study
    print("\n[3/4] Running ablation study...")
    print("  Running 3 configurations — each with full GridSearchCV tuning")
    print("  This takes ~3-5 minutes\n")

    results = []

    print("  Configuration 1/3: Baseline (no Altman Z-Score)...")
    r1 = run_svr_pipeline(X_train, X_test, None, None,
                          y_train_bps, y_test_bps, "Baseline")
    results.append(r1)
    print(f"    → RMSE={r1['rmse']:.1f} bps | R²={r1['r2']:.3f} | "
          f"params={r1['best_params']}")

    print("\n  Configuration 2/3: With Altman Z-Score added...")
    r2 = run_svr_pipeline(X_train_z, X_test_z, None, None,
                          y_train_bps, y_test_bps, "+ Altman Z")
    results.append(r2)
    print(f"    → RMSE={r2['rmse']:.1f} bps | R²={r2['r2']:.3f} | "
          f"params={r2['best_params']}")

    # Also run without market_cap to test its contribution
    print("\n  Configuration 3/3: Z-Score only, no market_cap (decompose sources)...")
    X_train_nomkt = X_train_z.drop(columns=["market_cap"], errors="ignore")
    X_test_nomkt  = X_test_z.drop(columns=["market_cap"], errors="ignore")
    r3 = run_svr_pipeline(X_train_nomkt, X_test_nomkt, None, None,
                          y_train_bps, y_test_bps, "Z-Score, no mktcap")
    results.append(r3)
    print(f"    → RMSE={r3['rmse']:.1f} bps | R²={r3['r2']:.3f} | "
          f"params={r3['best_params']}")

    # [4] Summary and plot
    print("\n[4/4] Plotting and summarising...")
    plot_ablation(results)

    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(f"\n  {'Configuration':<25} {'RMSE':>8} {'MAE':>8} {'R²':>7} {'#Features':>10}")
    print(f"  {'─'*60}")
    for r in results:
        delta_r2 = r["r2"] - results[0]["r2"]
        direction = f"(+{delta_r2:.3f})" if delta_r2 > 0 else f"({delta_r2:.3f})"
        print(f"  {r['label']:<25} {r['rmse']:>7.1f}  {r['mae']:>7.1f}  "
              f"{r['r2']:>6.3f} {direction:>10}  [{r['n_features']} features]")

    best = min(results, key=lambda r: r["rmse"])
    improvement = results[0]["rmse"] - best["rmse"]
    print(f"\n  Best configuration: '{best['label']}'")
    print(f"  RMSE improvement over baseline: {improvement:.1f} bps")

    if improvement > 2:
        verdict = "Z-Score adds meaningful predictive value — include in final model."
    elif improvement > 0:
        verdict = "Z-Score adds marginal improvement — include with caveat in report."
    else:
        verdict = "Z-Score does not improve this model — report this honestly."
    print(f"  Verdict: {verdict}")

    # Save Z-Score augmented datasets for use in walk-forward validation
    X_train_z.to_csv(PROCESSED_DIR / "X_train_augmented.csv")
    X_test_z.to_csv(PROCESSED_DIR / "X_test_augmented.csv")
    panel["altman_z"] = z_scores
    panel.to_csv(PROCESSED_DIR / "panel_augmented.csv")
    print(f"\n  Augmented datasets saved to data/processed/")
    print(f"  (Used by step6b_walkforward.py)")

    # Write summary
    summary_lines = [
        "ALTMAN Z-SCORE ABLATION SUMMARY",
        "=" * 50,
        "",
        "FEATURE CONSTRUCTION",
        "  Formula: Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5",
        "  X1 (WC/TA):   approximated from current_ratio + leverage_ratio",
        "  X2 (RE/TA):   approximated as roa × 5",
        "  X3 (EBIT/TA): approximated as roa",
        "  X4 (MktCap/Liab): (1-leverage)/leverage",
        "  X5 (Sales/TA): 1/debt_to_ebitda",
        "",
        "Z-SCORE DISTRIBUTION",
        f"  Observations: {len(z_clean)}",
        f"  Range:  {z_clean.min():.2f} → {z_clean.max():.2f}",
        f"  Distress zone (Z<1.81):   {(z_clean<1.81).mean()*100:.0f}%",
        f"  Grey zone (1.81-2.99):    {((z_clean>=1.81)&(z_clean<2.99)).mean()*100:.0f}%",
        f"  Safe zone (Z≥2.99):       {(z_clean>=2.99).mean()*100:.0f}%",
        f"  Correlation with spread:  {corr:.3f}",
        "",
        "ABLATION RESULTS",
    ]
    for r in results:
        summary_lines.append(
            f"  {r['label']:<25} RMSE={r['rmse']:.1f} MAE={r['mae']:.1f} R²={r['r2']:.3f}"
        )
    summary_lines += ["", f"Verdict: {verdict}"]

    path = OUTPUT_DIR / "altman_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Summary saved → {path}")

    print("\n✓ Step 6a complete. Run step6b_walkforward.py next.")