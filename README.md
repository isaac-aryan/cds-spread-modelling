# CDS Spread Modelling with Support Vector Regression

A complete end-to-end pipeline for estimating 5-year Credit Default Swap (CDS) spread proxies using Support Vector Regression, built entirely from publicly available data.

**[Link to Full Report (PDF)]()**

---

## Overview

Financial institutions must estimate credit risk daily for companies whose CDS spreads are not actively quoted in the market. This project builds a supervised learning pipeline that learns the relationship between macroeconomic conditions, company characteristics, and credit spread levels from a panel of 34 US companies observed quarterly from Q1 2018 to Q4 2023.

Key results:
- **R² = 0.849**, **RMSE = 33.3 bps** on a temporally stratified held-out test set
- Walk-forward validation across 16 quarterly folds reveals a **4.6× performance degradation** during COVID-19 stress quarters (RMSE: 22.9 bps in stable conditions → 106.3 bps in 2020)
- Support vector analysis identifies a monotonic relationship between credit rating and model difficulty — every B-rated company is a support vector against 50% for AAA-rated companies

---

## Project Structure

```
cds-svm/
├── src/
│   ├── fred.py               # Step 1 — Fetch macro data from FRED API
│   ├── equities.py           # Step 2 — Fetch equity prices and volatility from Yahoo Finance
│   ├── build_panel.py        # Step 3 — Merge sources, construct CDS spread proxy, build panel
│   ├── preprocess.py         # Step 4 — Encode, split, impute, scale, log-transform
│   ├── model.py              # Step 5 — SVR tuning, training, evaluation, and plots
│   ├── altman_zscore.py      # Step 6a — Altman Z-Score feature engineering and ablation
│   └── walk_forward.py       # Step 6b — Expanding-window walk-forward validation
├── data/
│   ├── raw/                  # Cached API outputs (auto-generated, not tracked)
│   └── processed/            # Model-ready CSVs (auto-generated, not tracked)
├── outputs/                  # Plots and model summary (auto-generated)
├── CDS_Spread_Modelling_with_SVR_Report.pdf
├── requirements.txt
└── README.md
```

Each script is independently executable and caches its output to disk. Any step can be re-run in isolation without repeating upstream API calls.

---

## Data Sources

| Data | Source | Series / Method |
|------|--------|-----------------|
| HY OAS spread | FRED (St. Louis Fed) | BAMLH0A0HYM2 |
| IG OAS spread | FRED (St. Louis Fed) | BAMLC0A4CBBBEY |
| VIX | FRED (St. Louis Fed) | VIXCLS |
| 10Y Treasury yield | FRED (St. Louis Fed) | DGS10 |
| USD Index | FRED (St. Louis Fed) | DTWEXBGS |
| Equity prices + volatility | Yahoo Finance (yfinance) | `.history()` |
| Credit ratings, sector, region | Public agency ratings | Hand-assigned at 2018 |
| CDS spreads | **Constructed proxy** | See Section 3.2 of report |

**Note on the target variable:** Real 5-year CDS spread quotes require Markit data (Bloomberg/WRDS). The proxy used here is driven by real FRED market prices and calibrated to empirical spread ranges. The pipeline is architected to accept real Markit data as a drop-in replacement — only the target CSV would change.

---

## Setup

```bash
git clone https://github.com/yourusername/cds-svm.git
cd cds-svm
pip install -r requirements.txt
```

A free FRED API key is required for Step 1. Register at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) and set it as an environment variable:

```bash
export FRED_API_KEY="your_key_here"
```

---

## Running the Pipeline

Run the scripts in order. Each step saves its output to `data/raw/` or `data/processed/` and loads it on subsequent runs — re-running a step does not repeat upstream API calls.

```bash
python src/fred.py            # ~10 seconds
python src/equities.py        # ~10 minutes (rate-limited API calls)
python src/build_panel.py     # ~30 seconds
python src/preprocess.py      # ~5 seconds
python src/model.py           # ~5 minutes (625 SVR fits)
python src/altman_zscore.py   # ~5 minutes (ablation study)
python src/walk_forward.py    # ~15 minutes (16 quarterly folds)
```

All results are saved to `outputs/`. The model summary is written to `outputs/model_summary.txt`.

---

## Results Summary

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| RMSE | 9.9 bps | 33.3 bps |
| MAE | 5.1 bps | 20.9 bps |
| R² | 0.993 | 0.849 |

**Walk-forward validation:**

| Method | RMSE | R² |
|--------|------|----|
| Single temporal split | 33.3 bps | 0.849 |
| Walk-forward (all 16 folds) | 61.0 bps | 0.764 |
| Walk-forward (non-COVID quarters) | 22.9 bps | — |
| Walk-forward (COVID quarters 2020) | 106.3 bps | — |

**Optimal hyperparameters:** C=10, ε=0.01, γ=0.05 (RBF kernel, 5-fold cross-validated grid search over 625 combinations)

**Support vectors:** 457 / 612 training observations (74.7%)

---

## Key Findings

**Regime-dependence under COVID stress.** The model performs well in stable market conditions but degrades 4.6× during the COVID-19 shock quarters of 2020. A model trained on 2018–2019 data had no precedent for simultaneous VIX > 80, HY OAS > 700 bps, and 100–400 bps spread moves across all sectors. This quantifies a general principle: historically-trained credit models require explicit stress-testing before deployment.

**Rating-dependent model difficulty.** Support vector membership increases monotonically from 50% for AAA-rated companies to 100% for B-rated companies. Investment-grade spreads are largely explained by systematic macro factors (VIX, market OAS), which the model captures. High-yield spreads are dominated by idiosyncratic factors — covenant structures, debt maturity profiles, company-specific events — that publicly available features cannot represent.

**Altman Z-Score adds no predictive value.** A controlled ablation study showed RMSE is unchanged by adding the Z-Score (18.7 → 19.1 bps). This is a feature redundancy finding: the baseline feature set already encodes the information the Z-Score summarises, since its constituent ratios are already present as direct features.

---

## Limitations

- **Proxy target variable:** Performance metrics measure accuracy against the constructed proxy, not against real Markit CDS quotes. See Section 6 of the report for full discussion.
- **Missing fundamentals:** Yahoo Finance does not archive historical quarterly balance sheets. Leverage ratio, interest coverage, ROA, and current ratio could not be included as time-varying features.
- **Fixed credit ratings:** Ratings are held constant at 2018 levels. Real rating migrations (e.g. Ford's 2020 downgrade to HY) are not captured.

---

## References

See the full reference list in the report. Core citations:

- Merton, R. C. (1974). On the pricing of corporate debt. *Journal of Finance*, 29(2), 449–470.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609.
- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
- Longstaff, F. A., Mithal, S., & Neis, E. (2005). Corporate yield spreads: Default risk or liquidity? *Journal of Finance*, 60(5), 2213–2253.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.