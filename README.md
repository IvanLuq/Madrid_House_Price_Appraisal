# Madrid House Price Appraisal

End-to-end Data Science project predicting Madrid property listing prices from public Idealista / Fotocasa data. Covers the full ML lifecycle: data cleaning, feature engineering, EDA, model selection via nested cross-validation, statistical comparison, model interpretation (PDP + SHAP), and a FastAPI appraisal webapp.

This project was developed as the final assignment for the *Applied Data Science* course at Rey Juan Carlos University (Madrid).

## Problem

Given a Madrid listing (size, rooms, neighbourhood, amenities, …), predict its `buy_price` in euros. Two research questions:

1. **Can listing-level features predict price accurately enough to be useful for valuation?**
2. **Which features actually drive the predicted price** — size, location, or amenities?

## Dataset

- **Source:** `houses_Madrid.csv` (~21,742 listings × 58 raw columns) — Idealista / Fotocasa snapshot.
- **Target:** `buy_price` (€), log-transformed via `np.log1p` during training.
- **Stored in:** `data/houses_Madrid.csv`.

## Pipeline summary

Preprocessing (pre-split):
1. Filter rows by `built_year ≥ 1850`.
2. Group-impute amenity flags by build-year decade cohort.
3. Drop columns with > 75 % effective missing ratio (binary-aware — protects `has_*` flags).
4. Mode-fill orientation flags.
5. Parse `neighborhood_id` → integer IDs + one-hot district encoding, drop reference categories.
6. Drop one column of every pair with `|ρ| > 0.9`.

Post-split (inside `ColumnTransformer`):
- Skewed continuous features → log + standard-scale.
- Non-skewed continuous → standard-scale.
- Binary flags → passthrough.
- `VarianceThreshold` filter.

Train / test split is **80/20**, stratified by neighbourhood density.

## Models compared

| Family | Models |
|---|---|
| Linear | LinearRegression, Ridge, ElasticNet |
| Trees | DecisionTree, RandomForest |
| Boosting | XGBoost, LightGBM, GradientBoosting |
| Neural | PyTorch residual MLP (exploratory) |

Selection done via **nested cross-validation** (5 outer × 3 inner folds) with `RandomizedSearchCV`.

## Headline results

**10-fold cross-validation on the training set** (pre-tuning):

| Model | R² | MAE (€) | MAPE | RMSE (€) |
|---|---|---|---|---|
| **XGBoost** | **0.946** | 111,465 | 15.82 % | 254,878 |
| LightGBM | 0.945 | 114,162 | 16.05 % | 259,755 |
| RandomForest | 0.940 | 110,880 | 16.21 % | 257,409 |
| GradientBoosting | 0.916 | 140,642 | 20.63 % | 311,751 |
| Ridge | 0.912 | 163,258 | 20.41 % | 459,806 |
| DecisionTree | 0.886 | 144,886 | 22.22 % | 347,300 |

**Final test-set evaluation (LightGBM with tuned hyperparameters):**

| Metric | Value |
|---|---|
| R² | **0.885** |
| MAE | € 114,756 |
| MedAE | € 42,705 |
| RMSE | € 242,619 |
| MAPE | 16.23 % |

Paired comparison (single test fold) found **XGBoost** and **LightGBM** statistically indistinguishable; LightGBM was chosen as the production model on training-time grounds. The final fitted model is saved as `best_lgbm.pkl`.

## Key findings

- `sq_mt_built` is by far the dominant non-leakage predictor (RF importance ≈ 0.77, single-feature R² ≈ 0.78). The ceiling around R² ≈ 0.94 in CV is largely carried by size.
- Validation curves all hit their optima at the boundary of the search range — the model is already near its data-imposed performance ceiling. Further hyperparameter tuning gives diminishing returns.
- Linear models are competitive on R² but pay a large MAPE penalty — they underestimate the very expensive listings.
- SHAP analysis confirms size as the global driver, with district encoding (Salamanca, Chamartín, Centro) as the second-strongest signal.

## What's in this repo

```
├── working.ipynb                       # The full working notebook
├── Data_Science_Final_Project.pdf      # Written report
├── Madrid_Property_Prices_Presentation.pptx
├── best_lgbm.pkl                       # Final fitted LightGBM model (used by webapp)
├── data/houses_Madrid.csv              # Input dataset
├── webapp/                             # FastAPI appraisal demo
├── plots/                              # Generated figures
└── *.png                               # Diagnostic plots (overfit, PDP, SHAP, …)
```

## Want the detail?

The notebook and the report cover everything in depth — preprocessing rationale, EDA, model selection, statistical tests, interpretation. **For a full walkthrough see:**

- **`Data_Science_Final_Project.pdf`** — written report (recommended starting point)
- **`working.ipynb`** — runnable end-to-end notebook with all code, plots, and outputs

## Running the notebook

```bash
pip install pandas numpy scikit-learn matplotlib scipy xgboost lightgbm torch shap regex
jupyter notebook working.ipynb
```

Tested on Python 3.14. The notebook expects to be launched from the repo root so `./data/houses_Madrid.csv` resolves.

## Webapp

A small FastAPI demo lives in `webapp/` and serves the fitted `best_lgbm.pkl` model as an interactive appraisal form. See `webapp/README.md` for setup.

## Authors

University group project — see commit history.
