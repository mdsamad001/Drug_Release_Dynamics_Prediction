# Predicting Early and Complete Drug Release from Long-Acting Injectables Using Explainable Machine Learning

This repository contains the source code for the paper:

> **Predicting Early and Complete Drug Release from Long-Acting Injectables Using Explainable Machine Learning**

## Overview

We present a novel time-independent machine learning framework to predict drug release dynamics from polymer-based long-acting injectable (LAI) microparticle formulations. Three prediction tasks are investigated using material characteristics as the sole input — no time or prior release information is used as a feature.

| Task | Description | Models |
|------|-------------|--------|
| 1 | Fractional drug release at 24h, 48h, 72h | Linear Regression, Random Forest, XGBoost |
| 2 | AUC of complete release profile (regression) | Linear Regression, Random Forest, XGBoost |
| 3 | Release profile type: burst vs sustained (AUC > 0.5) | Logistic Regression, Random Forest, XGBoost |
| 4 | Complete drug release profile prediction | FC-NN-GRU, FC-NN-LSTM, XGB-Time, XGB-NoTime, XGB-Multiregressor |

## Repository Structure

```
├── drug_release_pipeline.py       # Consolidated pipeline: data loading, CV splits,
│                                  # all model definitions and runners for all 4 tasks
├── run_early_tp.ipynb             # Tasks 1 & 2: fractional release and AUC prediction
├── run_classification.ipynb       # Task 3: burst vs sustained classification
├── run_complete_prediction.ipynb  # Task 4: complete release profile prediction
├── mp_dataset_processed_no_dupes.xlsx            # Formulation features (input)
└── mp_dataset_processed_time_release_only.xlsx   # Release profiles (input)
```

## Data

The dataset comprises 321 PLGA microparticle formulations from 88 unique drugs, compiled from 113 publications by Bao et al. [[1]](#references). Formulation features are listed in Table 1 of the paper. Release profiles are normalised to a uniform time grid of 11 interpolation points using Min-Max normalisation.

## Methods

### Cross-Validation
A 10×2 nested cross-validation scheme is used across all tasks:
- **Outer folds (n=10):** Drug-identity-based GroupKFold on Drug SMILES — no drug appears in both train and test, preventing leakage of drug-level physicochemical information
- **Inner folds (n=2):** Hyperparameter tuning via Optuna (50 trials, TPE sampler) within each outer train-val set

### Classification Balancing
For Task 3, random undersampling of the majority class is applied to training folds only. Test sets are never resampled.

### AUC Threshold
Release profiles are classified as burst (AUC > 0.5) or sustained (AUC ≤ 0.5) based on the normalised-time trapezoidal AUC.

## Usage

### Installation

```bash
pip install numpy pandas scikit-learn xgboost optuna shap torch tqdm openpyxl
```

### Running the Notebooks

Run notebooks in this order:

| Order | Notebook | Task |
|-------|----------|------|
| 1 | `run_early_tp.ipynb` | Fractional release at 24h/48h/72h + AUC regression |
| 2 | `run_classification.ipynb` | Burst vs sustained classification |
| 3 | `run_complete_prediction.ipynb` | Complete release profile prediction |

All notebooks import from `drug_release_pipeline.py` which must be in the same directory.

### Using the Pipeline Directly

```python
from drug_release_pipeline import (
    load_data, make_splits, verify_splits,
    run_tp_xgb, run_class_rf, run_gru, run_xgb_multicurve,
)

# Load all data
(X, X_xgb_notime, X_xgb_time,
 y_timepoints, y_auc, y_class,
 y_nn, y_xgb,
 groups_xgb, aucs,
 drug_groups, drug_groups_xgb,
 drug_id_to_smiles, feature_names) = load_data('formulations.xlsx', 'release.xlsx')

# Build CV splits
outer_splits, inner_splits_per_outer = make_splits(drug_groups)
verify_splits(outer_splits, drug_groups)

# Run a model
results = run_tp_xgb(X, y_timepoints, outer_splits, inner_splits_per_outer)
```

## Results Summary

| Task | Best Model | Metric | Score |
|------|-----------|--------|-------|
| 1 (72h) | XGBoost | Pearson r | 0.37 |
| 3 | Random Forest | F1-score | 0.72 |
| 4 | FC-NN-GRU | Overall RMSE | 0.183 |
| 4 | XGB-Multiregressor | Overall RMSE | 0.178 |

Full results are reported in the paper.

## References

[1] Bao et al. (2025). A dataset on formulation parameters and characteristics of drug-loaded PLGA microparticles. *Scientific Data*, 12, 364. https://doi.org/10.1038/s41597-025-04621-9
