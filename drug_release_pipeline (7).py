"""
drug_release_pipeline.py
========================
Consolidated pipeline for predicting drug release dynamics from polymer
microparticle characteristics.

Replaces four separate modules:
    drug_release_cv_analysis.py      → Sections 1–3, 10
    drug_release_class.py            → Sections 1–2, 4–6, 8–9
    drug_release_models.py           → Sections 1–2, 5, 7, 9
    drug_release_tasks_balanced.py   → Section 8 (balanced runners)

Prediction tasks
----------------
    Task 1 — Fractional drug release at 24h, 48h, 72h (multi-output regression)
    Task 2 — AUC of full release curve (single-output regression)
    Task 3 — Release type: burst vs sustained (binary classification)
              AUC > 0.5 of normalised-time curve → burst (1), else sustained (0)
    Task 4 — Complete drug release profile (sequence output; GRU / LSTM / XGB)

CV strategy
-----------
    Outer : GroupKFold(n=10) stratified by Drug SMILES.
            No drug appears in both train and test in any fold.
    Inner : GroupKFold(n=2) within each outer train-val set for Optuna search.
    All tasks share the same outer splits → per-fold metrics are paired and
    Wilcoxon signed-rank tests are valid.

Public API — data
-----------------
    load_data(form_path, release_path)   → all arrays needed by every task
    make_splits(drug_groups)             → outer_splits, inner_splits_per_outer
    verify_splits(outer_splits, drug_groups)
    analyze_folds(...)                   → per-fold distribution DataFrames

Public API — Task 1 runners (timepoint regression)
---------------------------------------------------
    run_tp_xgb(X, y_tp, outer_splits, inner_splits_per_outer)
    run_tp_rf(X, y_tp, outer_splits, inner_splits_per_outer)
    run_tp_linear(X, y_tp, outer_splits, inner_splits_per_outer)

Public API — Task 2 runners (AUC regression)
--------------------------------------------
    run_auc_xgb(X, y_auc, outer_splits, inner_splits_per_outer)
    run_auc_rf(X, y_auc, outer_splits, inner_splits_per_outer)
    run_auc_linear(X, y_auc, outer_splits, inner_splits_per_outer)

Public API — Task 3 runners (classification, standard)
-------------------------------------------------------
    run_class_xgb(X, y_class, outer_splits, inner_splits_per_outer)
    run_class_rf(X, y_class, outer_splits, inner_splits_per_outer)
    run_class_logistic(X, y_class, outer_splits, inner_splits_per_outer)

Public API — Task 3 runners (classification, balanced via random undersampling)
-------------------------------------------------------------------------------
    run_class_xgb_balanced(X, y_class, outer_splits, inner_splits_per_outer, drug_groups)
    run_class_rf_balanced(X, y_class, outer_splits, inner_splits_per_outer, drug_groups)
    run_class_logistic_balanced(X, y_class, outer_splits, inner_splits_per_outer, drug_groups)

Public API — Task 4 runners (complete profile prediction)
----------------------------------------------------------
    run_gru(X_form, y_nn, outer_splits, inner_splits_per_outer, aucs=None)
    run_lstm(X_form, y_nn, outer_splits, inner_splits_per_outer, aucs=None)
    run_xgb_time(X_xgb_time, y_xgb, groups_xgb, drug_groups_xgb, outer_splits, ..., aucs=None)
    run_xgb_notime(X_xgb_notime, y_xgb, groups_xgb, drug_groups_xgb, outer_splits, ..., aucs=None)
    run_xgb_multicurve(X_form, y_nn, outer_splits, inner_splits_per_outer, aucs=None)

Public API — comparison & statistics
-------------------------------------
    compare_tp_models(all_results)
    compare_auc_models(all_results)
    compare_class_models(all_results)
    compare_models(all_results)         ← Task 4 profile models
    wilcoxon_tests(all_results, metric)
    results_to_df(all_results)

CLI (fold distribution analysis)
---------------------------------
    python drug_release_pipeline.py --form <path> --release <path> [--save <dir>]
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.integrate import trapezoid
from scipy.stats import pearsonr, wilcoxon as _wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — GLOBAL CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEED            = 42
N_OUTER_FOLDS   = 10
N_INNER_FOLDS   = 2
N_TRIALS        = 50
TIMEPOINTS      = [1, 2, 3]       # days — Task 1 (24h, 48h, 72h)
NUM_INTERP_PTS  = 11              # interpolation grid points
NUM_FEATURES    = 11              # formulation feature count
AUC_THRESHOLD   = 0.5            # burst vs sustained boundary (normalised-time AUC)

# ── Neural network architecture defaults ─────────────────────────────────────
HIDDEN_DIM1   = 32
HIDDEN_DIM2   = 64
INNER_EPOCHS  = 250
OUTER_EPOCHS  = 500

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[drug_release_pipeline] Using device: {device}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data(form_path: str, release_path: str):
    """
    Load and preprocess both Excel files, returning arrays for all four tasks.

    Drug SMILES is captured before encoding/dropping to build drug-level
    group labels used for CV splitting. Preprocessing is consistent across
    all tasks.

    Parameters
    ----------
    form_path    : path to formulations Excel file
    release_path : path to release profiles Excel file

    Returns
    -------
    X                : np.ndarray (321, 11)   formulation features (Tasks 1–3)
    X_xgb_notime     : np.ndarray (3531, 11)  per-timepoint rows, no time col (Task 4)
    X_xgb_time       : np.ndarray (3531, 12)  per-timepoint rows, with norm-time (Task 4)
    y_timepoints     : np.ndarray (321, 3)    release at 1, 2, 3 days (Task 1)
    y_auc            : np.ndarray (321,)      raw-time AUC scalar (Task 2)
    y_class          : np.ndarray (321,)      burst label; 1 if norm-time AUC > 0.5 (Task 3)
    y_nn             : np.ndarray (321, 11, 1) interpolated release sequences (Task 4 RNN)
    y_xgb            : np.ndarray (3531,)     flat release values (Task 4 XGB)
    groups_xgb       : np.ndarray (3531,)     formulation-level group id repeated 11×
    aucs             : np.ndarray (321,)      normalised-time AUC per formulation
    drug_groups      : np.ndarray (321,)      integer drug group id per formulation
    drug_groups_xgb  : np.ndarray (3531,)     drug group id per XGB row
    drug_id_to_smiles: dict                   group id → SMILES string
    feature_names    : list of str            feature column names (for SHAP)
    """
    form_df    = pd.read_excel(form_path,    engine="openpyxl")
    release_df = pd.read_excel(release_path, engine="openpyxl")

    # ── Drug group labels (frequency-ordered: most common drug = group 0) ────
    drug_counts       = form_df["Drug SMILES"].value_counts()
    drug_to_id        = {smiles: i for i, smiles in enumerate(drug_counts.index)}
    drug_id_to_smiles = {i: smiles for smiles, i in drug_to_id.items()}
    drug_groups       = form_df["Drug SMILES"].map(drug_to_id).values.astype(int)  # (321,)
    drug_groups_xgb   = np.repeat(drug_groups, NUM_INTERP_PTS)                     # (3531,)

    # ── Feature matrix ────────────────────────────────────────────────────────
    enc_map = {v: i for i, v in enumerate(form_df["Formulation Method"].unique())}
    form_df["Formulation Method Encoded"] = form_df["Formulation Method"].map(enc_map)
    form_df.drop(columns=["Formulation Method", "Drug SMILES"], inplace=True)

    feature_names = form_df.drop(columns=["Formulation Index"]).columns.tolist()
    X = form_df.drop(columns=["Formulation Index"]).to_numpy()  # (321, 11)

    # ── Normalised time grid (shared) ─────────────────────────────────────────
    norm_t = np.linspace(0, 1, NUM_INTERP_PTS)

    # ── Task 1: fractional release at 24h / 48h / 72h ────────────────────────
    tp_rows = []
    for _, g in release_df.groupby("Formulation Index"):
        g = g.sort_values("Time")
        tp_rows.append(np.interp(TIMEPOINTS, g["Time"], g["Release"]))
    y_timepoints = np.array(tp_rows)  # (321, 3)

    # ── Tasks 2, 3 & 4: normalised-time interpolation ────────────────────────
    interpolated_dfs = []
    norm_auc_vals    = []

    for formulation, g in release_df.groupby("Formulation Index"):
        g = g.sort_values("Time")
        t_min, t_max = g["Time"].min(), g["Time"].max()
        t_norm   = (g["Time"] - t_min) / (t_max - t_min)
        interp   = np.interp(norm_t, t_norm, g["Release"])
        norm_auc_vals.append(np.trapz(interp, norm_t))
        interpolated_dfs.append(pd.DataFrame({
            "Formulation Index":    formulation,
            "Normalized Time":      norm_t,
            "Interpolated Release": interp,
        }))

    interp_df = pd.concat(interpolated_dfs, ignore_index=True)
    aucs      = np.array(norm_auc_vals)           # (321,) normalised-time AUC
    y_class   = (aucs > AUC_THRESHOLD).astype(int) # (321,) burst=1, sustained=0

    # ── Task 2: AUC using raw-time trapezoid ──────────────────────────────────
    auc_vals = []
    for _, g in release_df.groupby("Formulation Index"):
        g = g.sort_values("Time")
        t_min, t_max  = g["Time"].min(), g["Time"].max()
        interp_times  = np.linspace(t_min, t_max, NUM_INTERP_PTS)
        interp_release = np.interp(interp_times, g["Time"], g["Release"])
        auc_vals.append(np.trapz(interp_release, interp_times))
    y_auc = np.array(auc_vals)  # (321,)

    # ── Task 4 — RNN arrays ───────────────────────────────────────────────────
    groups_nn = interp_df.groupby("Formulation Index")["Interpolated Release"]
    y_nn = np.stack(
        [g.to_numpy().reshape(-1, 1) for _, g in groups_nn]
    )  # (321, 11, 1)

    # ── Task 4 — XGB arrays ───────────────────────────────────────────────────
    repeated_df  = form_df.loc[form_df.index.repeat(NUM_INTERP_PTS)].reset_index(drop=True)
    X_xgb_notime = repeated_df.drop(columns=["Formulation Index"]).to_numpy()  # (3531, 11)

    repeated_df["Normalized Time"] = interp_df["Normalized Time"].values
    X_xgb_time = repeated_df.drop(columns=["Formulation Index"]).to_numpy()    # (3531, 12)

    y_xgb      = interp_df["Interpolated Release"].to_numpy()                  # (3531,)
    groups_xgb = np.repeat(np.arange(len(X) ), NUM_INTERP_PTS)                # (3531,)

    return (
        X, X_xgb_notime, X_xgb_time,
        y_timepoints, y_auc, y_class,
        y_nn, y_xgb,
        groups_xgb, aucs,
        drug_groups, drug_groups_xgb,
        drug_id_to_smiles, feature_names,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — CV SPLITTING & FOLD ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_splits(drug_groups: np.ndarray):
    """
    Build outer and inner CV splits using drug-identity-based GroupKFold.

    All formulations of the same drug stay together so no drug appears in
    both train and test in the same fold. All tasks share these splits,
    making per-fold metrics genuinely paired for Wilcoxon comparisons.

    Returns
    -------
    outer_splits           : list of (train_val_idx, test_idx)
    inner_splits_per_outer : list of lists of (train_idx, val_idx)
    """
    n        = len(drug_groups)
    X_dummy  = np.zeros((n, 1))
    y_dummy  = np.zeros(n)
    outer_kf = GroupKFold(n_splits=N_OUTER_FOLDS)
    inner_kf = GroupKFold(n_splits=N_INNER_FOLDS)

    outer_splits = list(outer_kf.split(X_dummy, y_dummy, drug_groups))

    inner_splits_per_outer = []
    for tv_idx, _ in outer_splits:
        inner_groups = drug_groups[tv_idx]
        X_inner      = np.zeros((len(tv_idx), 1))
        inner_splits_per_outer.append(
            list(inner_kf.split(X_inner, inner_groups, inner_groups))
        )

    return outer_splits, inner_splits_per_outer


def verify_splits(outer_splits: list, drug_groups: np.ndarray) -> pd.DataFrame:
    """
    Assert no drug leakage across folds and that all formulations are covered.
    Prints and returns a fold composition summary DataFrame.
    """
    all_test = []
    for fold_i, (tv_idx, test_idx) in enumerate(outer_splits, start=1):
        overlap = set(drug_groups[tv_idx]) & set(drug_groups[test_idx])
        assert len(overlap) == 0, \
            f"Fold {fold_i}: drug leakage — group IDs {overlap} in both train and test"
        all_test.extend(test_idx.tolist())

    assert sorted(all_test) == list(range(len(drug_groups))), \
        "Splits do not cover all formulations exactly once"

    print(f"[verify_splits] OK — {N_OUTER_FOLDS} outer folds, "
          f"{len(drug_groups)} formulations, no drug leakage.")

    rows = []
    for fold_i, (tv_idx, test_idx) in enumerate(outer_splits, start=1):
        rows.append({
            "Fold":        fold_i,
            "Train forms": len(tv_idx),
            "Test forms":  len(test_idx),
            "Train drugs": len(set(drug_groups[tv_idx])),
            "Test drugs":  len(set(drug_groups[test_idx])),
        })
    df = pd.DataFrame(rows).set_index("Fold")
    print(df.to_string())
    return df


def analyze_folds(
    outer_splits:  list,
    drug_groups:   np.ndarray,
    y_timepoints:  np.ndarray,
    y_class:       np.ndarray,
    y_norm_auc:    np.ndarray,
) -> dict:
    """
    Per-fold distribution analysis across all tasks.

    Returns dict with keys "task1", "task2", "task3" — each a pd.DataFrame.
    """
    n_total = len(y_class)
    n_burst = int(y_class.sum())
    n_sust  = n_total - n_burst

    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total formulations   : {n_total}")
    print(f"  Unique drugs         : {len(set(drug_groups))}")
    print(f"  Burst  (AUC > 0.5)   : {n_burst} ({100*n_burst/n_total:.1f}%)")
    print(f"  Sustained (AUC ≤ 0.5): {n_sust}  ({100*n_sust/n_total:.1f}%)")
    print(f"  Mean AUC             : {y_norm_auc.mean():.3f} ± {y_norm_auc.std():.3f}")
    print()

    rows_t1, rows_t2, rows_t3 = [], [], []

    for fold_i, (tv_idx, test_idx) in enumerate(outer_splits):
        fold       = fold_i + 1
        n_train    = len(tv_idx)
        n_test     = len(test_idx)
        n_tr_drugs = len(set(drug_groups[tv_idx]))
        n_te_drugs = len(set(drug_groups[test_idx]))
        pct_test   = round(100 * n_test / n_total, 1)

        tp = y_timepoints[test_idx]
        rows_t1.append({
            "fold": fold, "train_forms": n_train, "test_forms": n_test,
            "test_%": pct_test, "train_drugs": n_tr_drugs, "test_drugs": n_te_drugs,
            "24h_mean": round(tp[:, 0].mean(), 3), "24h_std": round(tp[:, 0].std(), 3),
            "48h_mean": round(tp[:, 1].mean(), 3), "48h_std": round(tp[:, 1].std(), 3),
            "72h_mean": round(tp[:, 2].mean(), 3), "72h_std": round(tp[:, 2].std(), 3),
        })

        cls = y_class[test_idx]
        nb  = int(cls.sum())
        ns  = len(cls) - nb
        rows_t2.append({
            "fold": fold, "train_forms": n_train, "test_forms": n_test,
            "test_%": pct_test, "train_drugs": n_tr_drugs, "test_drugs": n_te_drugs,
            "burst_n": nb,    "burst_%":     round(100 * nb / len(cls), 1),
            "sustained_n": ns, "sustained_%": round(100 * ns / len(cls), 1),
        })

        aucs_fold = y_norm_auc[test_idx]
        rows_t3.append({
            "fold": fold, "train_forms": n_train, "test_forms": n_test,
            "test_%": pct_test, "train_drugs": n_tr_drugs, "test_drugs": n_te_drugs,
            "AUC_mean": round(aucs_fold.mean(), 3), "AUC_std": round(aucs_fold.std(), 3),
            "AUC_leq_0.5": int((aucs_fold <= 0.5).sum()),
            "AUC_gt_0.5":  int((aucs_fold  > 0.5).sum()),
        })

    df_t1 = pd.DataFrame(rows_t1)
    df_t2 = pd.DataFrame(rows_t2)
    df_t3 = pd.DataFrame(rows_t3)

    for title, df in [
        ("TASK 1 — Fractional release at 24h / 48h / 72h", df_t1),
        ("TASK 2 / 3 — Burst vs sustained per fold",       df_t2),
        ("TASK 4 — Complete profile (AUC distribution)",   df_t3),
    ]:
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(df.to_string(index=False))
        print()

    return {"task1": df_t1, "task2": df_t2, "task3": df_t3}


def summarize_drug_distribution(
    drug_groups: np.ndarray,
    drug_id_to_smiles: dict,
) -> pd.DataFrame:
    """Summary of formulation counts per drug, sorted descending."""
    ids, counts = np.unique(drug_groups, return_counts=True)
    df = pd.DataFrame({
        "drug_id":        ids,
        "smiles":         [drug_id_to_smiles[i] for i in ids],
        "n_formulations": counts,
    }).sort_values("n_formulations", ascending=False).reset_index(drop=True)

    df["cumulative_%"] = (
        df["n_formulations"].cumsum() / df["n_formulations"].sum() * 100
    ).round(1)

    print("=" * 70)
    print("DRUG DISTRIBUTION")
    print("=" * 70)
    print(f"  Unique drugs           : {len(df)}")
    print(f"  Max formulations/drug  : {df['n_formulations'].max()}")
    print(f"  Min formulations/drug  : {df['n_formulations'].min()}")
    print(f"  Mean formulations/drug : {df['n_formulations'].mean():.1f}")
    print(f"  Drugs with 1 form      : {(df['n_formulations'] == 1).sum()}")
    print(f"  Drugs with >= 5 forms  : {(df['n_formulations'] >= 5).sum()}")
    print()
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — PYTORCH MODEL DEFINITIONS (Task 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CustomMLPWithEmbedding(nn.Module):
    """Two-layer MLP that maps formulation features to a dense embedding."""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate=0.2):
        super().__init__()
        self.fc1      = nn.Linear(input_dim,   hidden_dim1)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2      = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return x


def _repeat_embedding(emb, repeat_len=NUM_INTERP_PTS):
    """Expand (batch, emb_dim) → (batch, repeat_len, emb_dim)."""
    return emb.unsqueeze(1).repeat(1, repeat_len, 1)


class MLPGRU(nn.Module):
    """FC-NN-GRU: MLP embedding replicated across time, decoded by GRU."""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2,
                 gru_hidden_dim, gru_layers, output_dim, dropout_rate=0.2):
        super().__init__()
        self.embedding_mlp = CustomMLPWithEmbedding(
            input_dim, hidden_dim1, hidden_dim2, dropout_rate
        )
        self.gru = nn.GRU(
            input_size=hidden_dim2,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            dropout=dropout_rate if gru_layers > 1 else 0,
            batch_first=True,
        )
        self.fc1      = nn.Linear(gru_hidden_dim, 32)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2      = nn.Linear(32, output_dim)

    def forward(self, x):
        emb        = self.embedding_mlp(x)
        emb_seq    = _repeat_embedding(emb, NUM_INTERP_PTS)
        gru_out, _ = self.gru(emb_seq)
        out        = self.dropout1(self.relu1(self.fc1(gru_out)))
        return self.fc2(out)


class MLPLSTM(nn.Module):
    """FC-NN-LSTM: identical architecture to MLPGRU but with LSTM decoder."""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2,
                 lstm_hidden_dim, lstm_layers, output_dim, dropout_rate=0.2):
        super().__init__()
        self.embedding_mlp = CustomMLPWithEmbedding(
            input_dim, hidden_dim1, hidden_dim2, dropout_rate
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim2,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            batch_first=True,
        )
        self.fc1      = nn.Linear(lstm_hidden_dim, 32)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2      = nn.Linear(32, output_dim)

    def forward(self, x):
        emb          = self.embedding_mlp(x)
        emb_seq      = _repeat_embedding(emb, NUM_INTERP_PTS)
        lstm_out, _  = self.lstm(emb_seq)
        out          = self.dropout1(self.relu1(self.fc1(lstm_out)))
        return self.fc2(out)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — SKLEARN MODEL WRAPPERS (Tasks 1–3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Regression wrappers ───────────────────────────────────────────────────────

class XGBMultiModel:
    """XGBoost multi-output wrapper for Task 1 (timepoint prediction)."""
    def __init__(self, **params):
        base = xgb.XGBRegressor(**params, random_state=SEED, n_jobs=-1,
                                 objective="reg:squarederror")
        self.model = MultiOutputRegressor(base)

    def fit(self, X, y):       self.model.fit(X, y)
    def predict(self, X):      return self.model.predict(X)
    def evaluate(self, X, y):
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


class XGBSingleModel:
    """XGBoost single-output wrapper for Task 2 (AUC prediction)."""
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params, random_state=SEED)

    def fit(self, X, y):       self.model.fit(X, y)
    def predict(self, X):      return self.model.predict(X)
    def evaluate(self, X, y):
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


class RFMultiModel:
    """Random Forest multi-output wrapper for Task 1."""
    def __init__(self, **params):
        base = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
        self.model = MultiOutputRegressor(base)

    def fit(self, X, y):       self.model.fit(X, y)
    def predict(self, X):      return self.model.predict(X)
    def evaluate(self, X, y):
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


class RFSingleModel:
    """Random Forest single-output wrapper for Task 2."""
    def __init__(self, **params):
        self.model = RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)

    def fit(self, X, y):       self.model.fit(X, y)
    def predict(self, X):      return self.model.predict(X)
    def evaluate(self, X, y):
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


class LinearMultiModel:
    """Linear Regression multi-output for Task 1. Scales features internally."""
    def __init__(self):
        self.model  = MultiOutputRegressor(LinearRegression())
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def evaluate(self, X, y):
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


class LinearSingleModel:
    """Linear Regression single-output for Task 2. Scales features internally."""
    def __init__(self):
        self.model  = LinearRegression()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def evaluate(self, X, y):
        return float(np.sqrt(mean_squared_error(y, self.predict(X))))


# ── Classification wrappers ───────────────────────────────────────────────────

class XGBClassifierModel:
    """XGBoost binary classifier — Task 3 (burst vs sustained)."""
    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params, random_state=SEED,
                                        eval_metric="logloss")

    def fit(self, X, y):              self.model.fit(X, y)
    def predict(self, X):             return self.model.predict(X)
    def predict_proba(self, X):       return self.model.predict_proba(X)[:, 1]
    def evaluate(self, X, y):
        return 1.0 - float(accuracy_score(y, self.predict(X)))


class RFClassifierModel:
    """Random Forest binary classifier — Task 3."""
    def __init__(self, **params):
        self.model = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)

    def fit(self, X, y):              self.model.fit(X, y)
    def predict(self, X):             return self.model.predict(X)
    def predict_proba(self, X):       return self.model.predict_proba(X)[:, 1]
    def evaluate(self, X, y):
        return 1.0 - float(accuracy_score(y, self.predict(X)))


class LogisticModel:
    """Logistic Regression binary classifier — Task 3.
    Scaling handled externally in the runner (not inside this wrapper)."""
    def __init__(self, **params):
        self.model = LogisticRegression(**params, solver="liblinear",
                                         random_state=SEED)

    def fit(self, X, y):        self.model.fit(X, y)
    def predict(self, X):       return self.model.predict(X)
    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.squeeze()
    def evaluate(self, X, y):
        return 1.0 - float(accuracy_score(y, self.predict(X)))


# ── Task 4 XGBoost wrapper ────────────────────────────────────────────────────

class XGBoostModel:
    """Thin XGBRegressor wrapper for Task 4 profile prediction."""
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(
            **params, tree_method="hist", random_state=SEED, enable_categorical=True
        )

    def fit(self, X, y):       self.model.fit(X, y)
    def predict(self, X):      return self.model.predict(X)
    def evaluate(self, X, y):
        return float(np.sqrt(np.mean((self.predict(X) - y) ** 2)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 6 — INTERNAL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _adj_r2(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def _fold_rmse(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ── Empty result dicts ────────────────────────────────────────────────────────

def _empty_results_tp():
    return dict(preds=[], targets=[], rmse=[], mse=[], corr=[], pval=[],
                rmse_per_tp={t: [] for t in TIMEPOINTS},
                corr_per_tp={t: [] for t in TIMEPOINTS},
                pval_per_tp={t: [] for t in TIMEPOINTS},
                adj_r2_per_tp={t: [] for t in TIMEPOINTS},
                shap_values=[], shap_X_test=[])


def _empty_results_auc():
    return dict(preds=[], targets=[], rmse=[], mse=[], r2=[], adj_r2=[],
                corr=[], pval=[], shap_values=[], shap_X_test=[])


def _empty_results_class():
    return dict(preds=[], proba=[], targets=[], accuracy=[], roc_auc=[],
                precision=[], recall=[], specificity=[], f1=[],
                shap_values=[], shap_X_test=[])


def _empty_results_profile():
    return dict(preds=[], targets=[], mse=[], rmse=[], r2=[], adj_r2=[],
                rmse_low=[], rmse_high=[], n_low=[], n_high=[],
                train_loss_curves=[])


# ── Optuna search spaces ──────────────────────────────────────────────────────

def _xgb_search_space(trial):
    return {
        "max_depth":        trial.suggest_int(  "max_depth",        3,    20),
        "learning_rate":    trial.suggest_float("learning_rate",    0.01, 0.3),
        "n_estimators":     trial.suggest_int(  "n_estimators",     50,   300),
        "subsample":        trial.suggest_float("subsample",        0.5,  1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,  1.0),
        "gamma":            trial.suggest_float("gamma",            0,    5),
        "reg_alpha":        trial.suggest_float("reg_alpha",        0,    1),
        "reg_lambda":       trial.suggest_float("reg_lambda",       0,    1),
    }


def _rf_search_space(trial):
    return {
        "n_estimators":      trial.suggest_int(        "n_estimators",      50,  300),
        "max_depth":         trial.suggest_int(        "max_depth",          3,   20),
        "min_samples_split": trial.suggest_int(        "min_samples_split",  2,   20),
        "min_samples_leaf":  trial.suggest_int(        "min_samples_leaf",   1,   10),
        "max_features":      trial.suggest_categorical("max_features",  ["sqrt", "log2", None]),
        "bootstrap":         trial.suggest_categorical("bootstrap",     [True, False]),
    }


def _lr_search_space(trial):
    return {
        "C":            trial.suggest_float(      "C",            1e-4, 1e4, log=True),
        "penalty":      trial.suggest_categorical("penalty",      ["l1", "l2"]),
        "max_iter":     trial.suggest_int(        "max_iter",     200,  1000),
        "tol":          trial.suggest_float(      "tol",          1e-6, 1e-3, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def _compute_shap_multi(model, X_test):
    """SHAP for MultiOutputRegressor — one array per output."""
    return [shap.TreeExplainer(est).shap_values(X_test)
            for est in model.model.estimators_]


def _compute_shap_single(model, X_test):
    return shap.TreeExplainer(model.model).shap_values(X_test)


# ── AUC bucket RMSE (Task 4) ──────────────────────────────────────────────────

def _auc_bucket_rmse(y_pred, y_true, test_form_idx, aucs, threshold):
    """Split test predictions into low/high AUC buckets and compute RMSE."""
    if aucs is None:
        return np.nan, np.nan, 0, 0

    fold_aucs  = aucs[test_form_idx]
    auc_labels = np.repeat(fold_aucs, NUM_INTERP_PTS)
    low_mask   = auc_labels <= threshold
    high_mask  = ~low_mask

    return (
        _fold_rmse(y_true[low_mask],  y_pred[low_mask]),
        _fold_rmse(y_true[high_mask], y_pred[high_mask]),
        int(low_mask.sum()),
        int(high_mask.sum()),
    )


# ── RNN Optuna helpers ────────────────────────────────────────────────────────

def _rnn_kwarg(ModelClass):
    return "gru_hidden_dim" if ModelClass is MLPGRU else "lstm_hidden_dim"


def _rnn_layers_kwarg(ModelClass):
    return "gru_layers" if ModelClass is MLPGRU else "lstm_layers"


def _optuna_objective_rnn(trial, X_tv, y_tv, inner_splits, ModelClass):
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers  = trial.suggest_categorical("num_layers",  [1, 2, 3])
    dropout     = trial.suggest_categorical("dropout",     [0.2, 0.3, 0.4])
    lr          = trial.suggest_categorical("lr",          [1e-3, 1e-4])
    batch_size  = trial.suggest_categorical("batch_size",  [16, 32, 64, 128])

    criterion    = nn.MSELoss()
    inner_scores = []

    for train_idx, val_idx in inner_splits:
        scaler = StandardScaler()
        Xtr = torch.tensor(scaler.fit_transform(X_tv[train_idx]),
                           dtype=torch.float32).to(device)
        Xva = torch.tensor(scaler.transform(X_tv[val_idx]),
                           dtype=torch.float32).to(device)
        ytr = torch.tensor(y_tv[train_idx], dtype=torch.float32).to(device)
        yva = torch.tensor(y_tv[val_idx],   dtype=torch.float32).to(device)

        model = ModelClass(
            input_dim=NUM_FEATURES, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2,
            **{_rnn_kwarg(ModelClass): hidden_size},
            **{_rnn_layers_kwarg(ModelClass): num_layers},
            output_dim=1, dropout_rate=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader    = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size)

        for _ in range(INNER_EPOCHS):
            model.train()
            for xb, yb in loader:
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss   = 0.0
        val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size)
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item()
        inner_scores.append(val_loss / len(val_loader))

    return float(np.mean(inner_scores))


# ── Undersampling helper (balanced classification) ────────────────────────────

def _random_undersample(X, y, groups=None, seed=SEED):
    """
    Randomly drop samples from the majority class to achieve balance.
    Applied ONLY to training splits — never to test/validation sets.
    """
    rng     = np.random.default_rng(seed)
    idx_maj = np.where(y == 1)[0]
    idx_min = np.where(y == 0)[0]

    if len(idx_maj) < len(idx_min):
        idx_maj, idx_min = idx_min, idx_maj

    n_min = len(idx_min)
    n_maj = len(idx_maj)

    if n_maj == n_min:
        return X, y, groups

    chosen_maj = rng.choice(idx_maj, size=n_min, replace=False)
    keep       = np.sort(np.concatenate([idx_min, chosen_maj]))

    print(f"    [undersample] {n_maj} → {n_min} majority samples "
          f"| total {len(keep)} (was {len(y)})")

    return X[keep], y[keep], (groups[keep] if groups is not None else None)


# ── Summary printers ──────────────────────────────────────────────────────────

def _print_summary_tp(label, res):
    print(f"\n[{label}] Overall RMSE : "
          f"{np.mean(res['rmse']):.4f} ± {np.std(res['rmse']):.4f}")
    print(f"{'Time':<8} {'RMSE':^20} {'Corr':^20} {'AdjR²':^20}")
    print("-" * 68)
    for t in TIMEPOINTS:
        r  = np.mean(res["rmse_per_tp"][t])
        rs = np.std(res["rmse_per_tp"][t])
        c  = np.mean(res["corr_per_tp"][t])
        a  = np.mean(res["adj_r2_per_tp"][t])
        print(f"{t}d{'':<6} {r:.4f} ± {rs:.4f}   {c:.4f}{'':>10} {a:.4f}")


def _print_summary_auc(label, res):
    print(f"\n[{label}] RMSE   : {np.mean(res['rmse']):.4f} ± {np.std(res['rmse']):.4f}")
    print(f"[{label}] Adj R² : {np.mean(res['adj_r2']):.4f} ± {np.std(res['adj_r2']):.4f}")
    print(f"[{label}] Corr   : {np.mean(res['corr']):.4f} ± {np.std(res['corr']):.4f}")


def _print_summary_class(label, res):
    print(f"\n[{label}] Accuracy    : {np.mean(res['accuracy']):.4f} ± {np.std(res['accuracy']):.4f}")
    print(f"[{label}] ROC-AUC     : {np.mean(res['roc_auc']):.4f} ± {np.std(res['roc_auc']):.4f}")
    print(f"[{label}] F1          : {np.mean(res['f1']):.4f} ± {np.std(res['f1']):.4f}")
    print(f"[{label}] Precision   : {np.mean(res['precision']):.4f} ± {np.std(res['precision']):.4f}")
    print(f"[{label}] Recall      : {np.mean(res['recall']):.4f} ± {np.std(res['recall']):.4f}")
    print(f"[{label}] Specificity : {np.mean(res['specificity']):.4f} ± {np.std(res['specificity']):.4f}")


def _print_summary_profile(label, res):
    rmse_arr   = np.array(res["rmse"])
    adj_r2_arr = np.array(res["adj_r2"])
    print(f"\n[{label}] RMSE   : {rmse_arr.mean():.4f} ± {rmse_arr.std():.4f}")
    print(f"[{label}] Adj R² : {adj_r2_arr.mean():.4f} ± {adj_r2_arr.std():.4f}")
    if not np.all(np.isnan(res["rmse_low"])):
        print(f"[{label}] RMSE low  : {np.nanmean(res['rmse_low']):.4f}  (AUC ≤ {AUC_THRESHOLD})")
        print(f"[{label}] RMSE high : {np.nanmean(res['rmse_high']):.4f}  (AUC >  {AUC_THRESHOLD})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 7 — GENERIC NESTED-CV RUNNERS (Tasks 1, 2, 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_tp_model(X, y, outer_splits, inner_splits_per_outer,
                  ModelClass, search_space_fn, label, compute_shap=True):
    """Nested-CV runner for Task 1 (multi-output timepoint regression)."""
    res    = _empty_results_tp()
    n_feat = X.shape[1]

    for outer_fold, (tv_idx, test_idx) in enumerate(outer_splits):
        print(f"\n===== {label} | OUTER FOLD {outer_fold + 1}/{N_OUTER_FOLDS} =====")
        X_tv, y_tv     = X[tv_idx],   y[tv_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        inner_splits   = inner_splits_per_outer[outer_fold]

        if search_space_fn is not None:
            trial_bar = tqdm(total=N_TRIALS, desc="  Optuna trials",
                             unit="trial", leave=False)

            def objective(trial, _X=X_tv, _y=y_tv, _inner=inner_splits,
                          _MC=ModelClass, _ss=search_space_fn):
                params = _ss(trial)
                losses = []
                for tr_idx, va_idx in _inner:
                    m = _MC(**params)
                    m.fit(_X[tr_idx], _y[tr_idx])
                    losses.append(m.evaluate(_X[va_idx], _y[va_idx]))
                return float(np.mean(losses))

            def trial_callback(study, trial):
                trial_bar.update(1)
                trial_bar.set_postfix(best=f"{study.best_value:.4f}")

            sampler = TPESampler(seed=SEED)
            study   = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=N_TRIALS, callbacks=[trial_callback])
            trial_bar.close()
            bp = study.best_params
            print(f"  Best params: {bp}")
            best_model = ModelClass(**bp)
        else:
            best_model = ModelClass()

        best_model.fit(X_tv, y_tv)
        y_pred = best_model.predict(X_test)

        mse        = float(mean_squared_error(y_test, y_pred))
        rmse       = float(np.sqrt(mse))
        corr, pval = pearsonr(y_test.flatten(), y_pred.flatten())
        n          = len(y_test)

        for i, t in enumerate(TIMEPOINTS):
            rmse_t       = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
            r2_t         = r2_score(y_test[:, i], y_pred[:, i])
            ar2_t        = _adj_r2(r2_t, n, n_feat)
            corr_t, pval_t = pearsonr(y_test[:, i], y_pred[:, i])
            res["rmse_per_tp"][t].append(rmse_t)
            res["corr_per_tp"][t].append(corr_t)
            res["pval_per_tp"][t].append(pval_t)
            res["adj_r2_per_tp"][t].append(ar2_t)
            print(f"  {t}d: RMSE={rmse_t:.4f}  Corr={corr_t:.4f}  AdjR²={ar2_t:.4f}")

        if compute_shap and search_space_fn is not None:
            np.random.seed(SEED)
            res["shap_values"].append(_compute_shap_multi(best_model, X_test))
            res["shap_X_test"].append(X_test)
        else:
            res["shap_values"].append(None)
            res["shap_X_test"].append(X_test)

        res["preds"].append(y_pred);  res["targets"].append(y_test)
        res["rmse"].append(rmse);     res["mse"].append(mse)
        res["corr"].append(float(corr)); res["pval"].append(float(pval))
        print(f"  Overall — RMSE={rmse:.4f}  Corr={corr:.4f}  p={pval:.2e}")

    _print_summary_tp(label, res)
    return res


def _run_auc_model(X, y, outer_splits, inner_splits_per_outer,
                   ModelClass, search_space_fn, label, compute_shap=True):
    """Nested-CV runner for Task 2 (single-output AUC regression)."""
    res    = _empty_results_auc()
    n_feat = X.shape[1]

    for outer_fold, (tv_idx, test_idx) in enumerate(outer_splits):
        print(f"\n===== {label} | OUTER FOLD {outer_fold + 1}/{N_OUTER_FOLDS} =====")
        X_tv, y_tv     = X[tv_idx],   y[tv_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        inner_splits   = inner_splits_per_outer[outer_fold]

        if search_space_fn is not None:
            trial_bar = tqdm(total=N_TRIALS, desc="  Optuna trials",
                             unit="trial", leave=False)

            def objective(trial, _X=X_tv, _y=y_tv, _inner=inner_splits,
                          _MC=ModelClass, _ss=search_space_fn):
                params = _ss(trial)
                losses = []
                for tr_idx, va_idx in _inner:
                    m = _MC(**params)
                    m.fit(_X[tr_idx], _y[tr_idx])
                    losses.append(m.evaluate(_X[va_idx], _y[va_idx]))
                return float(np.mean(losses))

            def trial_callback(study, trial):
                trial_bar.update(1)
                trial_bar.set_postfix(best=f"{study.best_value:.4f}")

            sampler = TPESampler(seed=SEED)
            study   = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(objective, n_trials=N_TRIALS, callbacks=[trial_callback])
            trial_bar.close()
            bp = study.best_params
            print(f"  Best params: {bp}")
            best_model = ModelClass(**bp)
        else:
            best_model = ModelClass()

        best_model.fit(X_tv, y_tv)
        y_pred     = best_model.predict(X_test)
        mse        = float(mean_squared_error(y_test, y_pred))
        rmse       = float(np.sqrt(mse))
        r2         = r2_score(y_test.flatten(), y_pred.flatten())
        ar2        = _adj_r2(r2, len(y_test), n_feat)
        corr, pval = pearsonr(y_test.flatten(), y_pred.flatten())

        if compute_shap and search_space_fn is not None:
            np.random.seed(SEED)
            res["shap_values"].append(_compute_shap_single(best_model, X_test))
            res["shap_X_test"].append(X_test)
        else:
            res["shap_values"].append(None)
            res["shap_X_test"].append(X_test)

        res["preds"].append(y_pred);  res["targets"].append(y_test)
        res["rmse"].append(rmse);     res["mse"].append(mse)
        res["r2"].append(r2);         res["adj_r2"].append(ar2)
        res["corr"].append(float(corr)); res["pval"].append(float(pval))
        print(f"  RMSE={rmse:.4f}  R²={r2:.4f}  AdjR²={ar2:.4f}  "
              f"Corr={corr:.4f}  p={pval:.2e}")

    _print_summary_auc(label, res)
    return res


def _run_rnn_model(X, y, outer_splits, inner_splits_per_outer,
                   ModelClass, label, aucs=None, threshold=AUC_THRESHOLD):
    """Nested-CV runner for GRU and LSTM profile models (Task 4)."""
    res       = _empty_results_profile()
    criterion = nn.MSELoss()

    for outer_fold, (tv_idx, test_idx) in enumerate(outer_splits):
        print(f"\n===== {label} | OUTER FOLD {outer_fold + 1} =====")
        X_tv, y_tv     = X[tv_idx],   y[tv_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        inner_splits   = inner_splits_per_outer[outer_fold]

        sampler = TPESampler(seed=SEED)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            lambda trial: _optuna_objective_rnn(
                trial, X_tv, y_tv, inner_splits, ModelClass
            ),
            n_trials=N_TRIALS,
        )
        bp = study.best_params
        print("  Best params:", bp)

        scaler = StandardScaler()
        Xtr_s  = torch.tensor(scaler.fit_transform(X_tv),  dtype=torch.float32).to(device)
        Xte_s  = torch.tensor(scaler.transform(X_test),    dtype=torch.float32).to(device)
        ytr_t  = torch.tensor(y_tv,   dtype=torch.float32).to(device)
        yte_t  = torch.tensor(y_test, dtype=torch.float32).to(device)

        model = ModelClass(
            input_dim=NUM_FEATURES, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2,
            **{_rnn_kwarg(ModelClass):        bp["hidden_size"]},
            **{_rnn_layers_kwarg(ModelClass): bp["num_layers"]},
            output_dim=1, dropout_rate=bp["dropout"],
        ).to(device)

        optimizer    = torch.optim.Adam(model.parameters(), lr=bp["lr"])
        train_loader = DataLoader(TensorDataset(Xtr_s, ytr_t),
                                  batch_size=bp["batch_size"])
        loss_curve   = []

        for _ in range(OUTER_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                out  = model(xb)
                loss = criterion(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss_curve.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            y_pred_np = model(Xte_s).cpu().numpy().squeeze().flatten()
            y_test_np = yte_t.cpu().numpy().squeeze().flatten()

        mse  = float(np.mean((y_pred_np - y_test_np) ** 2))
        rmse = float(np.sqrt(mse))
        r2   = r2_score(y_test_np, y_pred_np)
        ar2  = _adj_r2(r2, len(y_test_np), X.shape[1])
        rl, rh, nl, nh = _auc_bucket_rmse(y_pred_np, y_test_np, test_idx, aucs, threshold)

        res["preds"].append(y_pred_np);  res["targets"].append(y_test_np)
        res["mse"].append(mse);          res["rmse"].append(rmse)
        res["r2"].append(r2);            res["adj_r2"].append(ar2)
        res["rmse_low"].append(rl);      res["rmse_high"].append(rh)
        res["n_low"].append(nl);         res["n_high"].append(nh)
        res["train_loss_curves"].append(loss_curve)
        print(f"  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  AdjR²={ar2:.4f}")
        if aucs is not None:
            print(f"  RMSE low  (AUC≤{threshold}) = {rl:.4f}  (n_timepoints={nl})")
            print(f"  RMSE high (AUC>{threshold})  = {rh:.4f}  (n_timepoints={nh})")

    _print_summary_profile(label, res)
    return res


def _run_xgb_profile_model(X, y, groups_xgb, drug_groups_xgb,
                            outer_splits_form, label,
                            aucs=None, threshold=AUC_THRESHOLD):
    """Nested-CV runner for XGBoost profile models (Task 4)."""
    res      = _empty_results_profile()
    res["train_loss_curves"] = None
    inner_kf = GroupKFold(n_splits=N_INNER_FOLDS)

    for outer_fold, (tv_form_idx, test_form_idx) in enumerate(outer_splits_form):
        print(f"\n===== {label} | OUTER FOLD {outer_fold + 1} =====")
        tv_mask   = np.isin(groups_xgb, tv_form_idx)
        test_mask = np.isin(groups_xgb, test_form_idx)
        X_tv, y_tv     = X[tv_mask],   y[tv_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        groups_tv_form = groups_xgb[tv_mask]

        def objective(trial, _X=X_tv, _y=y_tv, _g=groups_tv_form):
            params = _xgb_search_space(trial)
            losses = []
            for tr_idx, va_idx in inner_kf.split(_X, _y, _g):
                m = XGBoostModel(**params)
                m.fit(_X[tr_idx], _y[tr_idx])
                losses.append(m.evaluate(_X[va_idx], _y[va_idx]))
            return float(np.mean(losses))

        sampler = TPESampler(seed=SEED)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        bp = study.best_params
        print("  Best params:", bp)

        best_model = XGBoostModel(**bp)
        best_model.fit(X_tv, y_tv)
        y_pred_np = best_model.predict(X_test)
        y_test_np = y_test

        mse  = float(mean_squared_error(y_test_np, y_pred_np))
        rmse = float(np.sqrt(mse))
        r2   = r2_score(y_test_np, y_pred_np)
        n, p = X_test.shape
        ar2  = _adj_r2(r2, n, p)
        rl, rh, nl, nh = _auc_bucket_rmse(y_pred_np, y_test_np,
                                           test_form_idx, aucs, threshold)

        res["preds"].append(y_pred_np);  res["targets"].append(y_test_np)
        res["mse"].append(mse);          res["rmse"].append(rmse)
        res["r2"].append(r2);            res["adj_r2"].append(ar2)
        res["rmse_low"].append(rl);      res["rmse_high"].append(rh)
        res["n_low"].append(nl);         res["n_high"].append(nh)
        print(f"  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  AdjR²={ar2:.4f}")
        if aucs is not None:
            print(f"  RMSE low  (AUC≤{threshold}) = {rl:.4f}  (n_timepoints={nl})")
            print(f"  RMSE high (AUC>{threshold})  = {rh:.4f}  (n_timepoints={nh})")

    _print_summary_profile(label, res)
    return res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 8 — CLASSIFICATION RUNNERS (Task 3, standard + balanced)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_class_model(X, y, outer_splits, inner_splits_per_outer,
                     ModelClass, search_space_fn, label,
                     compute_shap=True, drug_groups=None, balanced=False):
    """
    Generic nested-CV runner for Task 3 (binary classification).

    When balanced=True, random undersampling is applied to each outer and inner
    training split. drug_groups must be provided in that case.

    Scaling for Logistic Regression is handled explicitly here (not inside
    the model wrapper), matching the original notebook behaviour exactly.
    """
    res         = _empty_results_class()
    is_logistic = ModelClass is LogisticModel

    for outer_fold, (tv_idx, test_idx) in enumerate(outer_splits):
        print(f"\n===== {label} | OUTER FOLD {outer_fold + 1}/{N_OUTER_FOLDS} =====")
        X_tv_raw, y_tv_raw = X[tv_idx],   y[tv_idx]
        X_test,   y_test   = X[test_idx], y[test_idx]
        inner_splits       = inner_splits_per_outer[outer_fold]

        unique, counts = np.unique(y_test, return_counts=True)
        print(f"  Test  set class dist : {dict(zip(unique.tolist(), counts.tolist()))}")
        unique, counts = np.unique(y_tv_raw, return_counts=True)
        print(f"  Train set class dist : {dict(zip(unique.tolist(), counts.tolist()))}")

        # Undersample outer train-val if requested
        if balanced:
            tv_groups_raw = drug_groups[tv_idx]
            X_tv, y_tv, _ = _random_undersample(
                X_tv_raw, y_tv_raw, groups=tv_groups_raw, seed=SEED + outer_fold
            )
        else:
            X_tv, y_tv = X_tv_raw, y_tv_raw

        # ── Optuna inner search ───────────────────────────────────────────────
        trial_bar = tqdm(total=N_TRIALS, desc="  Optuna trials",
                         unit="trial", leave=False)

        def objective(trial, _Xtv=X_tv, _ytv=y_tv, _inner=inner_splits):
            params     = search_space_fn(trial)
            val_losses = []
            for tr_idx, va_idx in _inner:
                # Guard index bounds when balanced has shrunk the outer train-val
                tr_idx = tr_idx[tr_idx < len(_Xtv)]
                va_idx = va_idx[va_idx < len(_Xtv)]
                X_tr, X_va = _Xtv[tr_idx], _Xtv[va_idx]
                y_tr, y_va = _ytv[tr_idx], _ytv[va_idx]
                if balanced:
                    X_tr, y_tr, _ = _random_undersample(X_tr, y_tr,
                                                         seed=SEED + outer_fold)
                if is_logistic:
                    inner_scaler = StandardScaler()
                    X_tr = inner_scaler.fit_transform(X_tr)
                    X_va = inner_scaler.transform(X_va)
                m = ModelClass(**params)
                m.fit(X_tr, y_tr)
                val_losses.append(m.evaluate(X_va, y_va))
            return float(np.mean(val_losses))

        def trial_callback(study, trial):
            trial_bar.update(1)
            trial_bar.set_postfix(best=f"{1 - study.best_value:.4f} acc")

        sampler = TPESampler(seed=SEED)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[trial_callback])
        trial_bar.close()
        bp = study.best_params
        print(f"  Best params: {bp}")

        # ── Final fit on (possibly balanced) outer train-val ─────────────────
        if is_logistic:
            outer_scaler = StandardScaler()
            X_tv_fit     = outer_scaler.fit_transform(X_tv)
            X_test_fit   = outer_scaler.transform(X_test)
        else:
            X_tv_fit, X_test_fit = X_tv, X_test

        best_model = ModelClass(**bp)
        best_model.fit(X_tv_fit, y_tv)
        preds = best_model.predict(X_test_fit)
        proba = best_model.predict_proba(X_test_fit)

        # ── Metrics (always on original unbalanced test set) ─────────────────
        acc  = float(accuracy_score(y_test, preds))
        roc  = float(roc_auc_score(y_test, proba))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec  = float(recall_score(y_test, preds, zero_division=0))
        f1   = float(f1_score(y_test, preds, zero_division=0))
        try:
            tn, fp, fn, tp_val = confusion_matrix(y_test, preds).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError:
            spec = 0.0

        # ── SHAP ─────────────────────────────────────────────────────────────
        if compute_shap:
            np.random.seed(SEED)
            if is_logistic:
                explainer = shap.LinearExplainer(best_model.model, X_tv_fit)
                shap_vals = explainer.shap_values(X_test_fit)
            else:
                explainer = shap.TreeExplainer(best_model.model)
                shap_vals = explainer.shap_values(X_test_fit)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                elif shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]
            res["shap_values"].append(shap_vals)
            res["shap_X_test"].append(X_test_fit)
        else:
            res["shap_values"].append(None)
            res["shap_X_test"].append(X_test_fit)

        res["preds"].append(preds);      res["proba"].append(proba)
        res["targets"].append(y_test);   res["accuracy"].append(acc)
        res["roc_auc"].append(roc);      res["precision"].append(prec)
        res["recall"].append(rec);       res["specificity"].append(float(spec))
        res["f1"].append(f1)

        print(f"  ACC={acc:.4f}  ROC-AUC={roc:.4f}  F1={f1:.4f}  "
              f"Prec={prec:.4f}  Rec={rec:.4f}  Spec={spec:.4f}")

    _print_summary_class(label, res)
    return res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 9 — PUBLIC RUNNER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Task 1: timepoint regression ──────────────────────────────────────────────

def run_tp_xgb(X, y_tp, outer_splits, inner_splits_per_outer):
    """XGBoost — timepoint prediction (24h / 48h / 72h)."""
    return _run_tp_model(X, y_tp, outer_splits, inner_splits_per_outer,
                         XGBMultiModel, _xgb_search_space, "XGB-Timepoint")


def run_tp_rf(X, y_tp, outer_splits, inner_splits_per_outer):
    """Random Forest — timepoint prediction."""
    return _run_tp_model(X, y_tp, outer_splits, inner_splits_per_outer,
                         RFMultiModel, _rf_search_space, "RF-Timepoint")


def run_tp_linear(X, y_tp, outer_splits, inner_splits_per_outer):
    """Linear Regression — timepoint prediction (no Optuna, no SHAP)."""
    return _run_tp_model(X, y_tp, outer_splits, inner_splits_per_outer,
                         LinearMultiModel, None, "Linear-Timepoint",
                         compute_shap=False)


# ── Task 2: AUC regression ────────────────────────────────────────────────────

def run_auc_xgb(X, y_auc, outer_splits, inner_splits_per_outer):
    """XGBoost — AUC prediction."""
    return _run_auc_model(X, y_auc, outer_splits, inner_splits_per_outer,
                          XGBSingleModel, _xgb_search_space, "XGB-AUC")


def run_auc_rf(X, y_auc, outer_splits, inner_splits_per_outer):
    """Random Forest — AUC prediction."""
    return _run_auc_model(X, y_auc, outer_splits, inner_splits_per_outer,
                          RFSingleModel, _rf_search_space, "RF-AUC")


def run_auc_linear(X, y_auc, outer_splits, inner_splits_per_outer):
    """Linear Regression — AUC prediction (no Optuna, no SHAP)."""
    return _run_auc_model(X, y_auc, outer_splits, inner_splits_per_outer,
                          LinearSingleModel, None, "Linear-AUC",
                          compute_shap=False)


# ── Task 3: classification (standard) ────────────────────────────────────────

def run_class_xgb(X, y_class, outer_splits, inner_splits_per_outer):
    """XGBoost — burst vs sustained classification."""
    return _run_class_model(X, y_class, outer_splits, inner_splits_per_outer,
                            XGBClassifierModel, _xgb_search_space, "XGB-Classification")


def run_class_rf(X, y_class, outer_splits, inner_splits_per_outer):
    """Random Forest — burst vs sustained classification."""
    return _run_class_model(X, y_class, outer_splits, inner_splits_per_outer,
                            RFClassifierModel, _rf_search_space, "RF-Classification")


def run_class_logistic(X, y_class, outer_splits, inner_splits_per_outer):
    """Logistic Regression — burst vs sustained classification."""
    return _run_class_model(X, y_class, outer_splits, inner_splits_per_outer,
                            LogisticModel, _lr_search_space, "Logistic-Classification")


# ── Task 3: classification (balanced via random undersampling) ────────────────

def run_class_xgb_balanced(X, y_class, outer_splits, inner_splits_per_outer,
                            drug_groups):
    """XGBoost — burst vs sustained, balanced via random undersampling."""
    return _run_class_model(X, y_class, outer_splits, inner_splits_per_outer,
                            XGBClassifierModel, _xgb_search_space,
                            "XGB-Classification-Balanced",
                            drug_groups=drug_groups, balanced=True)


def run_class_rf_balanced(X, y_class, outer_splits, inner_splits_per_outer,
                           drug_groups):
    """Random Forest — burst vs sustained, balanced via random undersampling."""
    return _run_class_model(X, y_class, outer_splits, inner_splits_per_outer,
                            RFClassifierModel, _rf_search_space,
                            "RF-Classification-Balanced",
                            drug_groups=drug_groups, balanced=True)


def run_class_logistic_balanced(X, y_class, outer_splits, inner_splits_per_outer,
                                 drug_groups):
    """Logistic Regression — burst vs sustained, balanced via random undersampling."""
    return _run_class_model(X, y_class, outer_splits, inner_splits_per_outer,
                            LogisticModel, _lr_search_space,
                            "Logistic-Classification-Balanced",
                            drug_groups=drug_groups, balanced=True)


# ── Task 4: complete profile prediction ──────────────────────────────────────

def run_gru(X_form, y_nn, outer_splits, inner_splits_per_outer, aucs=None):
    """Nested-CV for the FC-NN-GRU model."""
    return _run_rnn_model(X_form, y_nn, outer_splits, inner_splits_per_outer,
                          MLPGRU, "FC-NN-GRU", aucs=aucs)


def run_lstm(X_form, y_nn, outer_splits, inner_splits_per_outer, aucs=None):
    """Nested-CV for the FC-NN-LSTM model."""
    return _run_rnn_model(X_form, y_nn, outer_splits, inner_splits_per_outer,
                          MLPLSTM, "FC-NN-LSTM", aucs=aucs)


def run_xgb_time(X_xgb_time, y_xgb, groups_xgb, drug_groups_xgb,
                 outer_splits, inner_splits_per_outer, aucs=None):
    """Nested-CV for XGBoost with normalised time as a feature."""
    return _run_xgb_profile_model(X_xgb_time, y_xgb, groups_xgb, drug_groups_xgb,
                                   outer_splits, "XGB-Time", aucs=aucs)


def run_xgb_notime(X_xgb_notime, y_xgb, groups_xgb, drug_groups_xgb,
                   outer_splits, inner_splits_per_outer, aucs=None):
    """Nested-CV for XGBoost without the time feature."""
    return _run_xgb_profile_model(X_xgb_notime, y_xgb, groups_xgb, drug_groups_xgb,
                                   outer_splits, "XGB-NoTime", aucs=aucs)


def run_xgb_multicurve(X_form, y, outer_splits, inner_splits_per_outer,
                        aucs=None, threshold=AUC_THRESHOLD):
    """
    XGBoost MultiOutput — one row per formulation → 11 outputs (Task 4).
    y must be shape (321, 11) — flat, not (321, 11, 1).
    Matches xgb_multicurve.ipynb exactly.

    Result dict keys: preds, targets, rmse, mse, r2, adj_r2, corr, pval,
                      rmse_low, rmse_high, n_low, n_high,
                      shap_values, shap_X_test, train_loss_curves (None)
    """
    res = dict(
        preds=[], targets=[], rmse=[], mse=[], r2=[], adj_r2=[],
        corr=[], pval=[],
        rmse_low=[], rmse_high=[], n_low=[], n_high=[],
        shap_values=[], shap_X_test=[],
        train_loss_curves=None,
    )

    n_feat   = X_form.shape[1]
    inner_kf = GroupKFold(n_splits=N_INNER_FOLDS)

    for outer_fold, (tv_idx, test_idx) in enumerate(outer_splits):
        print(f"\n===== XGB-MultiCurve | OUTER FOLD {outer_fold + 1}/{N_OUTER_FOLDS} =====")
        X_tv, y_tv     = X_form[tv_idx],   y[tv_idx]
        X_test, y_test = X_form[test_idx], y[test_idx]
        inner_splits   = inner_splits_per_outer[outer_fold]

        # ── Optuna inner search ───────────────────────────────────────────────
        trial_bar = tqdm(total=N_TRIALS, desc="  Optuna trials",
                         unit="trial", leave=False)

        def objective(trial, _X=X_tv, _y=y_tv, _inner=inner_splits):
            params = _xgb_search_space(trial)
            losses = []
            for tr_idx, va_idx in _inner:
                m = XGBMultiModel(**params)
                m.fit(_X[tr_idx], _y[tr_idx])
                losses.append(m.evaluate(_X[va_idx], _y[va_idx]))
            return float(np.mean(losses))

        def trial_callback(study, trial):
            trial_bar.update(1)
            trial_bar.set_postfix(best=f"{study.best_value:.4f}")

        sampler = TPESampler(seed=SEED)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[trial_callback])
        trial_bar.close()
        bp = study.best_params
        print(f"  Best params: {bp}")

        # ── Train on full train-val ───────────────────────────────────────────
        best_model = XGBMultiModel(**bp)
        best_model.fit(X_tv, y_tv)
        y_pred = best_model.predict(X_test)   # (n_test, 11)

        # ── Metrics ───────────────────────────────────────────────────────────
        mse        = float(mean_squared_error(y_test, y_pred))
        rmse       = float(np.sqrt(mse))
        r2         = r2_score(y_test.flatten(), y_pred.flatten())
        n          = y_test.shape[0] * y_test.shape[1]   # total timepoints
        ar2        = 1 - (1 - r2) * (n - 1) / (n - n_feat - 1)
        corr, pval = pearsonr(y_test.flatten(), y_pred.flatten())

        # ── AUC-stratified RMSE ───────────────────────────────────────────────
        fold_aucs  = aucs[test_idx] if aucs is not None else None
        if fold_aucs is not None:
            auc_labels = np.repeat(fold_aucs, NUM_INTERP_PTS)
            low_mask   = auc_labels <= threshold
            high_mask  = ~low_mask
            rl = _fold_rmse(y_test.flatten()[low_mask],  y_pred.flatten()[low_mask])
            rh = _fold_rmse(y_test.flatten()[high_mask], y_pred.flatten()[high_mask])
            nl, nh = int(low_mask.sum()), int(high_mask.sum())
        else:
            rl, rh, nl, nh = np.nan, np.nan, 0, 0

        # ── SHAP (per output) ─────────────────────────────────────────────────
        print("  Computing SHAP values...")
        np.random.seed(SEED)
        shap_per_output = []
        for estimator in best_model.model.estimators_:
            explainer = shap.TreeExplainer(estimator)
            shap_per_output.append(explainer.shap_values(X_test))
        res["shap_values"].append(shap_per_output)
        res["shap_X_test"].append(X_test)

        # ── Store ─────────────────────────────────────────────────────────────
        res["preds"].append(y_pred);       res["targets"].append(y_test)
        res["mse"].append(mse);            res["rmse"].append(rmse)
        res["r2"].append(r2);              res["adj_r2"].append(ar2)
        res["corr"].append(float(corr));   res["pval"].append(float(pval))
        res["rmse_low"].append(rl);        res["rmse_high"].append(rh)
        res["n_low"].append(nl);           res["n_high"].append(nh)

        print(f"  RMSE={rmse:.4f}  R²={r2:.4f}  AdjR²={ar2:.4f}  Corr={corr:.4f}")
        if aucs is not None:
            print(f"  RMSE low  (AUC≤{threshold}) = {rl:.4f}  (n_timepoints={nl})")
            print(f"  RMSE high (AUC>{threshold})  = {rh:.4f}  (n_timepoints={nh})")

    _print_summary_profile("XGB-MultiCurve", res)
    return res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 10 — COMPARISON & STATISTICAL TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compare_tp_models(all_results: dict) -> pd.DataFrame:
    """Per-fold summary DataFrame for Task 1 models."""
    rows = []
    for name, res in all_results.items():
        for fold_i in range(len(res["rmse"])):
            row = dict(model=name, fold=fold_i + 1,
                       rmse=res["rmse"][fold_i],
                       corr=res["corr"][fold_i],
                       pval=res["pval"][fold_i])
            for t in TIMEPOINTS:
                row[f"rmse_{t}d"]   = res["rmse_per_tp"][t][fold_i]
                row[f"corr_{t}d"]   = res["corr_per_tp"][t][fold_i]
                row[f"adj_r2_{t}d"] = res["adj_r2_per_tp"][t][fold_i]
            rows.append(row)
    df = pd.DataFrame(rows)
    metric_cols = ["rmse"] + [f"rmse_{t}d" for t in TIMEPOINTS]
    print("\n=== Timepoint Model Comparison ===")
    print(df.groupby("model")[metric_cols].agg(["mean", "std"]).to_string())
    return df


def compare_auc_models(all_results: dict) -> pd.DataFrame:
    """Per-fold summary DataFrame for Task 2 models."""
    rows = []
    for name, res in all_results.items():
        for fold_i in range(len(res["rmse"])):
            rows.append(dict(model=name, fold=fold_i + 1,
                             rmse=res["rmse"][fold_i], r2=res["r2"][fold_i],
                             adj_r2=res["adj_r2"][fold_i],
                             corr=res["corr"][fold_i], pval=res["pval"][fold_i]))
    df = pd.DataFrame(rows)
    print("\n=== AUC Model Comparison ===")
    print(df.groupby("model")[["rmse", "r2", "adj_r2", "corr"]].agg(["mean", "std"]).to_string())
    return df


def compare_class_models(all_results: dict) -> pd.DataFrame:
    """Per-fold summary DataFrame for Task 3 classification models."""
    rows = []
    for name, res in all_results.items():
        for fold_i in range(len(res["accuracy"])):
            rows.append(dict(model=name, fold=fold_i + 1,
                             accuracy=res["accuracy"][fold_i],
                             roc_auc=res["roc_auc"][fold_i],
                             precision=res["precision"][fold_i],
                             recall=res["recall"][fold_i],
                             specificity=res["specificity"][fold_i],
                             f1=res["f1"][fold_i]))
    df = pd.DataFrame(rows)
    metric_cols = ["accuracy", "roc_auc", "precision", "recall", "specificity", "f1"]
    print("\n=== Classification Model Comparison ===")
    print(df.groupby("model")[metric_cols].agg(["mean", "std"]).to_string())
    return df


def compare_models(all_results: dict) -> pd.DataFrame:
    """Per-fold summary DataFrame for Task 4 profile models."""
    rows = []
    for name, res in all_results.items():
        for fold_i in range(len(res["rmse"])):
            rows.append(dict(model=name, fold=fold_i + 1,
                             mse=res["mse"][fold_i], rmse=res["rmse"][fold_i],
                             r2=res["r2"][fold_i],   adj_r2=res["adj_r2"][fold_i],
                             rmse_low=res["rmse_low"][fold_i],
                             n_low=res["n_low"][fold_i],
                             rmse_high=res["rmse_high"][fold_i],
                             n_high=res["n_high"][fold_i]))
    df = pd.DataFrame(rows)
    metric_cols = ["mse", "rmse", "r2", "adj_r2", "rmse_low", "rmse_high"]
    print("\n=== Profile Model Comparison ===")
    print(df.groupby("model")[metric_cols].agg(["mean", "std"]).to_string())
    return df


def wilcoxon_tests(all_results: dict, metric: str):
    """
    Pairwise Wilcoxon signed-rank tests on per-fold metric.
    Handles nested per-timepoint metrics (e.g. 'rmse_1d', 'corr_2d').
    Folds with NaN in either model are dropped silently.
    """
    names = list(all_results.keys())

    def get_scores(res, metric):
        if metric in res:
            return np.array(res[metric])
        for t in TIMEPOINTS:
            if metric == f"rmse_{t}d":   return np.array(res["rmse_per_tp"][t])
            if metric == f"corr_{t}d":   return np.array(res["corr_per_tp"][t])
            if metric == f"adj_r2_{t}d": return np.array(res["adj_r2_per_tp"][t])
        raise KeyError(f"Metric '{metric}' not found in results dict.")

    scores = {n: get_scores(all_results[n], metric) for n in names}

    print(f"\n=== Wilcoxon tests on {metric} ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            mask = ~(np.isnan(scores[a]) | np.isnan(scores[b]))
            if mask.sum() < 2:
                print(f"  {a} vs {b}: insufficient non-NaN folds")
                continue
            stat, p = _wilcoxon(scores[a][mask], scores[b][mask])
            sig = "**" if p < 0.05 else "ns"
            print(f"  {a} vs {b}: W={stat:.1f}, p={p:.4f}  {sig}")


def results_to_df(all_results: dict) -> pd.DataFrame:
    """Flatten predictions to a long DataFrame (model, fold, y_pred, y_true)."""
    rows = []
    for name, res in all_results.items():
        for fold_i, (preds, targets) in enumerate(
            zip(res["preds"], res["targets"]), start=1
        ):
            for p, t in zip(preds, targets):
                rows.append(dict(model=name, fold=fold_i, y_pred=p, y_true=t))
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 11 — CLI ENTRY POINT (fold distribution analysis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Drug-based CV split analysis for LAI drug release prediction."
    )
    parser.add_argument("--form",    required=True,
                        help="Path to formulations Excel file.")
    parser.add_argument("--release", required=True,
                        help="Path to release profiles Excel file.")
    parser.add_argument("--save",    default=None,
                        help="Optional directory to save fold distribution CSVs.")
    args = parser.parse_args()

    print("\nLoading data...")
    (X, _, _, y_timepoints, y_auc, y_class, _, _, _, aucs,
     drug_groups, _, drug_id_to_smiles, feature_names) = load_data(
        args.form, args.release
    )

    print(f"  X shape         : {X.shape}")
    print(f"  Features        : {feature_names}")
    print(f"  y_timepoints    : {y_timepoints.shape}")
    print(f"  y_class         : {y_class.shape}  "
          f"(burst={y_class.sum()}, sustained={len(y_class)-y_class.sum()})")
    print(f"  y_auc           : {y_auc.shape}\n")

    summarize_drug_distribution(drug_groups, drug_id_to_smiles)

    print("Building CV splits...")
    outer_splits, inner_splits_per_outer = make_splits(drug_groups)
    verify_splits(outer_splits, drug_groups)
    print()

    fold_tables = analyze_folds(outer_splits, drug_groups,
                                y_timepoints, y_class, aucs)

    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        summarize_drug_distribution(drug_groups, drug_id_to_smiles).to_csv(
            save_dir / "drug_distribution.csv", index=False
        )
        fold_tables["task1"].to_csv(save_dir / "fold_dist_task1.csv", index=False)
        fold_tables["task2"].to_csv(save_dir / "fold_dist_task2.csv", index=False)
        fold_tables["task3"].to_csv(save_dir / "fold_dist_task3.csv", index=False)
        print(f"Tables saved to {save_dir}/")

    return outer_splits, inner_splits_per_outer, fold_tables


if __name__ == "__main__":
    main()
