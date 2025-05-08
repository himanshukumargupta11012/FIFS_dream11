import pandas as pd
import numpy as np
import os
import json
import joblib
import pickle
import lightgbm as lgb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Use ExperimentAnalysis for loading results post-run in newer Ray versions
try:
    from ray.tune import ExperimentAnalysis
except ImportError:
    from ray.tune.analysis import ExperimentAnalysis # Older versions
import ray
# --- Assuming these are imported ---
from feature_utils import (
    process,
    compute_overlap_robust,
    compute_overlap_true_test
    # Your MLP model definition
)
from model_utils import CustomMLPModel

# --- Configuration ---
# !!! IMPORTANT: UPDATE ALL PATHS AND SETTINGS BELOW !!!
MLP_RESULTS_DIR = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/ray_results/MLP_Hyperparam_Tuning_20250414-195621" # Dir for BEST MLP run
LGBM_RESULTS_DIR = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/ray_results/LGBM_Hyperparam_Tuning_20250415-040205" # Dir for BEST LGBM run

# Metrics used to select best models during tuning AND for weighting
MLP_METRIC = "val_overlap" # Metric used TO SELECT best MLP
MLP_MODE = "max"
LGBM_METRIC = "val_overlap"   # Metric used TO SELECT best LGBM
LGBM_MODE = "max"

# ** Metric for Ensemble Weighting ** Choose one:
# Option 1: Use validation overlap score (higher is better)
WEIGHTING_METRIC = "val_overlap"
# Option 2: Use validation MAE (lower is better)
# WEIGHTING_METRIC = "val_mae" # (or "val_loss" if MAE was reported as val_loss)

DATA_FILE_PATH = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/processed/combined/10_window_ipl.csv" # Path to the final feature CSV

# Target transformation setting (should match the best model configs)
# These might be overridden by loaded config below
USE_LOG1P_TARGET_MLP = True
USE_LOG1P_TARGET_LGBM = True

OUTPUT_DIR = "final_evaluation_results_weighted" # Directory for plots and metrics

# Set device for PyTorch (MLP)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not ray.is_initialized():
    print("Initializing Ray for analysis...")
    ray.init(ignore_reinit_error=True, log_to_driver=False) # Basic init for analysis

# --- Helper to Load Best Config & Metrics ---
def get_best_trial_data(exp_dir, metric, mode):
    print(f"Loading best trial data from: {exp_dir}")
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    analysis = ExperimentAnalysis(exp_dir)
    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="all")
    if not best_trial:
        raise ValueError(f"No best trial found for metric '{metric}' in {exp_dir}")
    best_config = best_trial.config
    best_metrics = best_trial.last_result if best_trial.last_result else {}
    best_epoch = best_metrics.get("epoch", None)
    best_iteration_lgbm = best_metrics.get("num_boost_round", None)
    print(f"Best Trial ID (inferred): {os.path.basename(best_trial.local_path)}")
    print(f"  Best Metric ({metric}): {best_metrics.get(metric, 'N/A'):.4f}")
    print(f"  Validation Overlap: {best_metrics.get('val_overlap', 'N/A'):.4f}")
    print(f"  Validation MAE: {best_metrics.get('val_mae', best_metrics.get('val_loss', 'N/A')):.4f}") # Handle if MAE reported as val_loss
    return best_config, best_metrics, best_epoch, best_iteration_lgbm

# --- Main Evaluation Logic ---
print("--- Final Model Evaluation: MLP vs LGBM vs Weighted Ensemble ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Best Configs and **Validation Metrics for Weighting**
print("\n--- Loading Best Configurations & Validation Metrics ---")
w_mlp, w_lgbm = 0.5, 0.5 # Default weights (simple average)
try:
    mlp_config, mlp_best_metrics, mlp_best_epoch, _ = get_best_trial_data(MLP_RESULTS_DIR, MLP_METRIC, MLP_MODE)
    lgbm_config, lgbm_best_metrics, _, lgbm_best_iteration = get_best_trial_data(LGBM_RESULTS_DIR, LGBM_METRIC, LGBM_MODE)

    # --- Calculate Ensemble Weights based on Validation Performance ---
    mlp_weight_metric_val = mlp_best_metrics.get(WEIGHTING_METRIC)
    lgbm_weight_metric_val = lgbm_best_metrics.get(WEIGHTING_METRIC)

    if mlp_weight_metric_val is not None and lgbm_weight_metric_val is not None:
        print(f"\nCalculating weights based on Validation Metric: {WEIGHTING_METRIC}")
        # Handle MAE weighting (lower is better)
        if "loss" in WEIGHTING_METRIC or "mae" in WEIGHTING_METRIC:
             # Use Inverse weighting for MAE/loss
             if mlp_weight_metric_val > 1e-6 and lgbm_weight_metric_val > 1e-6: # Avoid division by zero
                 inv_mlp = 1.0 / mlp_weight_metric_val
                 inv_lgbm = 1.0 / lgbm_weight_metric_val
                 total_inv = inv_mlp + inv_lgbm
                 w_mlp = inv_mlp / total_inv
                 w_lgbm = inv_lgbm / total_inv
                 print(f"Inverse MAE/Loss weighting: MLP={w_mlp:.3f}, LGBM={w_lgbm:.3f}")
             else: print("MAE/Loss is zero or near-zero, falling back to simple average.")
        # Handle Overlap weighting (higher is better)
        elif "overlap" in WEIGHTING_METRIC:
             if mlp_weight_metric_val >= 0 and lgbm_weight_metric_val >= 0:
                 total_metric = mlp_weight_metric_val + lgbm_weight_metric_val
                 if total_metric > 1e-6: # Avoid division by zero
                     w_mlp = mlp_weight_metric_val / total_metric
                     w_lgbm = lgbm_weight_metric_val / total_metric
                     print(f"  Overlap weighting: MLP={w_mlp:.3f}, LGBM={w_lgbm:.3f}")
                 else: print("  Overlap scores are zero, falling back to simple average.")
             else: print("  Overlap scores invalid (<0), falling back to simple average.")
        else: print(f"  Unknown weighting metric '{WEIGHTING_METRIC}', falling back to simple average.")
    else:
        print(f"  Weighting metric '{WEIGHTING_METRIC}' not found for one or both models, falling back to simple average.")

    # Final check on training iterations/epochs
    if mlp_best_epoch is None: mlp_best_epoch = 50; print(f"Warning: Using default {mlp_best_epoch} MLP epochs.")
    else: mlp_best_epoch = int(mlp_best_epoch) + 1
    if lgbm_best_iteration is None: raise ValueError("Best LGBM iteration not found.")
    else: lgbm_best_iteration = int(lgbm_best_iteration)

except Exception as e:
    print(f"Error loading configurations or weights: {e}")
    exit()

# Override log transform flags if present in loaded configs
USE_LOG1P_TARGET_MLP = mlp_config.get("use_log1p_target", USE_LOG1P_TARGET_MLP)
USE_LOG1P_TARGET_LGBM = lgbm_config.get("use_log1p_target", USE_LOG1P_TARGET_LGBM)


# 2. Load Data and Create FINAL Train/Test Splits (Same as before)
# ... (Keep the exact same data loading and splitting logic) ...
print("\n--- Loading and Splitting Data ---")
if not os.path.exists(DATA_FILE_PATH): raise FileNotFoundError(f"Data file not found: {DATA_FILE_PATH}")
combined_df = pd.read_csv(DATA_FILE_PATH); combined_df["date"] = pd.to_datetime(combined_df["date"])
start_date = pd.to_datetime("2008-04-18"); split_date = pd.to_datetime("2020-10-17"); end_date = pd.to_datetime("2023-04-04")
train_val_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)].copy().sort_values("date")
test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)].copy()
unique_dates = train_val_df["date"].unique()
if len(unique_dates) < 2: raise ValueError("Not enough unique dates to split.")
split_idx = int(0.8 * len(unique_dates)); split_idx = max(0, min(split_idx, len(unique_dates) - 2))
threshold_date = unique_dates[split_idx]
train_df = train_val_df[train_val_df["date"] <= threshold_date].copy()
val_df = train_val_df[train_val_df["date"] > threshold_date].copy()
train_final_df = pd.concat([train_df, val_df], ignore_index=True)
print(f"Final Train shape: {train_final_df.shape}, Test shape: {test_df.shape}")
if train_final_df.empty or test_df.empty: raise ValueError("Final Train or Test DataFrame empty.")


# 3. Prepare Data for Each Model (Same as before)
# ... (Keep the exact same data preparation logic for MLP and LGBM) ...
print("\n--- Preparing Data ---")
data_file_name = os.path.basename(DATA_FILE_PATH)
try: k = int(data_file_name.split("_")[0])
except Exception as e: raise ValueError(f"Cannot extract 'k' from data file name: {data_file_name}") from e
# MLP Data Prep
print("Preparing MLP data...")
X_train_mlp_raw, feature_names = process(train_df, k) # Fit scaler on original train
X_train_final_mlp_raw, _ = process(train_final_df, k)
X_test_mlp_raw, _ = process(test_df, k)
y_train_orig = train_df["fantasy_points"].values
y_train_final_orig = train_final_df["fantasy_points"].values
y_test_orig = test_df["fantasy_points"].values
scaler_X_mlp = StandardScaler().fit(X_train_mlp_raw)
if USE_LOG1P_TARGET_MLP:
    y_train_target_mlp_unscaled = np.log1p(np.maximum(0, y_train_orig))
    scaler_y_mlp = StandardScaler().fit(y_train_target_mlp_unscaled.reshape(-1, 1))
    y_train_final_target_mlp = scaler_y_mlp.transform(np.log1p(np.maximum(0, y_train_final_orig)).reshape(-1, 1)).flatten()
else:
    y_train_target_mlp_unscaled = y_train_orig
    scaler_y_mlp = StandardScaler().fit(y_train_target_mlp_unscaled.reshape(-1, 1))
    y_train_final_target_mlp = scaler_y_mlp.transform(y_train_final_orig.reshape(-1, 1)).flatten()
X_train_final_mlp = scaler_X_mlp.transform(X_train_final_mlp_raw)
X_test_mlp = scaler_X_mlp.transform(X_test_mlp_raw)
mlp_batch_size = mlp_config.get("batch_size", 1024)
train_final_dataset = TensorDataset(torch.tensor(X_train_final_mlp, dtype=torch.float32), torch.tensor(y_train_final_target_mlp, dtype=torch.float32).view(-1, 1))
train_final_loader = DataLoader(train_final_dataset, batch_size=mlp_batch_size, shuffle=True)
X_test_tensor = torch.tensor(X_test_mlp, dtype=torch.float32)
# LGBM Data Prep
print("Preparing LGBM data...")
X_train_final_lgbm, _ = process(train_final_df, k) # No scaling needed for features
X_test_lgbm, _ = process(test_df, k)
if USE_LOG1P_TARGET_LGBM: y_train_final_target_lgbm = np.log1p(np.maximum(0, y_train_final_orig))
else: y_train_final_target_lgbm = y_train_final_orig


# 4. Retrain Final MLP Model (Same as before)
# ... (Keep the exact same MLP retraining loop) ...
print("\n--- Retraining Final MLP Model ---")
num_input_features = X_train_final_mlp.shape[1]
mlp_model = CustomMLPModel(input_size=num_input_features,hidden_layers=mlp_config["hidden_layers"],activation=mlp_config["activation"],dropout_prob=mlp_config["dropout_prob"],weight_init=mlp_config["weight_init"]).to(DEVICE)
loss_fn_mlp = nn.MSELoss() # Or nn.L1Loss()
optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=mlp_config["lr"])
start_mlp_train = time.time()
mlp_model.train()
for epoch in range(mlp_best_epoch):
    for batch_x, batch_y in train_final_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE); optimizer_mlp.zero_grad(); predictions = mlp_model(batch_x); loss = loss_fn_mlp(predictions, batch_y); loss.backward(); optimizer_mlp.step()
    if (epoch + 1) % 20 == 0 or epoch == mlp_best_epoch - 1: print(f"MLP Retrain Epoch {epoch+1}/{mlp_best_epoch}") # Light progress
end_mlp_train = time.time(); print(f"MLP retraining took: {end_mlp_train - start_mlp_train:.2f}s")


# 5. Retrain Final LGBM Model (Same as before)
# ... (Keep the exact same LGBM retraining) ...
print("\n--- Retraining Final LGBM Model ---")
lgbm_final_params = {k:v for k,v in lgbm_config.items() if k not in ['data_file', 'use_log1p_target']}
lgbm_final_params['n_estimators'] = lgbm_best_iteration
lgbm_model = lgb.LGBMRegressor(**lgbm_final_params)
start_lgbm_train = time.time()
lgbm_model.fit(X_train_final_lgbm, y_train_final_target_lgbm)
end_lgbm_train = time.time(); print(f"LGBM retraining took: {end_lgbm_train - start_lgbm_train:.2f}s")


# 6. Predict on Test Set (Same as before)
# ... (Keep the exact same prediction logic for MLP and LGBM) ...
print("\n--- Predicting on Test Set ---")
mlp_model.eval();
with torch.no_grad(): y_pred_mlp_scaled = mlp_model(X_test_tensor.to(DEVICE)).cpu().numpy()
if USE_LOG1P_TARGET_MLP: y_pred_mlp_log = scaler_y_mlp.inverse_transform(y_pred_mlp_scaled).flatten(); y_pred_mlp_orig = np.maximum(0, np.expm1(y_pred_mlp_log))
else: y_pred_mlp_orig = np.maximum(0, scaler_y_mlp.inverse_transform(y_pred_mlp_scaled).flatten())
y_pred_lgbm_target = lgbm_model.predict(X_test_lgbm)
if USE_LOG1P_TARGET_LGBM: y_pred_lgbm_orig = np.maximum(0, np.expm1(y_pred_lgbm_target))
else: y_pred_lgbm_orig = np.maximum(0, y_pred_lgbm_target)


# 7. Create **WEIGHTED** Ensemble Prediction
print(f"Creating weighted ensemble prediction (MLP w={w_mlp:.3f}, LGBM w={w_lgbm:.3f})...")
y_pred_ensemble_orig = w_mlp * y_pred_mlp_orig + w_lgbm * y_pred_lgbm_orig # USE WEIGHTS


# Prepare for metrics
y_test_orig_int = y_test_orig.astype(int)
y_pred_mlp_int = y_pred_mlp_orig.astype(int)
y_pred_lgbm_int = y_pred_lgbm_orig.astype(int)
y_pred_ensemble_int = y_pred_ensemble_orig.astype(int)
test_match_ids = test_df["match_id"].values

# 8. Evaluate Models on Test Set (Add Weighted Ensemble)
print("\n--- Test Set Evaluation Results ---")
results = {}
results['MLP'] = {'Overlap (Robust)': compute_overlap_true_test(y_test_orig_int, y_pred_mlp_int, test_match_ids),'MAE': mean_absolute_error(y_test_orig, y_pred_mlp_orig),'MSE': mean_squared_error(y_test_orig, y_pred_mlp_orig)}
results['LGBM'] = {'Overlap (Robust)': compute_overlap_true_test(y_test_orig_int, y_pred_lgbm_int, test_match_ids),'MAE': mean_absolute_error(y_test_orig, y_pred_lgbm_orig),'MSE': mean_squared_error(y_test_orig, y_pred_lgbm_orig)}
# ---> ADDED WEIGHTED ENSEMBLE <---
results['Ensemble (Weighted)'] = {'Overlap (Robust)': compute_overlap_true_test(y_test_orig_int, y_pred_ensemble_int, test_match_ids),'MAE': mean_absolute_error(y_test_orig, y_pred_ensemble_orig),'MSE': mean_squared_error(y_test_orig, y_pred_ensemble_orig)}

print("\n--- Test Set Metrics ---")
for model_name, metrics_dict in results.items():
    print(f"Model: {model_name}")
    for metric_name, value in metrics_dict.items(): print(f"  {metric_name}: {value:.4f}")
    print("-" * 20)

# 9. Generate Final Plots (Add Weighted Ensemble Plot)
# ... (Keep MLP and LGBM Plots) ...
print("\n--- Generating Final Plots ---")
# MLP Plot
plt.figure(figsize=(8, 8)); plt.scatter(y_test_orig, y_pred_mlp_orig, alpha=0.5)
max_val = max(max(y_test_orig, default=0), max(y_pred_mlp_orig, default=0)); min_val = min(min(y_test_orig, default=0), min(y_pred_mlp_orig, default=0))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)'); plt.title(f'MLP Predicted vs. Actual FP (TEST Set)')
plt.xlabel('Actual Points'); plt.ylabel('Predicted Points'); plt.grid(True); plt.legend(); plt.axis('equal')
plt.savefig(os.path.join(OUTPUT_DIR, "test_pred_vs_actual_MLP.png")); plt.close(); print(f"Saved MLP Test Plot.")
# LGBM Plot
plt.figure(figsize=(8, 8)); plt.scatter(y_test_orig, y_pred_lgbm_orig, alpha=0.5)
max_val = max(max(y_test_orig, default=0), max(y_pred_lgbm_orig, default=0)); min_val = min(min(y_test_orig, default=0), min(y_pred_lgbm_orig, default=0))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)'); plt.title(f'LGBM Predicted vs. Actual FP (TEST Set)')
plt.xlabel('Actual Points'); plt.ylabel('Predicted Points'); plt.grid(True); plt.legend(); plt.axis('equal')
plt.savefig(os.path.join(OUTPUT_DIR, "test_pred_vs_actual_LGBM.png")); plt.close(); print(f"Saved LGBM Test Plot.")
# ---> ADDED WEIGHTED ENSEMBLE PLOT <---
plt.figure(figsize=(8, 8)); plt.scatter(y_test_orig, y_pred_ensemble_orig, alpha=0.5)
max_val = max(max(y_test_orig, default=0), max(y_pred_ensemble_orig, default=0)); min_val = min(min(y_test_orig, default=0), min(y_pred_ensemble_orig, default=0))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)'); plt.title(f'WEIGHTED Ensemble Predicted vs. Actual FP (TEST Set)')
plt.xlabel('Actual Points'); plt.ylabel('Predicted Points'); plt.grid(True); plt.legend(); plt.axis('equal')
plt.savefig(os.path.join(OUTPUT_DIR, "test_pred_vs_actual_WEIGHTED_ENSEMBLE.png")); plt.close(); print(f"Saved Weighted Ensemble Test Plot.")


print("\nShutting down Ray...")
ray.shutdown()
print("\n--- Final Evaluation Complete ---")