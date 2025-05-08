import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import pickle
import json
import joblib # For loading the model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Use ExperimentAnalysis for loading results post-run in newer Ray versions
# If Analysis API fails, try ExperimentAnalysis
try:
    # Newer Ray versions might have ExperimentAnalysis directly under tune
    from ray.tune import ExperimentAnalysis
except ImportError:
    # Older versions or different structure might have it here
    from ray.tune.analysis import ExperimentAnalysis


# --- Assuming these are imported ---
from feature_utils import (
    process,
    compute_overlap_robust,
    compute_overlap_true_test
    # compute_loss, # Only if needed
    # calculate_baseline_overlap # Optional
)

# --- Configuration ---
# IMPORTANT: Update this path
RAY_RESULTS_DIR = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/ray_results/LGBM_Hyperparam_Tuning_20250418-224729" # Experiment directory
METRIC_TO_OPTIMIZE = "val_loss"  # Metric used (e.g., val_loss which is MAE)
OPTIMIZATION_MODE = "min"         # Mode for the metric

ANALYSIS_OUTPUT_DIR = "analysis_lgbm_best" # Subdirectory to save plots

# --- Main Analysis Logic ---
print(f"--- Evaluating Best LightGBM Model from {RAY_RESULTS_DIR} ---")

if not os.path.isdir(RAY_RESULTS_DIR):
    raise FileNotFoundError(f"Ray results directory not found: {RAY_RESULTS_DIR}")

# Load the analysis object using ExperimentAnalysis
print(f"Loading experiment analysis from: {RAY_RESULTS_DIR}")
analysis = ExperimentAnalysis(RAY_RESULTS_DIR)

# Get the best trial/result based on the specified metric and mode
# Note: ExperimentAnalysis might return trial path or analysis dataframe depending on usage
# We often use .get_best_trial or .get_best_config methods
best_trial = analysis.get_best_trial(metric=METRIC_TO_OPTIMIZE, mode=OPTIMIZATION_MODE, scope="all") # Or "all"

if not best_trial:
    raise ValueError(f"No best trial found for metric '{METRIC_TO_OPTIMIZE}' with mode '{OPTIMIZATION_MODE}'. Check directory, metric name, and mode.")

# ExperimentAnalysis might store results differently, access necessary info
best_trial_logdir = best_trial.checkpoint.path
best_config = best_trial.config
# Metrics might be under last_result or a different attribute depending on Ray version
best_metrics = best_trial.last_result # Common location
if not best_metrics:
     raise ValueError(f"Could not retrieve metrics from the best trial result object: {best_trial}")

# Checkpoint loading needs the path. ExperimentAnalysis might give Trial object or path.
# If best_trial object has a 'checkpoint' attribute directly use it.
# Otherwise, construct path manually or use specific methods if available.
# Often need to get the Checkpoint object from the analysis results.
# Let's try getting the best checkpoint path directly if available
# NOTE: The exact API to get the best Checkpoint object can vary slightly across Ray versions.
# This approach tries common methods. Adjust if needed based on your specific Ray version's API.
best_checkpoint_result = analysis.get_best_checkpoint(trial=best_trial, metric=METRIC_TO_OPTIMIZE, mode=OPTIMIZATION_MODE)

if best_checkpoint_result is None:
     # Sometimes checkpoint info is directly attached to the trial result if using newer reporting
     if hasattr(best_trial, 'checkpoint') and best_trial.checkpoint:
         best_checkpoint = best_trial.checkpoint
         print("Using checkpoint object directly from best trial result.")
     else:
        # Fallback: Try constructing path manually - assumes standard naming
        checkpoint_num_str = str(best_metrics.get("checkpoint_dir_name","checkpoint_000000")).split('_')[-1] # Heuristic
        best_checkpoint_path_manual = os.path.join(best_trial_logdir, f"checkpoint_{checkpoint_num_str}")
        if os.path.exists(best_checkpoint_path_manual):
             print(f"Warning: Could not retrieve checkpoint object directly. Using manually constructed path: {best_checkpoint_path_manual}")
             # Need to load from path, requires slightly different handling below
             best_checkpoint_dir = best_checkpoint_path_manual
             best_checkpoint = None # Mark that we don't have the object
        else:
             raise ValueError(f"Could not find best checkpoint object or directory for trial {best_trial.trial_id}")
elif isinstance(best_checkpoint_result, str): # Older API might return path
     best_checkpoint_dir = best_checkpoint_result
     best_checkpoint = None
     print(f"Using checkpoint path returned by API: {best_checkpoint_dir}")
else: # Newer API likely returns Checkpoint object
     best_checkpoint = best_checkpoint_result # ray.air.Checkpoint object
     best_checkpoint_dir = None # Will get dir from object later
     print(f"Using checkpoint object returned by API.")


print(f"\n--- Best Trial Information ---")
print(f"Trial ID: {best_trial.trial_id if hasattr(best_trial, 'trial_id') else os.path.basename(best_trial_logdir)}")
print(f"Log Directory: {best_trial_logdir}")
print(f"Validation MAE (val_loss): {best_metrics.get(METRIC_TO_OPTIMIZE, 'N/A'):.4f}")
print(f"Validation Robust Overlap: {best_metrics.get('val_overlap_robust', 'N/A'):.4f}")
# print(f"Config: {best_config}")

# --- Reload Data ---
# ... (Data loading and splitting remains the same) ...
data_file_path = best_config["data_file"]
use_log1p_target = best_config["use_log1p_target"]
data_file_name = os.path.basename(data_file_path)
try: k = int(data_file_name.split("_")[0])
except Exception as e: raise ValueError(f"Cannot extract 'k' from data file name: {data_file_name}") from e
# start_date = pd.to_datetime("2008-04-18"); split_date = pd.to_datetime("2020-10-17")
print(f"\nReloading data from: {data_file_path} with k={k}")
if not os.path.exists(data_file_path): raise FileNotFoundError(f"Data file not found: {data_file_path}")
# combined_df = pd.read_csv(data_file_path); combined_df["date"] = pd.to_datetime(combined_df["date"])
# train_val_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)].copy().sort_values("date")
# unique_dates = train_val_df["date"].unique()
# if len(unique_dates) < 2: raise ValueError("Not enough unique dates to split.")
# split_idx = int(0.8 * len(unique_dates)); split_idx = max(0, min(split_idx, len(unique_dates) - 2))
# threshold_date = unique_dates[split_idx]
# val_df = train_val_df[train_val_df["date"] > threshold_date].copy()
combined_df = pd.read_csv(data_file_path)
combined_df["date"] = pd.to_datetime(combined_df["date"])
start_date = pd.to_datetime("2008-04-18")
split_date = pd.to_datetime("2023-10-17") # Train Val vs Test split
end_date = pd.to_datetime("2025-04-04") # End of test data

# Check if test set exists based on date conditions
# Make sure split_date is within the data range and before today/end_date
test_available = (split_date < combined_df["date"].max()) and \
                    (split_date < pd.to_datetime("today"))

# 1. Split into train_val and test based on dates.
train_val_df = combined_df.copy()
test_df = combined_df[(combined_df["date"] > end_date) & (combined_df["date"] <= pd.to_datetime("today"))].copy()

# # 2. Further split train_val into training and validation sets (time-based 80/20 split).
# train_val_df = train_val_df.sort_values("date")
# unique_dates = train_val_df["date"].unique()
# if len(unique_dates) < 2:
#      raise ValueError("Not enough unique dates in train_val_df to perform a split.")
# split_idx = int(0.8 * len(unique_dates))
# # Ensure split_idx is valid, handle edge case of very few unique dates
# split_idx = max(0, min(split_idx, len(unique_dates) - 2)) # Ensure at least one date for validation
# threshold_date = unique_dates[split_idx]

train_df = train_val_df[train_val_df["date"] <= split_date]
val_df = train_val_df[(train_val_df["date"] > split_date) & (train_val_df["date"] <= end_date)]
print(f"Recreated Validation DF shape: {val_df.shape}")
if val_df.empty: raise ValueError("Validation DataFrame empty.")
X_val, feature_names = process(val_df, k)
y_val_orig = val_df["fantasy_points"].values
val_match_ids = val_df["match_id"].values

# --- Load Best Model ---
print("Loading best LGBM model from checkpoint...")
# Check if we have the Checkpoint object or just the directory path
if best_checkpoint is not None:
    # Preferred way using Checkpoint object
    with best_checkpoint.as_directory() as ckpt_dir:
        model_path = os.path.join(ckpt_dir, "model.joblib")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found in checkpoint: {model_path}")
        model = joblib.load(model_path)
elif best_checkpoint_dir is not None and os.path.isdir(best_checkpoint_dir):
     # Fallback using directory path
     model_path = os.path.join(best_checkpoint_dir, "model.joblib")
     if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found in checkpoint dir: {model_path}")
     model = joblib.load(model_path)
else:
     raise ValueError("Could not determine valid checkpoint path or object.")

print(f"Model loaded successfully.")


# --- Perform Inference ---
print("Performing inference on validation data...")
predictions_target_space = model.predict(X_val)

# --- Inverse Transform Predictions ---
if use_log1p_target:
    predictions_orig = np.expm1(predictions_target_space)
    predictions_orig = np.maximum(0, predictions_orig)
else:
    predictions_orig = predictions_target_space

predictions_orig_int = predictions_orig.astype(int)
targets_orig_int = y_val_orig.astype(int)
lr = best_config.get('learning_rate', 'NA')
num_leaves = best_config.get('num_leaves', 'NA')
feat_frac = best_config.get('feature_fraction', 'NA')
bag_frac = best_config.get('bagging_fraction', 'NA')
l1 = best_config.get('lambda_l1', 'NA')
l2 = best_config.get('lambda_l2', 'NA')

# Format hyperparameters into a string (keep it reasonably short)
# Using f-string formatting for numbers where appropriate
hyperparam_str = f"lr{lr:.4f}_nl{num_leaves}_ff{feat_frac:.3f}_bf{bag_frac:.3f}_l1{l1:.2f}_l2{l2:.2f}"
# Clean the string for filename usage (replace dots, etc. if needed)
hyperparam_str_safe = hyperparam_str.replace('.', '_')
# --- Generate Plots ---
# ... (Plotting code remains the same) ...
print("\n--- Generating Plots ---")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
plt.figure(figsize=(8, 8))
plt.scatter(y_val_orig, predictions_orig, alpha=0.5)
max_val = max(max(y_val_orig, default=0), max(predictions_orig, default=0))
min_val = min(min(y_val_orig, default=0), min(predictions_orig, default=0))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
plt.title(f'Predicted vs. Actual FP (Validation Set - Best LGBM)')
plt.xlabel('Actual Points'); plt.ylabel('Predicted Points'); plt.grid(True); plt.legend(); plt.axis('equal')
plot_filename_scatter = os.path.join(ANALYSIS_OUTPUT_DIR, f"best_lgbm_pred_vs_actual_{hyperparam_str_safe}.png")
plt.savefig(plot_filename_scatter); plt.close()
print(f"Saved predicted vs actual plot: {plot_filename_scatter}")


# --- Verify Model Metrics ---
# ... (Metrics calculation remains the same) ...
print("\n--- Verifying Best LGBM Model Performance ---")
model_overlap_robust = compute_overlap_robust(targets_orig_int, predictions_orig_int, val_match_ids)
model_overlap_score = compute_overlap_true_test(targets_orig_int, predictions_orig_int, val_match_ids)


print(f"Model Overlap Score (Robust): {model_overlap_robust:.4f}")
print(f"Model Overlap Score : {model_overlap_score:.4f}")

model_mse = mean_squared_error(y_val_orig, predictions_orig)
model_mae = mean_absolute_error(y_val_orig, predictions_orig)
print(f"Model MSE: {model_mse:.4f}")
print(f"Model MAE: {model_mae:.4f}")

print("\n--- Best LGBM Evaluation Complete ---")