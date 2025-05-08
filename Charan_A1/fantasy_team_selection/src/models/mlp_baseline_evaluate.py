import torch
# from torch import nn # Not directly needed in this script snippet
# from torch.utils.data import DataLoader, TensorDataset # Not directly needed
import pandas as pd
import os
# Assuming these are imported from your project modules
from model_utils import CustomMLPModel #, train_model, evaluate_model
# from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data
from feature_utils import process, compute_overlap_true_test, compute_loss # Assuming normalise_data isn't needed directly here
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
CHECKPOINT_FILE_PATH = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/ray_results/MLP_Hyperparam_Tuning_20250418-221720/train_mlp_tune_cd28e_00024_24_activation=relu,batch_size=256,dropout_prob=0.2819,hidden_layers=128,lr=0.0006,smooth_beta=15,weight_2025-04-18_22-20-12/checkpoint_000011/checkpoint.pt"
ANALYSIS_OUTPUT_DIR = "analysis_plots_specific" # Subdirectory to save plots
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Functions (calculate_baseline_overlap, get_trial_info_from_checkpoint) ---
# Keep these functions as they were in the previous version
def calculate_baseline_overlap(df, feature_name, target_col='fantasy_points', group_col='match_id'):
    """Calculates overlap score using a baseline feature for prediction ranking."""
    all_overlaps = []
    grouped = df.groupby(group_col)
    if feature_name not in df.columns:
        print(f"Error: Baseline feature '{feature_name}' not found in DataFrame.")
        return 0.0
    if pd.api.types.is_numeric_dtype(df[feature_name]) :
        for match_id, group in grouped:
            if len(group) < 11: continue
            top_predicted_indices = group[feature_name].nlargest(11).index[:11]
            top_actual_indices = group[target_col].nlargest(11).index[:11]
            overlap_count = len(set(top_predicted_indices).intersection(set(top_actual_indices)))
            all_overlaps.append(overlap_count)
        return np.mean(all_overlaps) if all_overlaps else 0.0
    else :
        return 0.0

def get_trial_info_from_checkpoint(ckpt_path):
    """Navigates up from checkpoint file to find trial dir and load params.json"""
    if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Ckpt not found: {ckpt_path}")
    checkpoint_dir = os.path.dirname(ckpt_path)
    trial_dir = os.path.dirname(checkpoint_dir)
    params_path = os.path.join(trial_dir, "params.json")
    if not os.path.exists(params_path):
        params_path_pkl = os.path.join(trial_dir, "params.pkl")
        if not os.path.exists(params_path_pkl): raise FileNotFoundError(f"No params file found in {trial_dir}")
        print("Loading params from params.pkl")
        with open(params_path_pkl, 'rb') as f: params = pickle.load(f)
    else:
        print(f"Loading params from: {params_path}")
        with open(params_path, 'r') as f: params = json.load(f)
    return trial_dir, params

# --- Main Analysis Logic ---
print(f"Analyzing specific checkpoint: {CHECKPOINT_FILE_PATH}")
try: # Load config
    trial_dir, trial_config = get_trial_info_from_checkpoint(CHECKPOINT_FILE_PATH)
    print(f"Loaded config from trial directory: {trial_dir}")
except Exception as e: raise RuntimeError(f"Failed to load trial info: {e}")

# # Extract config details
# data_file_path = trial_config.get("data_file")
# model_name_prefix = trial_config.get("model_name", "MLP")
# hidden_layers = trial_config.get("hidden_layers")
# activation = trial_config.get("activation")
# dropout_prob = trial_config.get("dropout_prob")
# weight_init = trial_config.get("weight_init")
# if not all([data_file_path, hidden_layers, activation, dropout_prob is not None, weight_init]):
#      raise ValueError("Missing essential parameters in loaded config.")
# data_file_name = os.path.basename(data_file_path)
# try: k = int(data_file_name.split("_")[0])
# except Exception as e: raise ValueError(f"Cannot extract 'k' from data file name: {data_file_name}") from e
# start_date = pd.to_datetime("2008-04-18")
# split_date = pd.to_datetime("2020-10-17")

# try: # Load scalers
#     analysis_script_dir = os.path.dirname(os.path.abspath(__file__))
#     artifact_base_path = os.path.join(analysis_script_dir, "..", "model_artifacts", f"{model_name_prefix}_d-{k}_sd-{start_date.strftime('%Y-%m-%d')}_ed-{split_date.strftime('%Y-%m-%d')}")
#     scaler_path = f"{artifact_base_path}_scalers.pkl"
#     print(f"Attempting to load scalers from: {scaler_path}")
#     with open(scaler_path, "rb") as file: scalers_dict = pickle.load(file)
#     scaler_X = scalers_dict["x"]; scaler_y = scalers_dict["y"]
#     print("Scalers loaded successfully.")
# except Exception as e: raise RuntimeError(f"Error loading scalers from '{scaler_path}': {e}")

# # Load and prepare validation data
# print(f"Reloading data from: {data_file_path} with k={k}")
# if not os.path.exists(data_file_path): raise FileNotFoundError(f"Data file not found: {data_file_path}")
data_file_path = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/processed/combined/10_ipl.csv"
combined_df = pd.read_csv(data_file_path)
combined_df["date"] = pd.to_datetime(combined_df["date"])
# train_val_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)].copy().sort_values("date")
# unique_dates = train_val_df["date"].unique()
# if len(unique_dates) < 2: raise ValueError("Not enough unique dates to split.")
# split_idx = int(0.8 * len(unique_dates)); split_idx = max(0, min(split_idx, len(unique_dates) - 2))
# threshold_date = unique_dates[split_idx]
# val_df = train_val_df[train_val_df["date"] > threshold_date].copy()
val_df = combined_df[combined_df["date"] >= pd.to_datetime("2023-02-02")]

# print(f"Recreated Validation DF shape: {val_df.shape}")
# if val_df.empty: raise ValueError("Validation DataFrame empty.")
# X_val, cols = process(val_df, k)
# X_val_scaled = scaler_X.transform(X_val)
# y_val_orig = val_df["fantasy_points"].values # Keep the original target for final evaluation
# val_match_ids = val_df["match_id"].values



# # Load model
# print(f"Loading model state from: {CHECKPOINT_FILE_PATH}")
# checkpoint_data = torch.load(CHECKPOINT_FILE_PATH, map_location=device)
# num_input_features = X_val_scaled.shape[1]
# model = CustomMLPModel(input_size=num_input_features, hidden_layers=hidden_layers, activation=activation, dropout_prob=dropout_prob, weight_init=weight_init).to(device)
# model.load_state_dict(checkpoint_data["model_state_dict"])
# model.eval()
# print(f"Model loaded successfully from epoch {checkpoint_data.get('epoch', 'N/A')}.")

# # Perform Inference
# print("Performing inference on validation data...")
# X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
# with torch.no_grad():
#     predictions_scaled = model(X_val_tensor) # Model predicts scaled log(1+y)
# predictions_scaled_np = predictions_scaled.cpu().numpy()

# # --- Inverse transform predictions - ADJUSTED FOR LOG1P ---
# try:
#     # Step 1: Inverse scale using the scaler fitted on log-transformed data
#     predictions_log_unscaled = scaler_y.inverse_transform(predictions_scaled_np).flatten()

#     # Step 2: Inverse the log1p transformation
#     predictions_orig = np.expm1(predictions_log_unscaled)

#     # Step 3: Clip predictions at 0 as expm1 can sometimes produce small negatives
#     predictions_orig = np.maximum(0, predictions_orig)

# except Exception as e:
#     # Fallback if scaling/transform fails - less likely now but good practice
#     print(f"Warning: Inverse transform failed during analysis: {e}. Check scaler compatibility.")
#     # If scaler_y was NOT fitted on log-transformed data, this block might be hit.
#     # The fallback below assumes direct prediction (no log transform) which would be WRONG if log1p was used in training.
#     predictions_orig = scaler_y.inverse_transform(predictions_scaled_np).flatten() # Incorrect if trained with log1p
#     predictions_orig = np.maximum(0, predictions_orig) # Basic clipping

# print("Inference complete. Predictions reverted to original fantasy point scale.")

# # Convert final predictions and original targets for metrics
# predictions_orig_int = predictions_orig.astype(int)
# targets_orig_int = y_val_orig.astype(int) # Use original targets

# # --- Generate Plots (Using original scale targets and predictions) ---
# print("\n--- Generating Plots ---")
# os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# # Plot 1: Distribution of Actual Target Variable (y_val_orig)
# plt.figure(figsize=(10, 5))
# sns.histplot(y_val_orig, kde=True, bins=50) # Plot distribution of ORIGINAL points
# plt.title(f'Distribution of Actual FP (Validation Set - Ckpt Epoch {checkpoint_data.get("epoch", "N/A")})')
# plt.xlabel('Fantasy Points'); plt.ylabel('Frequency'); plt.grid(True)
# plot_filename_hist = os.path.join(ANALYSIS_OUTPUT_DIR, f"target_distribution_ckpt_ep{checkpoint_data.get('epoch', 'N_A')}.png")
# plt.savefig(plot_filename_hist); plt.close()
# print(f"Saved target distribution plot: {plot_filename_hist}")

# # Plot 2: Predicted vs. Actual Scatter Plot (predictions_orig vs y_val_orig)
# plt.figure(figsize=(8, 8))
# plt.scatter(y_val_orig, predictions_orig, alpha=0.5) # Use original scale for both axes
# max_val = max(max(y_val_orig, default=0), max(predictions_orig, default=0))
# min_val = min(min(y_val_orig, default=0), min(predictions_orig, default=0))
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
# plt.title(f'Predicted vs. Actual FP (Validation Set - Ckpt Epoch {checkpoint_data.get("epoch", "N/A")})')
# plt.xlabel('Actual Points'); plt.ylabel('Predicted Points'); plt.grid(True); plt.legend(); plt.axis('equal')
# # Create a more manageable filename string
# hid_layer_str = '_'.join(map(str, hidden_layers)) if isinstance(hidden_layers, list) else str(hidden_layers)
# plot_filename_scatter = os.path.join(ANALYSIS_OUTPUT_DIR, f"pred_vs_actual_ckpt_ep{checkpoint_data.get('epoch', 'N_A')}_{activation}_{hid_layer_str}_{dropout_prob:.4f}_{weight_init}.png")
# plt.savefig(plot_filename_scatter); plt.close()
# print(f"Saved predicted vs actual plot: {plot_filename_scatter}")

# --- Calculate Baselines (using original data in val_df) ---
print("\n--- Calculating Baseline Overlap Scores ---")
baseline_features = val_df.columns
available_baseline_features = [f for f in baseline_features if f in val_df.columns]
if not available_baseline_features: print("Warning: Baseline features not found.")
else:
    for feature in available_baseline_features:
        val_df_baseline = val_df.copy()
        if val_df_baseline[feature].isnull().any():
             print(f"Note: Filling NaNs in baseline '{feature}' with 0.")
             val_df_baseline[feature].fillna(0, inplace=True)
        baseline_overlap = calculate_baseline_overlap(val_df_baseline, feature, target_col='fantasy_points')
        print(f"Baseline Overlap using '{feature}': {baseline_overlap:.4f}")

# # --- Verify Model Metrics (using original scale targets and predictions) ---
# print("\n--- Verifying Model Performance on Validation Set ---")
# # Use original integer targets (targets_orig_int) and integer predictions (predictions_orig_int) for overlap
# model_overlap_verification = compute_overlap_true_test(targets_orig_int, predictions_orig_int, val_match_ids)
# print(f"Model Overlap Score (recalculated): {model_overlap_verification:.4f}")
# # Use original float targets (y_val_orig) and float predictions (predictions_orig) for MSE/MAE
# model_mse = mean_squared_error(y_val_orig, predictions_orig)
# model_mae = mean_absolute_error(y_val_orig, predictions_orig)
# print(f"Model MSE (recalculated): {model_mse:.4f}")
# print(f"Model MAE (recalculated): {model_mae:.4f}")

# print("\n--- Specific Checkpoint Analysis Complete ---")