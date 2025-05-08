# python train_lgbm_fixed_params.py --data_file /path/to/your/feature_file.csv

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import pickle
import joblib # For saving the LGBM model object
import argparse
import time

# Assuming process is correctly imported from your project's feature_utils
from feature_utils import process

# --- Seed Setting ---
def set_seeds(seed=42): # Use a fixed seed consistent with LGBM params if needed
    np.random.seed(seed)
    # Note: LGBM seeding is primarily controlled by the 'seed' parameter in its config

set_seeds()

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LightGBM model on the full dataset with fixed hyperparameters.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Full path to the CSV feature data file (e.g., /path/to/data/processed/combined/10_ipl.csv)")
    args = parser.parse_args()

    # ----- Hardcoded Hyperparameters -----
    output_dir = "./lgbm_fixed_output"
    use_log1p_target = True # Set based on your findings (True/False)

    lgbm_params = {
        # --- Best Values ---
        'learning_rate': 0.09069382114542723,       
        'num_leaves': 98,             
        'max_depth': 8,              
        'feature_fraction': 0.7275015837385048,     
        'bagging_fraction': 0.6063662863102393,    
        'bagging_freq': 1,           
        'lambda_l1': 0.033554454545662304,             
        'lambda_l2': 1.6041939342789853,             
        # --- Fixed ---
        'objective': 'regression_l1', # MAE loss. Change to 'regression' for MSE if needed
        'metric': 'mae',              # Monitor MAE during training (internal, not used for early stopping here)
        'n_estimators': 1500,         # Number of boosting rounds (fixed, no early stopping)
        'boosting_type': 'gbdt',
        'n_jobs': 1,                 # Use all available cores
        'seed': 42,
        'verbose': -1,                # Suppress verbose training output from LGBM
    }
    # -----------------------------------

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting LightGBM training run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {args.data_file}")
    print("Using hardcoded hyperparameters:")
    print(f"  Output Directory: {output_dir}")
    print(f"  Use log1p Target: {use_log1p_target}")
    for key, value in lgbm_params.items():
        print(f"  {key}: {value}")

    # --- 1. Load Data ---
    data_file_path = args.data_file
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    data_file_name = os.path.basename(data_file_path)
    try:
        k = int(data_file_name.split("_")[0])
        print(f"Inferred window size 'k' = {k} from filename.")
    except Exception as e:
        print(f"Warning: Could not infer 'k' from filename '{data_file_name}'. Using k=0 for processing.")
        k = 0

    print(f"Loading data from: {data_file_path}")
    combined_df = pd.read_csv(data_file_path)
    combined_df["date"] = pd.to_datetime(combined_df["date"]) # Ensure date is datetime if process uses it

    print(f"Total data shape: {combined_df.shape}")
    if combined_df.empty:
        raise ValueError("Input DataFrame is empty.")

    # --- 2. Process Features (on full data) ---
    print("Processing features...")
    X_full, feature_names = process(combined_df, k)
    print(f"Processed features. X_full shape: {X_full.shape}")
    if isinstance(feature_names, tuple): # Handle if process returns tuple (like X, cols)
        feature_names = feature_names[1] # Assuming second element is column names

    nan_count = np.isnan(X_full).sum()
    inf_count = np.isinf(X_full).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaNs and {inf_count} Infs in processed features. LGBM might handle NaNs; check behavior if issues arise.")
        # LGBM can handle NaNs internally, but Infs might cause issues.

    # --- 3. Prepare Target Variable (on full data) ---
    print("Preparing target variable...")
    y_full_orig = combined_df["fantasy_points"].values

    if use_log1p_target:
        print("Applying log1p transformation to target.")
        y_full_target = np.log1p(np.maximum(0, y_full_orig))
    else:
        y_full_target = y_full_orig

    # --- 4. Initialize and Train LightGBM Model ---
    print(f"Initializing LightGBM model...")
    model = lgb.LGBMRegressor(**lgbm_params)

    print(f"Starting training for {lgbm_params['n_estimators']} boosting rounds...")
    # --- Prepare feature_name parameter correctly ---
    feature_name_param = 'auto' # Default value
    if feature_names is not None:
        if isinstance(feature_names, pd.Index):
            # Convert Pandas Index to list if it's not empty
            if not feature_names.empty:
                feature_name_param = feature_names.tolist()
        elif isinstance(feature_names, list):
            # Use the list directly if it's not empty
            if feature_names:
                feature_name_param = feature_names
        # If feature_names is neither Index nor list, or if it's empty,
        # feature_name_param will remain 'auto'

    # --- Corrected model.fit call ---
    model.fit(X_full, y_full_target,feature_name=feature_name_param,categorical_feature='auto')
    # --- 5. Save Model and Feature Names ---
    training_duration = time.time() - start_time
    print(f"\nFinished Training in {training_duration:.2f} seconds.")

    # Save Model using Joblib
    model_filename = f"lgbm_k{k}_final_model.joblib"
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Saved final LightGBM model to {model_path}")

    # Save Feature Names using Pickle
    features_filename = f"lgbm_k{k}_feature_names.pkl"
    features_path = os.path.join(output_dir, features_filename)
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Saved feature names to {features_path}")

    print("Script completed.")