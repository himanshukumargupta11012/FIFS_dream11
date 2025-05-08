import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import pickle
import json
import joblib # For saving the scikit-learn model object
import tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time 
from ray import tune,air
from ray.air import session
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# --- Assuming these are imported from your project ---
# from feature_utils import process, compute_overlap_robust, compute_loss
# Example placeholder imports (replace with your actual imports)
from feature_utils import (
    compute_overlap_true_test,
    process,
    compute_overlap_robust, # Using the robust version
    # compute_loss # Not strictly needed unless reported as metric
    compute_overlap_true_test
)

# --- Configuration ---
# Default config values (will be overridden by Ray Tune)
DEFAULT_CONFIG = {
    "data_file": "/path/to/your/feature_file.csv", # IMPORTANT: Provide a default or ensure it's always in the tune config
    "use_log1p_target": True, # Set based on your best MLP setup
    # --- LightGBM Search Space ---
    "n_estimators": 2000, # Fixed high value, rely on early stopping
    "learning_rate": tune.loguniform(1e-3, 1e-1),
    "num_leaves": tune.randint(20, 150),
    "max_depth": tune.randint(3, 15),
    "feature_fraction": tune.uniform(0.5, 0.9), # Equivalent to colsample_bytree
    "bagging_fraction": tune.uniform(0.5, 0.9), # Equivalent to subsample
    "bagging_freq": tune.choice([1, 3, 5]),
    "lambda_l1": tune.loguniform(1e-2, 10.0), # L1 regularization
    "lambda_l2": tune.loguniform(1e-2, 10.0), # L2 regularization
    # --- Fixed LGBM Params ---
    "objective": "regression_l1", # MAE loss
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_jobs": 1,
    "seed": 42,
}

# --- Trainable Function for Ray Tune ---
def train_lgbm_tune(config):
    """
    Trainable function for Ray Tune with LightGBM.
    Handles data loading, preprocessing, training, evaluation, metric reporting,
    and checkpointing.
    """
    # --- 1. Load Data and Apply Splits ---
    data_file_path = config["data_file"]
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    # Extract k value (assuming file naming convention)
    data_file_name = os.path.basename(data_file_path)
    try: k = int(data_file_name.split("_")[0])
    except Exception as e: raise ValueError(f"Cannot extract 'k' from data file name: {data_file_name}") from e

    # # Dates (Must match the data split used for MLP)
    # start_date = pd.to_datetime("2008-04-18")
    # split_date = pd.to_datetime("2020-10-17")

    combined_df = pd.read_csv(data_file_path)
    combined_df["date"] = pd.to_datetime(combined_df["date"])

    # # Re-apply the exact same train/validation split logic
    # train_val_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)].copy()
    # train_val_df = train_val_df.sort_values("date")
    # unique_dates = train_val_df["date"].unique()
    # if len(unique_dates) < 2: raise ValueError("Not enough unique dates to split.")
    # split_idx = int(0.8 * len(unique_dates)); split_idx = max(0, min(split_idx, len(unique_dates) - 2))
    # threshold_date = unique_dates[split_idx]

    # train_df = train_val_df[train_val_df["date"] <= threshold_date].copy()
    # val_df = train_val_df[train_val_df["date"] > threshold_date].copy()
    # if train_df.empty or val_df.empty: raise ValueError("Train or Validation DataFrame empty.")
     # Define date boundaries (consider making these configurable if they change often)
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
    # --- 2. Process Features ---
    # No Scaling needed for LightGBM features
    X_train, feature_names = process(train_df, k)
    X_val, _ = process(val_df, k)
    print(f"Processed features. Train X shape: {X_train.shape}, Val X shape: {X_val.shape}")

    # --- 3. Prepare Target Variable ---
    y_train_orig = train_df["fantasy_points"].values
    y_val_orig = val_df["fantasy_points"].values # Keep original targets for evaluation

    if config["use_log1p_target"]:
        # print("Using log1p transformation for target variable.") # Reduce verbosity in tune
        y_train_target = np.log1p(np.maximum(0, y_train_orig))
        y_val_target = np.log1p(np.maximum(0, y_val_orig))
    else:
        y_train_target = y_train_orig
        y_val_target = y_val_orig

    # --- 4. Initialize and Train LightGBM Model ---
    lgbm_params = {
        'objective': config["objective"],
        'metric': config["metric"],
        'n_estimators': config["n_estimators"],
        'learning_rate': config["learning_rate"],
        'num_leaves': config["num_leaves"],
        'max_depth': config["max_depth"],
        'feature_fraction': config["feature_fraction"],
        'bagging_fraction': config["bagging_fraction"],
        'bagging_freq': config["bagging_freq"],
        'lambda_l1': config["lambda_l1"],
        'lambda_l2': config["lambda_l2"],
        'boosting_type': config["boosting_type"],
        'seed': config["seed"],
        'n_jobs': config["n_jobs"],
        'verbose': -1, # Suppress verbose training output from LGBM
    }

    model = lgb.LGBMRegressor(**lgbm_params)

    # Train with early stopping (using validation set defined above)
    model.fit(X_train, y_train_target,
              eval_set=[(X_val, y_val_target)],
              eval_metric='mae', # Monitor MAE on validation set
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]) # Stop if MAE doesn't improve

    # --- 5. Evaluate on Validation Set ---
    if model.best_iteration_ is None or model.best_iteration_ <= 0:
         print("Warning: Early stopping triggered too early or failed. Using last iteration.")
         best_iteration = config["n_estimators"] # Fallback, may not be ideal
    else:
         best_iteration = model.best_iteration_

    predictions_target_space = model.predict(X_val, num_iteration=best_iteration)

    # Inverse Transform Predictions if necessary
    if config["use_log1p_target"]:
        predictions_orig = np.expm1(predictions_target_space)
        predictions_orig = np.maximum(0, predictions_orig) # Clip
    else:
        predictions_orig = predictions_target_space

    # Convert for metrics
    predictions_orig_int = predictions_orig.astype(int)
    targets_orig_int = y_val_orig.astype(int) # Use original targets

    # --- 6. Calculate Metrics ---
    val_match_ids = val_df["match_id"].values
    # Calculate Robust Overlap Score
    val_overlap_robust = compute_overlap_robust(targets_orig_int, predictions_orig_int, val_match_ids)
    val_overlap = compute_overlap_true_test(targets_orig_int, predictions_orig_int, val_match_ids)
    # Calculate standard metrics on original scale
    val_mse = mean_squared_error(y_val_orig, predictions_orig)
    val_mae = mean_absolute_error(y_val_orig, predictions_orig)

    # --- 7. Checkpointing & Reporting ---
    metrics = {
        "val_loss": val_mae, # Use MAE as the primary loss metric reported to Tune scheduler
        "val_overlap_robust": val_overlap_robust,
        "val_overlap" : val_overlap,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "num_boost_round": best_iteration # Log the number of boosting rounds used
    }

    # Save the model using joblib within a temporary directory for Ray Checkpoint
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        model_path = os.path.join(temp_checkpoint_dir, "model.joblib")
        joblib.dump(model, model_path)

        # Add feature names list to checkpoint if needed for evaluation later
        # feature_names_path = os.path.join(temp_checkpoint_dir, "feature_names.pkl")
        # with open(feature_names_path, 'wb') as f:
        #     pickle.dump(feature_names, f)

        # Report metrics and checkpoint to Ray Tune
        session.report(metrics=metrics, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))


# --- Main execution block for running Ray Tune ---
if __name__ == "__main__":
    # IMPORTANT: Update this path to your actual feature file
    DATA_FILE = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/processed/combined/10_ipl.csv" # Use the SAME feature file as the MLP benchmark

    # Check if the data file exists before starting Tune
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Please update the DATA_FILE variable in train_lgbm_tune.py. File not found: {DATA_FILE}")

    # Define the search space using the default config
    search_space = DEFAULT_CONFIG.copy()
    search_space["data_file"] = DATA_FILE # Ensure the correct data file is used

    # Configure the ASHA scheduler (optional, but good for efficiency)
    scheduler = ASHAScheduler(
        # metric="val_loss", # Optimize based on validation MAE
        # mode="min",
        max_t=DEFAULT_CONFIG['n_estimators'], # Max boosting rounds (early stopping can finish sooner)
        grace_period=50, # Allow at least 50 rounds before stopping trials
        reduction_factor=2
    )

    # Define resources per trial (adjust based on your machine)
    resources_per_trial = {"cpu": 1} # Example: 2 CPUs per trial

    print("Starting Ray Tune for LightGBM...")
    tuner = tune.Tuner(
        train_lgbm_tune,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=25,  # Number of hyperparameter combinations to try
            scheduler=scheduler,
            metric="val_loss", # Metric to optimize
            mode="min",         # Minimize MAE
        ),
        run_config= air.RunConfig(
            name="LGBM_Hyperparam_Tuning" + "_" +  time.strftime("%Y%m%d-%H%M%S"), # Experiment name
            storage_path="/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/ray_results", # IMPORTANT: Set storage path
            # local_dir="/path/to/your/ray_results", # Optional: Alias for storage_path often used
            stop={"training_iteration": 1}, # Each trial runs only 1 "iteration" (the full fit call)
             checkpoint_config=air.CheckpointConfig(
                 num_to_keep=3, # Keep top 3 checkpoints based on metric
                 checkpoint_score_attribute="val_loss", # Use val_loss for ranking
                 checkpoint_score_order="min",
             ),
        ),
         # resources=resources_per_trial # Use if using older Ray versions < 2.9
         # resources_per_trial={"cpu": 2} # Example for Ray 2.9+ Tuner API
         # You might need to adjust how resources are specified depending on your Ray version
    )
    results = tuner.fit()

    print("Ray Tune finished.")
    best_result = results.get_best_result(metric="val_loss", mode="min")
    print("--- Best Trial ---")
    print(f"Checkpoint Path: {best_result.checkpoint.path}")
    print(f"Validation MAE (val_loss): {best_result.metrics['val_loss']:.4f}")
    print(f"Validation Overlap : ,{best_result.metrics['val_overlap']:.4f}")
    print(f"Validation Robust Overlap: {best_result.metrics['val_overlap_robust']:.4f}")
    print(f"Hyperparameters: {best_result.config}")