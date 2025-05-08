# python mlp_baseline_tuner.py --data_file /home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/processed/combined/10_window_ipl.csv --num_samples 25 --epochs 100 --storage_path /home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/ray_results

from itertools import count
import os
import argparse
import pickle
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
# Removed: from torch.utils.tensorboard import SummaryWriter # No longer needed here
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tempfile # Added for checkpointing

# Import custom functions and models from your modules.
from model_utils import CustomMLPModel,LogSpaceWeightedMSELoss
from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data,compute_overlap_robust
from ray.train import CheckpointConfig
# Import Ray Tune components
from ray import tune
from ray.tune import CLIReporter
import ray
# Import AIR components for modern API
from ray.air import session
from ray.tune import Checkpoint
# Import the TensorBoard callback
from ray.tune.logger import TBXLoggerCallback
# Initialize Ray (considerdoing this within the main block)
# ray.init(num_gpus=3, ignore_reinit_error=True) # Use ignore_reinit_error if running interactively

# Set seeds for reproducibility
def set_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Check if CUDA is available before setting CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train_mlp_tune(config): # Removed checkpoint_dir argument, use session API instead
    """
    Train the MLP model using Ray Tune with AIR session API.
    Metrics reported via session.report() are automatically logged by callbacks (like TensorBoard).
    Checkpoints are handled via session.report() and session.get_checkpoint().

    config should include:
         - data_file: string (full path to the CSV file)
         - epochs: int, number of epochs to train
         - batch_size: int, training batch size
         - hidden_layers: list of int, hidden layer sizes
         - activation: str, activation function name
         - dropout_prob: float, dropout probability
         - weight_init: str, weight initialization method
         - lr: float, learning rate
         - model_name: str, name prefix to save artifacts.
         - do_test: bool, if True then run evaluation on a separate test set after training.
    """
    # ----- Setup -----
    # TensorBoard SummaryWriter is no longer created manually here.
    # The TensorboardLoggerCallback passed to tune.run handles it.

    # Unpack configuration
    data_file_path = config["data_file"] # Expecting full path now
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["lr"]
    model_name_prefix = config.get("model_name", "MLP") # Prefix for artifact names
    do_test = config.get("do_test", False)
    beta = config.get("smooth_beta",10)
    # Extract k from the filename (e.g., "/path/to/7_final.csv")
    data_file_name = os.path.basename(data_file_path) # Get "7_final.csv"
    try:
        # Handle potential ".csv" extension
        k = int(data_file_name.split("_")[0])
    except Exception as e:
        raise ValueError(f"Data file name base ('{data_file_name}') should start with an integer indicating 'k', e.g., '7_final.csv'") from e

    scalers_dict = {}
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of this script

    print(f"\n--------- Running Trial for Data File: {data_file_name} ---------")

    # ----- Data Loading and Processing -----
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"Data file not found at: {data_file_path}")
    combined_df = pd.read_csv(data_file_path)
    combined_df["date"] = pd.to_datetime(combined_df["date"])

    # print(f"NaNs in initial combined_df: {combined_df.isnull().sum().sum()}")
    numeric_cols_initial = combined_df.select_dtypes(include=np.number).columns
    # print(f"Infs in initial combined_df: {np.isinf(combined_df[numeric_cols_initial]).sum().sum()}")

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

    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}, Test shape: {test_df.shape}")
    if train_df.empty or val_df.empty:
        raise ValueError("Training or Validation DataFrame is empty after split.")

    # 3. Process features and extract target
    X_train, cols = process(train_df, k)
    X_val, _ = process(val_df, k)
    # print(cols)

    # print(f"Shape of X_train before normalization: {X_train.shape}")
    # print(f"Shape of X_val before normalization: {X_val.shape}")

    # Check for NaNs AFTER processing
    print(f"NaNs in X_train after process(): {np.isnan(X_train).sum()}")
    print(f"NaNs in X_val after process(): {np.isnan(X_val).sum()}")

    # Check for Infs AFTER processing
    print(f"Infs in X_train after process(): {np.isinf(X_train).sum()}")
    print(f"Infs in X_val after process(): {np.isinf(X_val).sum()}")
    y_train = train_df["fantasy_points"].values
    y_val = val_df["fantasy_points"].values

    y_train = np.maximum(0,y_train)
    y_val = np.maximum(0,y_val)

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    threshold_raw = np.log1p(70)

    # Fit normalization on training set and apply to validation
    X_train, y_train_log_scaled, scaler_X, scaler_y = normalise_data(X_train, y_train_log, MinMax=False)
    X_val = scaler_X.transform(X_val)
    y_val_log_scaled = scaler_y.transform(y_val_log.reshape(-1, 1)).flatten()
    threshold_scaled = scaler_y.transform(threshold_raw.reshape(-1,1)).flatten()

    scalers_dict["x"] = scaler_X
    scalers_dict["y"] = scaler_y

    # Define path for saving scalers (within the trial directory for better organization)
    # Note: Ray Tune might clean up trial directories depending on settings.
    # If you need scalers *after* the run, consider saving them outside or using Tune's result artifacts.
    # For now, let's save them relative to the script location as before, but maybe add trial ID?
    # Or better: save them with the final model checkpoint or as a reported artifact.
    # Let's stick to the original path for now for simplicity, but be aware of this.
    artifact_base_path = os.path.join(current_dir, "..", "model_artifacts", f"{model_name_prefix}_d-{k}_sd-{start_date.strftime('%Y-%m-%d')}_ed-{split_date.strftime('%Y-%m-%d')}")
    os.makedirs(os.path.dirname(artifact_base_path), exist_ok=True) # Ensure directory exists
    with open(f"{artifact_base_path}_scalers.pkl", "wb") as file:
        pickle.dump(scalers_dict, file)
    print(f"Saved scalers to {artifact_base_path}_scalers.pkl")

    # Create TensorDatasets and DataLoaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_log_scaled, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) # Added num_workers and pin_memory

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_log_scaled, dtype=torch.float32).view(-1, 1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Prepare Test DataLoader (if doing test evaluation)
    test_loader = None
    if do_test and test_available and not test_df.empty:
        X_test, _ = process(test_df, k)
        y_test = test_df["fantasy_points"].values
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    elif do_test:
        print("Warning: do_test=True but test data is not available or empty.")

    # 5. Create model, loss function and optimizer
    num_input_features = X_train.shape[1]
    print("Number of input features:", num_input_features)
    model = CustomMLPModel(
        input_size=num_input_features,
        hidden_layers=config["hidden_layers"],
        activation=config["activation"],
        dropout_prob=config["dropout_prob"],
        weight_init=config["weight_init"]
    ).to(device)

    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss(beta = beta)

    # alpha = config["alpha_loss"]
    # gamma = config["gamma_loss"] 
    
    # loss_fn = LogSpaceWeightedMSELoss(alpha,threshold_scaled,gamma)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ----- Load Checkpoint (using session API) -----
    start_epoch = 0
    # Use session.get_checkpoint() to retrieve the Checkpoint object
    checkpoint = session.get_checkpoint()
    if checkpoint:
        try:
            # Load from the directory provided by the Checkpoint object
            with checkpoint.as_directory() as checkpoint_dir:
                 checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
                 if os.path.exists(checkpoint_path):
                     checkpoint_data = torch.load(checkpoint_path, map_location=device)
                     model.load_state_dict(checkpoint_data["model_state_dict"])
                     optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                     start_epoch = checkpoint_data["epoch"] + 1
                     print(f"----- Resumed from epoch {start_epoch} -----")
                 else:
                     print("Checkpoint file 'checkpoint.pt' not found in checkpoint directory.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from epoch 0.")
            start_epoch = 0


    # ----- Training Loop -----
    print(f"Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True) # Use non_blocking with pin_memory
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        total_val_loss = 0.0
        predictions_list = []
        targets_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)
                total_val_loss += loss.item()
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(batch_y.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader)

        # Compute additional metrics on original scale for validation set
        predictions_all = np.concatenate(predictions_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)
        try:
            # Ensure inverse_transform gets 2D array if needed by scaler
            predictions_orig_log = scaler_y.inverse_transform(predictions_all).flatten()
            targets_orig_log = scaler_y.inverse_transform(targets_all).flatten()
            # Cast to int *after* inverse transform if necessary for metrics
            predictions_orig = np.expm1(predictions_orig_log)
            targets_orig = np.expm1(targets_orig_log)

            predictions_orig = np.maximum(0,predictions_orig)
            targets_orig = np.maximum(0,targets_orig)

            predictions_orig_int = predictions_orig.astype(int)
            targets_orig_int = targets_orig.astype(int)
        except ValueError as e:
            print(f"Warning: Scaler inverse transform failed: {e}. Using normalized values for metrics.")
            predictions_orig_int = predictions_all.flatten() # Fallback
            targets_orig_int = targets_all.flatten() # Fallback
        

        val_match_ids = val_df["match_id"].values
        overlap_robust_metric = compute_overlap_robust(targets_orig_int, predictions_orig_int, val_match_ids)
        overlap_metric = compute_overlap_true_test(targets_orig_int, predictions_orig_int, val_match_ids)
        mse_loss = mean_squared_error(targets_orig, predictions_orig) # Use non-int for MSE/MAE
        mae_loss = mean_absolute_error(targets_orig, predictions_orig)

        # Ensure 'predicted_points' column exists before calling compute_loss
        val_metrics_df = val_df[["match_id", "fantasy_points"]].copy()
        val_metrics_df['predicted_points'] = predictions_orig_int
        MAE_custom, MAPE_custom = compute_loss(val_metrics_df) # Use the df with predictions

        # Log metrics manually using writer.add_scalar is NO LONGER needed here.
        # The TensorboardLoggerCallback logs metrics reported via session.report()

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val MSE: {mse_loss:.4f} | Val MAE: {mae_loss:.4f} | Val Overlap: {overlap_robust_metric:.4f} | Val MAPE: {MAPE_custom:.4f}"
        )

        # ----- Reporting Metrics and Checkpointing (using session API) -----
        metrics_to_report = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mse": mse_loss,
            "val_mae": mae_loss,
            "val_overlap": overlap_metric,
            "val_overlap_robust": overlap_robust_metric,
            "val_mape": MAPE_custom,
            "epoch": epoch # Include epoch if useful
        }

        # Create checkpoint data
        checkpoint_data = {
             "epoch": epoch,
             "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             # Optionally save scalers with checkpoint if needed for resuming/inference later
             # "scaler_x_state": scaler_X,
             # "scaler_y_state": scaler_y
         }

        # Save checkpoint data to a temporary directory first
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                checkpoint_data,
                os.path.join(temp_checkpoint_dir, "checkpoint.pt"),
            )
            # Create Checkpoint object from the temporary directory
            checkpoint_obj = Checkpoint.from_directory(temp_checkpoint_dir)

            # Report metrics and checkpoint to Ray Tune session
            session.report(metrics=metrics_to_report, checkpoint=checkpoint_obj)


    print("Finished Training")

    # ----- Final Test Evaluation (Optional) -----
    final_test_metrics = {}
    if do_test and test_loader is not None:
        print("Running final evaluation on test set...")
        model.eval()
        total_test_loss = 0.0
        predictions_list = []
        targets_list = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)
                total_test_loss += loss.item()
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(batch_y.cpu().numpy())
        avg_test_loss = total_test_loss / len(test_loader)

        predictions_all = np.concatenate(predictions_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)
        try:
             predictions_orig = scaler_y.inverse_transform(predictions_all).flatten()
             targets_orig = scaler_y.inverse_transform(targets_all).flatten()
             predictions_orig_int = predictions_orig.astype(int)
             targets_orig_int = targets_orig.astype(int)
        except ValueError as e:
             print(f"Warning: Scaler inverse transform failed for test set: {e}. Using normalized values.")
             predictions_orig_int = predictions_all.flatten()
             targets_orig_int = targets_all.flatten()

        test_mse = mean_squared_error(targets_orig, predictions_orig)
        test_mae = mean_absolute_error(targets_orig, predictions_orig)
        test_match_ids = test_df["match_id"].values
        test_overlap = compute_overlap_true_test(targets_orig_int, predictions_orig_int, test_match_ids)
        test_metrics_df = test_df[["match_id", "fantasy_points"]].copy()
        test_metrics_df['predicted_points'] = predictions_orig_int
        test_MAE_custom, test_MAPE_custom = compute_loss(test_metrics_df)

        print(f"Final Test Loss (Norm): {avg_test_loss:.4f} | Test MSE: {test_mse:.4f} | Test MAE: {test_mae:.4f} | Test Overlap: {test_overlap:.4f} | Test MAPE: {test_MAPE_custom:.4f}")

        # You can report final test metrics ONCE after the loop if needed
        # These won't be used for hyperparameter tuning but stored in the trial results.
        final_test_metrics = {
            "final_test_loss_norm": avg_test_loss,
            "final_test_mse": test_mse,
            "final_test_mae": test_mae,
            "final_test_overlap": test_overlap,
            "final_test_mape": test_MAPE_custom
        }
        # Report final metrics (optional, useful for analysis)
        session.report(metrics=final_test_metrics)

    # Save final model (optional, outside the loop)
    # Consider saving within the final checkpoint or using Ray Tune's artifact logging
    # For simplicity, saving it like before but be aware it might be overwritten by parallel trials if using the same base name.
    # A better approach is to use the best_trial results later to load the best checkpoint and save the model from there.
    final_model_path = f"{artifact_base_path}_final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # writer.close() # No longer needed


# ===== Ray Tune Execution =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for the CustomMLPModel")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Full path to the CSV data file (e.g., /path/to/data/processed/combined/7_final.csv)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of hyperparameter configurations to try.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for each trial.")
    parser.add_argument("--cpus_per_trial", type=int, default=2, help="Number of CPUs per trial.")
    parser.add_argument("--gpus_per_trial", type=int, default=1, help="Number of GPUs per trial.")
    parser.add_argument("--run_test_set", default = False, help="Run evaluation on the test set after training each trial.")
    parser.add_argument("--exp_name", type=str, default="MLP_Hyperparam_Tuning", help="Name for the Ray Tune experiment.")
    parser.add_argument("--storage_path", type=str, default="~/ray_results", help="Root directory for Ray Tune results.")

    args = parser.parse_args()
    args.exp_name = args.exp_name + "_" +  time.strftime("%Y%m%d-%H%M%S") # Append timestamp to experiment name 
    if not ray.is_initialized():
         # Adjust num_gpus based on your system availability if needed
         ray.init(num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)

    # Define search space
    config = {
        "data_file": os.path.abspath(os.path.expanduser(args.data_file)), # Use absolute path
        "epochs": args.epochs, # Fixed number of epochs from args
        "batch_size": tune.choice([256,512, 1024, 2048]),
        # "batch_size": tune.choice([2048]),
        "hidden_layers": tune.choice([[128],[128, 64], [256, 128], [512, 256, 128],[512,256,128,64]]),
        # "hidden_layers" : tune.choice([[256,128]]),
        "activation": tune.choice(["relu", "gelu", "selu"]), # Removed leaky_relu if not needed
        # "activation" : tune.choice(["gelu"]),
        "dropout_prob": tune.uniform(0.1, 0.4),
        # "dropout_prob" : tune.choice([0.3]),
        "weight_init": tune.choice(["xavier", "he", "lecun","default"]),
        # "weight_init" :  tune.choice(["he"]),
        "lr": tune.loguniform(1e-4, 5e-3),
        # "lr": tune.choice([7e-3]),
        "smooth_beta" : tune.choice([10,15,20]),
        # "alpha_loss" : tune.uniform(2,4),
        # # "alpha_loss" : tune.choice([3]),
        # "gamma_loss" : tune.uniform(0.4 , 0.6),
        # # "gamma_loss" : tune.choice([0.5]),
        "model_name": "tuned_mlp", # Consistent prefix
        "do_test": args.run_test_set, # Control test set evaluation via flag
        # "log_dir" key is REMOVED from config
    }

    # Reporter to show progress in console
    reporter = CLIReporter(
        metric_columns=[
            "epoch",
            "train_loss",
            "val_loss",
            "val_mse",
            "val_mae",
            "val_overlap",
            "val_mape",
            "training_iteration" # Ray Tune internal counter
        ],
        parameter_columns=list(config.keys()) # Show relevant config params
    )

    # Run the tuning job
    analysis = tune.run(
        train_mlp_tune,
        metric="val_overlap",    # Primary metric to optimize
        mode="max",           # Set the mode for the metric
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        config=config,
        checkpoint_config = CheckpointConfig(
            checkpoint_score_attribute="val_overlap",
            checkpoint_score_order="max",
            num_to_keep=3, # Keep only the best checkpoint
            # checkpoint_frequency=1, # Save every epoch
        ),
        num_samples=args.num_samples, # Number of trials
        progress_reporter=reporter,
        storage_path=os.path.expanduser(args.storage_path), # Use path from args
        name=args.exp_name,       # Use name from args
        # === Add the TensorboardLoggerCallback ===
        callbacks=[TBXLoggerCallback()],
        max_concurrent_trials=10,
    )

    # Get the best trial based on the validation loss
    best_trial = analysis.get_best_trial("val_loss", "min", "last")

    print("-" * 40)
    print("Hyperparameter Tuning Results")
    print("-" * 40)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']:.4f}")
    if 'val_overlap' in best_trial.last_result:
         print(f"Best trial final validation overlap: {best_trial.last_result['val_overlap']:.4f}")

    ray.shutdown()
    print("Ray shutdown.")