# train_settransformer_tune_no_role.py

import pandas as pd
import numpy as np
import os
import pickle
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score
# Removed StandardScaler import, using normalize_data
import time
import tempfile
import argparse
import copy

# Corrected Ray Imports
import ray
from ray import tune, air # air contains RunConfig, CheckpointConfig, session
from ray.air import session # Import session explicitly
from ray.tune import CLIReporter, Checkpoint as TuneCheckpoint # Use different alias if needed
from ray.tune.logger import TBXLoggerCallback
from ray.air import CheckpointConfig

# Assuming these are imported
from feature_utils import (
    process,
    compute_overlap_true_test as compute_overlap_standard,
    compute_overlap_segmented,
    normalise_data # Use this function
)
from set_transformer import SetTransformerMultiTask

# --- Configuration ---
DEFAULT_CONFIG = {
    # Data & Setup
    "data_file": "/path/to/your/final_features_with_ORACLE_LABELS.csv", # MUST have 'in_oracle_team'
    "use_log1p_target_points": True,
    "epochs": 50, "batch_size": 32,
    # Model Hyperparameters
    "embedding_dim": tune.choice([64, 128]),
    "num_transformer_blocks": tune.choice([2,3,4]),
    "num_heads": tune.choice([2,4,8]),
    "hidden_dim_transformer_factor": tune.choice([2, 4]),
    "point_head_layers": tune.choice([[64], [128, 64]]),
    "prob_head_layers": tune.choice([[64], [128, 64]]),
    "dropout": tune.uniform(0.1, 0.3),
    # Optimizer
    "lr": tune.loguniform(5e-5, 5e-4),
    "weight_decay": tune.loguniform(1e-6, 1e-3),
    # Loss Weights
    "alpha_loss_points": tune.uniform(0.5, 1.0),
    "beta_loss_prob": tune.uniform(2, 3),
    "points_loss_type": tune.choice(["mse"]),
}

# --- Seeds and Device ---
def set_seeds(seed=0): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seeds()
# Device set inside trainable

# --- Custom Dataset and Collate ---
# (MatchSetDataset and collate_set_batch definitions are unchanged)
class ProcessedMatchSetDataset(Dataset):
    def __init__(self, features_tensor, points_target_tensor, prob_target_tensor,
                 orig_points_array, match_id_array):
        self.features = features_tensor
        self.points_target = points_target_tensor # SCALED points
        self.prob_target = prob_target_tensor   # 0/1 probs
        self.orig_points = orig_points_array    # Original points
        self.match_ids = match_id_array         # Match IDs
        # Group indices by match_id
        self.match_indices_df = pd.DataFrame({'match_id': match_id_array, 'orig_index': np.arange(len(match_id_array))})
        self.match_indices = self.match_indices_df.groupby('match_id')['orig_index'].apply(list).to_dict()
        self.unique_match_ids = list(self.match_indices.keys())
    def __len__(self): return len(self.unique_match_ids)
    def __getitem__(self, idx):
        match_id = self.unique_match_ids[idx]; indices = self.match_indices[match_id]
        return (self.features[indices], self.points_target[indices], self.prob_target[indices], self.orig_points[indices], match_id)

def collate_processed_set_batch(batch):
    features_list, points_target_list, probs_target_list, orig_points_target_list, match_ids = zip(*batch)
    max_len = max(f.shape[0] for f in features_list); num_features = features_list[0].shape[1]
    features_padded_t = torch.zeros((len(batch), max_len, num_features), dtype=torch.float32); points_target_padded_t = torch.zeros((len(batch), max_len, 1), dtype=torch.float32)
    probs_target_padded_t = torch.zeros((len(batch), max_len, 1), dtype=torch.float32); orig_points_target_padded = np.zeros((len(batch), max_len), dtype=np.float32)
    attn_mask_t = torch.ones((len(batch), max_len), dtype=torch.bool) # True = Padding
    match_id_per_item = []
    for i, features in enumerate(features_list):
        length = features.shape[0]; features_padded_t[i, :length, :] = features; points_target_padded_t[i, :length, :] = points_target_list[i]; probs_target_padded_t[i, :length, :] = probs_target_list[i]; orig_points_target_padded[i, :length] = orig_points_target_list[i]; attn_mask_t[i, :length] = False # False = Real data
        match_id_per_item.extend([match_ids[i]] * length)
    collated_batch = {"features": features_padded_t, "points_target": points_target_padded_t, "prob_target": probs_target_padded_t, "attn_mask": attn_mask_t, "orig_points_target_padded": orig_points_target_padded, "match_ids": match_ids, "match_id_per_item": match_id_per_item}
    return collated_batch

# --- Trainable Function ---
def train_settransformer_tune(config):
    # --- Device Setup ---
    use_gpu = config.get("use_gpu", torch.cuda.is_available())
    device = torch.device("cuda:0" if use_gpu else "cpu")
    # device = "cpu"
    print(f"Trial using device: {device}")

    # --- Data Loading & Basic Prep ---
    data_file_path = config["data_file"]; use_log1p_points = config["use_log1p_target_points"]; epochs = config["epochs"]; batch_size = config["batch_size"]; learning_rate = config["lr"]
    if not os.path.exists(data_file_path): raise FileNotFoundError(f"Data file not found: {data_file_path}")
    combined_df = pd.read_csv(data_file_path); combined_df["date"] = pd.to_datetime(combined_df["date"])
    data_file_name = os.path.basename(data_file_path)
    try: k = int(data_file_name.split("_")[0])
    except: k=10; print("Warning: k not found, defaulting to 10")

    # --- Split Data ---
    start_date=pd.to_datetime("2008-04-18"); train_end_date=pd.to_datetime("2023-10-17"); val_end_date=pd.to_datetime("2025-04-04")
    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= train_end_date)].copy()
    val_df = combined_df[(combined_df["date"] > train_end_date) & (combined_df["date"] <= val_end_date)].copy()
    print(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}")
    if train_df.empty or val_df.empty: raise ValueError("Training or Validation DataFrame empty.")

    # --- Process Features AFTER Splitting ---
    print("Processing train features...")
    # process() now returns ONLY numerical base features and their names
    X_train_processed, feature_names = process(train_df, k)
    for col in feature_names : 
        print(col)
    if X_train_processed.dtype == 'object': raise TypeError("process() returned object dtype for train set.")
    num_input_features = X_train_processed.shape[1] # Determine feature dim here
    print(f"Number of processed features: {num_input_features}")

    print("Processing validation features...")
    X_val_processed, _ = process(val_df, k)
    if X_val_processed.dtype == 'object': raise TypeError("process() returned object dtype for validation set.")

    # Check/Impute NaNs/Infs in processed features
    X_train_processed = np.nan_to_num(X_train_processed, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_processed = np.nan_to_num(X_val_processed, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Target Preparation ---
    y_train_points_orig = train_df['fantasy_points'].values
    y_val_points_orig = val_df['fantasy_points'].values
    if use_log1p_points: y_train_target_points = np.log1p(np.maximum(0, y_train_points_orig))
    else: y_train_target_points = y_train_points_orig
    y_train_target_prob = train_df['in_oracle_team'].values.astype(np.float32)
    y_val_target_prob = val_df['in_oracle_team'].values.astype(np.float32)

    # --- Normalize Features and Points Target using normalise_data ---
    print("Scaling features and points target...")
    X_train_scaled, _, scaler_X, _ = normalise_data(X_train_processed, y_train_target_points, MinMax=False) # Scale X
    _, y_train_target_scaled, _, scaler_y_points = normalise_data(X_train_processed, y_train_target_points, MinMax=False) # Scale Y
    X_val_scaled = scaler_X.transform(X_val_processed) # Transform Val X
    # Get scaled Val Y for loss calc during validation
    if use_log1p_points: y_val_target_points = np.log1p(np.maximum(0, y_val_points_orig))
    else: y_val_target_points = y_val_points_orig
    y_val_target_scaled = scaler_y_points.transform(y_val_target_points.reshape(-1, 1)).flatten()


    # --- Create Datasets Directly from Arrays ---
    # Ensure alignment before creating dataset
    # This requires that process() doesn't change row order or drop rows compared to original train_df/val_df
    # If it does, need to merge processed features back based on index before getting other columns
    print("Creating datasets...")
    try:
        train_match_ids_array = train_df["match_id"].values # Assumes alignment
        val_match_ids_array = val_df["match_id"].values   # Assumes alignment

        train_features_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        train_points_target_tensor = torch.tensor(y_train_target_scaled, dtype=torch.float32).unsqueeze(-1)
        train_prob_target_tensor = torch.tensor(y_train_target_prob, dtype=torch.float32).unsqueeze(-1)

        val_features_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        val_points_target_tensor = torch.tensor(y_val_target_scaled, dtype=torch.float32).unsqueeze(-1) # Scaled for Val Loss
        val_prob_target_tensor = torch.tensor(y_val_target_prob, dtype=torch.float32).unsqueeze(-1)


        train_dataset = ProcessedMatchSetDataset(train_features_tensor, train_points_target_tensor, train_prob_target_tensor, y_train_points_orig, train_match_ids_array)
        val_dataset = ProcessedMatchSetDataset(val_features_tensor, val_points_target_tensor, val_prob_target_tensor, y_val_points_orig, val_match_ids_array)
    except IndexError as e:
         raise RuntimeError(f"Index alignment error creating datasets. Does process() modify rows? Error: {e}")


    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_processed_set_batch, num_workers=1, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collate_processed_set_batch, num_workers=1, pin_memory=use_gpu)


    # --- Model, Loss, Optimizer ---
    print(f"Initializing model with {num_input_features} input features.")
    model = SetTransformerMultiTask(input_feature_dim=num_input_features, embed_dim=config["embedding_dim"], num_heads=config["num_heads"], num_transformer_blocks=config["num_transformer_blocks"], hidden_dim_transformer_factor=config["hidden_dim_transformer_factor"], point_head_layers=config["point_head_layers"], prob_head_layers=config["prob_head_layers"], dropout=config["dropout"]).to(device)
    criterion_prob = nn.BCELoss(reduction='none'); alpha = config["alpha_loss_points"]; beta = config["beta_loss_prob"]
    if config["points_loss_type"] == "mse": criterion_points = nn.MSELoss(reduction='none')
    else: criterion_points = nn.L1Loss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # --- Checkpoint loading ---
    # ... (unchanged) ...
    start_epoch = 0; checkpoint = session.get_checkpoint()
    if checkpoint: # Identical
        with checkpoint.as_directory() as ckpt_dir:
            try: ckpt_data = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=device); model.load_state_dict(ckpt_data["model_state_dict"]); optimizer.load_state_dict(ckpt_data["optimizer_state_dict"]); start_epoch = ckpt_data["epoch"] + 1; print(f"Resumed from epoch {start_epoch}")
            except Exception as e: print(f"Error loading ckpt: {e}. Starting fresh.")

    # ----- Training Loop -----
    print(f"Starting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, config["epochs"]):
        # --- Training Step ---
        # ... (unchanged) ...
        model.train(); total_train_loss = 0.0; total_items = 0
        for batch in train_loader: # Use collated batch dictionary
            features = batch["features"].to(device)
            points_target = batch["points_target"].to(device)
            prob_target = batch["prob_target"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            mask = ~attn_mask.unsqueeze(-1)
            optimizer.zero_grad()
            pred_points_scaled, pred_probs = model(features, attn_mask=attn_mask)
            loss_points = criterion_points(pred_points_scaled, points_target)
            loss_points_masked = torch.sum(loss_points * mask) / torch.sum(mask).clamp(min=1)
            loss_prob = criterion_prob(pred_probs, prob_target)
            loss_prob_masked = torch.sum(loss_prob * mask) / torch.sum(mask).clamp(min=1)
            combined_loss = alpha * loss_points_masked + beta * loss_prob_masked
            if torch.isnan(combined_loss): 
                print("NaN loss detected in training!")
                continue
            combined_loss.backward(); optimizer.step(); total_train_loss += combined_loss.item() * len(features); total_items += mask.sum().item()
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # --- Validation Step ---
        # ... (unchanged metric calculation logic) ...
        model.eval(); total_val_loss = 0.0; total_val_batches = 0; val_preds_points_orig_flat = []; val_probs_flat = []; val_targets_points_orig_flat = []; val_targets_prob_flat = []; val_match_ids_flat = []
        with torch.no_grad():
            for batch in val_loader: # Use collated batch dictionary
                features = batch["features"].to(device)
                points_target_scaled_val = batch["points_target"].to(device); prob_target_val = batch["prob_target"].to(device); attn_mask = batch["attn_mask"].to(device); mask = ~attn_mask.unsqueeze(-1); mask_flat = mask.cpu().numpy().flatten().astype(bool)
                pred_points_scaled, pred_probs = model(features, attn_mask=attn_mask); loss_points = criterion_points(pred_points_scaled, points_target_scaled_val); loss_points_masked = torch.sum(loss_points * mask) / torch.sum(mask).clamp(min=1); loss_prob = criterion_prob(pred_probs, prob_target_val); loss_prob_masked = torch.sum(loss_prob * mask) / torch.sum(mask).clamp(min=1); combined_loss = alpha * loss_points_masked + beta * loss_prob_masked; total_val_loss += combined_loss.item(); total_val_batches += 1
                pred_points_scaled_cpu = pred_points_scaled.cpu().numpy()
                pred_points_target_space = scaler_y_points.inverse_transform(pred_points_scaled_cpu.reshape(-1, 1)).flatten()
                if use_log1p_points: pred_points_orig = np.maximum(0, np.expm1(pred_points_target_space))
                else: pred_points_orig = np.maximum(0, pred_points_target_space)
                orig_points_target_np = batch["orig_points_target_padded"]; probs_target_np = prob_target_val.cpu().numpy(); pred_probs_np = pred_probs.cpu().numpy()
                val_preds_points_orig_flat.extend(pred_points_orig[mask_flat]); val_probs_flat.extend(pred_probs_np.flatten()[mask_flat]); val_targets_points_orig_flat.extend(orig_points_target_np.flatten()[mask_flat]); val_targets_prob_flat.extend(probs_target_np.flatten()[mask_flat]); val_match_ids_flat.extend(batch["match_id_per_item"])
        avg_val_loss = total_val_loss / total_val_batches if total_val_batches > 0 else 0
        # Calculate Metrics
        val_overlap_standard = np.nan
        val_overlap_top_half = np.nan
        val_overlap_bottom_half = np.nan
        val_mae = np.nan; val_prob_auc = np.nan
        val_prob_accuracy = np.nan
        try:
            if val_targets_points_orig_flat:
                 targets_points_orig_np = np.array(val_targets_points_orig_flat); preds_points_orig_np = np.array(val_preds_points_orig_flat); targets_points_int = targets_points_orig_np.astype(int); preds_points_int = preds_points_orig_np.astype(int); match_ids_eval = np.array(val_match_ids_flat)
                 val_overlap_standard = compute_overlap_standard(targets_points_int, preds_points_int, match_ids_eval); val_overlap_top_half, val_overlap_bottom_half = compute_overlap_segmented(targets_points_int, preds_points_int, match_ids_eval); val_mae = mean_absolute_error(targets_points_orig_np, preds_points_orig_np)
                 targets_prob_np = np.array(val_targets_prob_flat); preds_probs_np = np.array(val_probs_flat)
                 if len(np.unique(targets_prob_np)) > 1: val_prob_auc = roc_auc_score(targets_prob_np, preds_probs_np)
                 val_prob_accuracy = accuracy_score(targets_prob_np, (preds_probs_np > 0.5).astype(int))
        except Exception as e: print(f"Error during metric calculation: {e}")

        # --- Reporting ---
        # ... (unchanged reporting logic) ...
        print(f"Epoch {epoch}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f} | Val Overlap Std: {val_overlap_standard:.4f} | Top Half: {val_overlap_top_half:.4f} | Btm Half: {val_overlap_bottom_half:.4f} | Val Prob AUC: {val_prob_auc:.4f}")
        metrics_to_report = {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_mae_points": val_mae, "val_overlap": val_overlap_standard, "val_overlap_top_half": val_overlap_top_half, "val_overlap_bottom_half": val_overlap_bottom_half, "val_prob_auc": val_prob_auc, "val_prob_accuracy": val_prob_accuracy,}
        # Checkpointing
        checkpoint_data = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(checkpoint_data, os.path.join(temp_checkpoint_dir, "checkpoint.pt"))
            session.report(metrics=metrics_to_report, checkpoint=TuneCheckpoint.from_directory(temp_checkpoint_dir))


    print("Finished Training Trial.")


# ===== Ray Tune Execution =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Set Transformer Tuning (no role features)")
    # ... (parser arguments unchanged) ...
    parser.add_argument("--data_file", type=str, required=True, help="Path to CSV with 'in_oracle_team'")
    parser.add_argument("--num_samples", type=int, default=20); parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cpus_per_trial", type=int, default=2); parser.add_argument("--gpus_per_trial", type=float, default=1.0)
    parser.add_argument("--exp_name", type=str, default="SetTransformer_MultiTask_NoRole"); parser.add_argument("--storage_path", type=str, default="~/ray_results_settransformer")
    parser.add_argument("--no_log1p_target", action='store_true', help="Disable log1p for points target.")
    parser.add_argument("--optimize_metric", type=str, default="val_overlap", help="Metric to optimize.")
    parser.add_argument("--optimize_mode", type=str, default="max", help="Optimization mode.")
    args = parser.parse_args()

    # Prerequisite Check
    if not os.path.exists(args.data_file): raise FileNotFoundError(f"Data file not found: {args.data_file}")
    try: pd.read_csv(args.data_file, nrows=5)['in_oracle_team']
    except KeyError: raise ValueError("CRITICAL: 'in_oracle_team' column not found.")
    timestamp = time.strftime("%Y%m%d-%H%M%S"); args.exp_name = f"{args.exp_name}_{timestamp}"; num_available_gpus = torch.cuda.device_count()
    if not ray.is_initialized(): ray.init(num_gpus=num_available_gpus, ignore_reinit_error=True, log_to_driver=False)

    # Config Setup
    config = DEFAULT_CONFIG.copy()
    config["data_file"] = os.path.abspath(os.path.expanduser(args.data_file)); config["epochs"] = args.epochs; config["use_log1p_target_points"] = not args.no_log1p_target; config["use_gpu"] = args.gpus_per_trial > 0 and num_available_gpus > 0

    # Reporter
    reporter = CLIReporter(metric_columns=["epoch", "train_loss", "val_loss", "val_mae_points", "val_overlap", "val_overlap_top_half", "val_overlap_bottom_half", "val_prob_auc"])

    # Resource Allocation Dictionary
    resources_per_trial_request = {"cpu": args.cpus_per_trial}
    if config["use_gpu"]: resources_per_trial_request["gpu"] = args.gpus_per_trial
    else: print("Configuring trials for CPU only.")

    # Checkpoint Config (used by tune.run)
    checkpoint_config = CheckpointConfig( # Use air.CheckpointConfig for tune.run
         num_to_keep=3,
         checkpoint_score_attribute=args.optimize_metric,
         checkpoint_score_order=args.optimize_mode, # Mode IS needed here
     )

    # --- Run using tune.run functional API ---
    print("Starting Ray Tune using tune.run API...")
    analysis = tune.run(
        train_settransformer_tune,
        # metric=args.optimize_metric,    # Metric to optimize
        # mode=args.optimize_mode,      # Mode (min/max)
        config=config,                # Hyperparameter search space
        num_samples=args.num_samples, # Number of trials
        resources_per_trial=resources_per_trial_request, # Pass resources directly
        storage_path=os.path.expanduser(args.storage_path),
        name=args.exp_name,
        progress_reporter=reporter,
        callbacks=[TBXLoggerCallback()], # Pass callbacks list
        stop={"training_iteration": args.epochs}, # Stopping criteria
        # Use the checkpoint_config defined above
        checkpoint_config=checkpoint_config, # Pass air.CheckpointConfig
        # local_dir=os.path.expanduser(args.storage_path), # Explicit local dir if needed
        verbose=1 # Tune verbosity (0, 1, 2, 3)
    )

    # --- Get Best Trial from Analysis object ---
    # tune.run returns an ExperimentAnalysis object
    best_trial = analysis.get_best_trial(args.optimize_metric, args.optimize_mode, "last") # Get Trial object

    if not best_trial:
         print("Warning: Could not retrieve best trial information.")
         # Optionally print summary of all trials: print(analysis.dataframe())
    else:
        pass
        # print("-" * 40); print("Set Transformer Tuning Results"); print("-" * 40)
        # print(f"Best trial log directory: {best_trial.logdir}")
        # print(f"Best trial config: {best_trial.config}")
        # print(f"Metric Optimized: {args.optimize_metric} ({args.optimize_mode})")
        # # Access metrics from last_result attribute of the trial object
        # if best_trial.last_result:
        #     print(f"Best trial final {args.optimize_metric}: {best_trial.last_result.get(args.optimize_metric, np.nan):.4f}")
        #     print(f"Best trial final Val Overlap (Top Half): {best_trial.last_result.get('val_overlap_top_half', np.nan):.4f}")
        #     print(f"Best trial final Val Overlap (Bottom Half): {best_trial.last_result.get('val_overlap_bottom_half', np.nan):.4f}")
        #     print(f"Best trial final Val MAE (Points): {best_trial.last_result.get('val_mae_points', np.nan):.4f}")
        #     print(f"Best trial final Val Prob AUC: {best_trial.last_result.get('val_prob_auc', np.nan):.4f}")
        # else:
        #      print("Could not retrieve metrics from best trial's last result.")


    ray.shutdown()
    print("Ray shutdown.")