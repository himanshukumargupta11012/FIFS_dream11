#  python mlp_final.py --data_file /home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/processed/combined/10_ipl.csv --epochs 57
import os
import argparse
import pickle
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model_utils_charan import CustomMLPModel
from feature_utils_charan import process, normalise_data

def set_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_hidden_layers(layers_str):
    if not layers_str:
        return []
    try:
        return [int(x.strip()) for x in layers_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid format for hidden_layers string. Use comma-separated integers. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CustomMLPModel with fixed hyperparameters.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Full path to the CSV data file.")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs.")
    args = parser.parse_args()

    # ----- Hardcoded Hyperparameters -----
    output_dir = "./mlp_fixed_output"
    learning_rate = 0.0006
    batch_size = 1024    
    hidden_layers_str = "128"
    activation = "gelu"    
    dropout_prob = 0.1975     
    weight_init = "lecun"       
    loss_type = "mse"     
    # -----------------------------------

    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting training run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {args.data_file}")
    print(f"Epochs: {args.epochs}")
    print("Using hardcoded hyperparameters:")
    print(f"  Output Directory: {output_dir}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Hidden Layers: {hidden_layers_str}")
    print(f"  Activation: {activation}")
    print(f"  Dropout Probability: {dropout_prob}")
    print(f"  Weight Initialization: {weight_init}")
    print(f"  Loss Type: {loss_type}")

    data_file_name = os.path.basename(args.data_file)
    try:
        k = int(data_file_name.split("_")[0])
        print(f"Inferred window size 'k' = {k} from filename.")
    except Exception as e:
        print(f"Warning: Could not infer 'k' from filename '{data_file_name}'. Using k=0 for processing.")
        k = 0

    print(f"Loading data from: {args.data_file}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found at: {args.data_file}")
    combined_df = pd.read_csv(args.data_file)
    combined_df["date"] = pd.to_datetime(combined_df["date"])

    print(f"Total data shape: {combined_df.shape}")
    if combined_df.empty:
        raise ValueError("Input DataFrame is empty.")

    print("Processing features...")
    X_full, cols = process(combined_df, k)
    y_full = combined_df["fantasy_points"].values
    y_full = np.maximum(0, y_full)
    y_full_log = np.log1p(y_full)

    nan_count = np.isnan(X_full).sum()
    inf_count = np.isinf(X_full).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaNs and {inf_count} Infs in processed features.")

    print("Normalizing data...")
    X_full_scaled, y_full_log_scaled, scaler_X, scaler_y = normalise_data(X_full, y_full_log, MinMax=False)

    scalers_dict = {"x": scaler_X, "y": scaler_y}
    scaler_path = os.path.join(output_dir, f"mlp_k{k}_scalers.pkl")
    with open(scaler_path, "wb") as file:
        pickle.dump(scalers_dict, file)
    print(f"Saved scalers to {scaler_path}")

    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full_log_scaled, dtype=torch.float32).view(-1, 1)
    full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Created DataLoader with batch size {batch_size} and {len(full_loader)} batches.")

    num_input_features = X_full.shape[1]
    hidden_layers_list = parse_hidden_layers(hidden_layers_str)
    print(f"Number of input features: {num_input_features}")

    model = CustomMLPModel(
        input_size=num_input_features,
        hidden_layers=hidden_layers_list,
        activation=activation,
        dropout_prob=dropout_prob,
        weight_init=weight_init
    ).to(device)

    if loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif loss_type == "mae":
        loss_fn = nn.L1Loss()
    elif loss_type == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(beta=1.0) # Default beta
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    print(f"Using loss function: {loss_type.upper()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Using Optimizer: Adam (lr={learning_rate})")

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_epoch_loss = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(full_loader):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        avg_epoch_loss = total_epoch_loss / len(full_loader)
        epoch_end_time = time.time()

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_epoch_loss:.4f} | Time: {epoch_end_time - epoch_start_time:.2f}s"
        )

    training_duration = time.time() - start_time
    print(f"\nFinished Training in {training_duration:.2f} seconds.")

    final_model_path = os.path.join(output_dir, f"mlp_k{k}_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    print("Script completed.")