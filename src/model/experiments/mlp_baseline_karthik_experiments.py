# python mlp_baseline_final.py -f 15_ODI -k 15 -e 20 -dim 128 -batch_size 1024 -lr 0.005

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import os
from model_utils_karthik import MLPModel, train_model, test_model, EnsembleModel
import argparse
from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)



# function for training the model
def MLP_train(args):
    data_file_name = args.f
    batch_size = args.batch_size
    k = args.k
    dim = args.dim
    noise_factor = args.noise_factor

    scalers_dict = {}

    print(f" -------------------------------- {data_file_name} ---------------------------------------")
    train_data_path = os.path.join("..", "data", "processed", "train", f"{data_file_name}.csv")
    test_data_path = os.path.join("..", "data", "processed", "test", f"{data_file_name}.csv")

    test_df = pd.read_csv(test_data_path)
    train_df = pd.read_csv(train_data_path)

    y_test_id = test_df["match_id"].values

    X_train, _ = process(train_df, k)
    X_test, _ = process(test_df, k)

    y_train = train_df["fantasy_points"].values
    y_test = test_df["fantasy_points"].values
    
    X_train, y_train, scaler_X, scaler_y = normalise_data(X_train, y_train, MinMax=False)
    
    # Multiple noise augmentation techniques
    if noise_factor > 0:
        # Gaussian noise
        gaussian_noise = torch.normal(mean=0., std=noise_factor, size=X_train.shape)
        
        # Multiplicative noise
        multiplicative_noise = 1 + torch.normal(mean=0., std=noise_factor/2, size=X_train.shape)
        
        # Create augmented dataset
        X_train_augmented = np.vstack([
            X_train,  # Original data
            X_train + gaussian_noise.numpy(),  # Additive noise
            X_train * multiplicative_noise.numpy(),  # Multiplicative noise
            X_train + np.roll(gaussian_noise.numpy(), shift=1, axis=0)  # Temporal noise
        ])
        
        y_train_augmented = np.tile(y_train, 4)
        match_ids_augmented = np.tile(train_df["match_id"].values, 4)  # Also augment match IDs
        
        # Update training data
        X_train = X_train_augmented
        y_train = y_train_augmented
        train_match_ids = match_ids_augmented
    else:
        train_match_ids = train_df["match_id"].values
    
    scalers_dict[f"{data_file_name}_x"] = scaler_X
    scalers_dict[f"{data_file_name}_y"] = scaler_y

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    num_input_features = X_train.shape[1]
    full_model = EnsembleModel(input_features=num_input_features, 
                              hidden_units=dim, 
                              num_models=3).to(device)

    # Include match_ids in the dataset
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    train_ids_tensor = torch.tensor(train_match_ids, dtype=torch.long)  # Use augmented IDs

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    test_ids_tensor = torch.tensor(test_df["match_id"].values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, train_ids_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, test_ids_tensor)
    
    # Group samples by match_id for better batch construction
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # We'll handle shuffling within matches
        collate_fn=lambda x: custom_collate(x, batch_size)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_model(full_model, test_loader, device=device)
    model = train_model(full_model, train_loader, test_loader, args, should_save_best_model=False, device=device)

    test_predictions, uncertainties = test_model(model, test_loader, 
                                              device=device, 
                                              num_mc_samples=args.mc_samples,
                                              noise_factor=args.noise_factor)
    
    # Scale predictions and uncertainties back to original scale
    test_predictions_scaled = test_predictions.detach().numpy()
    uncertainties_scaled = uncertainties.detach().numpy()
    
    true_values = y_test  # Add this line to fix the reference
    true_values_original = test_df["fantasy_points"].values
    test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten().astype(int)
    uncertainties_original = scaler_y.scale_ * uncertainties_scaled
    
    # Add predictions and uncertainties to test_df
    test_df["predicted_points"] = test_predictions_original
    test_df["prediction_uncertainty"] = uncertainties_original
    
    # Print range predictions for a few examples
    for i in range(min(5, len(test_df))):
        pred = test_df["predicted_points"].iloc[i]
        uncert = test_df["prediction_uncertainty"].iloc[i]
        true = test_df["fantasy_points"].iloc[i]
        print(f"True: {true}, Predicted: {pred:.1f} ± {2*uncert:.1f}")
    
    mse_loss = mean_squared_error(true_values_original, test_predictions_original)
    print("mse_loss_original_scale : ", mse_loss)

    # Use scaled values for overlap computation
    results = compute_overlap_true_test(true_values, test_predictions, y_test_id)
    print(f"Average Overlap: {results}")

    MAE, MAPE = compute_loss(test_df[["match_id", "predicted_points", "fantasy_points"]])
    print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")


def custom_collate(batch, batch_size):
    # Group by match_id
    match_groups = {}
    for x, y, match_id in batch:
        if match_id.item() not in match_groups:
            match_groups[match_id.item()] = []
        match_groups[match_id.item()].append((x, y, match_id))
    
    # Create batches that keep players from same match together
    X_batch, y_batch, id_batch = [], [], []
    for match_id, group in match_groups.items():
        for x, y, mid in group:
            X_batch.append(x)
            y_batch.append(y)
            id_batch.append(mid)
            
        # Pad to batch_size if needed
        if len(X_batch) >= batch_size:
            break
    
    return (torch.stack(X_batch), 
            torch.stack(y_batch), 
            torch.stack(id_batch))


def main():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")
    parser.add_argument("-f", type=str, required=True, help="File name")
    parser.add_argument("-k", type=int, required=True, help="The value of k")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-dim", type=int, default=128, help="MLP layer")
    parser.add_argument("-batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")  # Reduced default LR
    parser.add_argument("-noise", "--noise_factor", type=float, default=0.15, 
                      help="Amount of random noise to add during training")
    parser.add_argument("-mc","--mc_samples", type=int, default=10, 
                      help="Number of Monte Carlo samples during inference")
    parser.add_argument("-ni","--noise_inference", type=float, default=0.1,
                      help="Amount of noise to add during inference")
    parser.add_argument("-ensemble", type=int, default=3, help="Number of models in ensemble")
    parser.add_argument("-curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("-ow", "--overlap_weight", type=float, default=0.4,
                      help="Weight for overlap loss term (between 0 and 1)")
    
    args = parser.parse_args()


    MLP_train(args)

if __name__ == "__main__":
    main()

