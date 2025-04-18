# python mlp_baseline_final_himanshu.py -f 15_himanshu -k 15 -e 20 -dim 128 -batch_size 1024 -lr 0.005

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from model_utils import MLPModel, train_model, evaluate_model
import argparse
from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from sklearn.preprocessing import StandardScaler
import numpy as np

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import os

# Define your MLP model class
class MLPModel(nn.Module):
    def __init__(self, layer_sizes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Process data (dummy process function, replace with your actual data processing)
def process(df, k):
    # Placeholder function to process data
    return df.drop(columns=["fantasy_points"]), df["fantasy_points"]

# Normalization function (you can use MinMax or StandardScaler)
def normalise_data(X_train, y_train, MinMax=True):
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler() if MinMax else StandardScaler()
    scaler_y = MinMaxScaler() if MinMax else StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    return X_train, y_train, scaler_X, scaler_y

# Define the model function in Pyro
def model(X_data, y_data, model):
    # Priors for weights and biases (Normal distributions with mean 0 and variance 1)
    priors = {
        'fc1.weight': dist.Normal(torch.zeros_like(model.fc1.weight), torch.ones_like(model.fc1.weight)),
        'fc1.bias': dist.Normal(torch.zeros_like(model.fc1.bias), torch.ones_like(model.fc1.bias)),
        'fc2.weight': dist.Normal(torch.zeros_like(model.fc2.weight), torch.ones_like(model.fc2.weight)),
        'fc2.bias': dist.Normal(torch.zeros_like(model.fc2.bias), torch.ones_like(model.fc2.bias))
    }

    # Likelihood (normal distribution for observed data)
    with pyro.plate('data', X_data.size(0)):  # Correctly batched
        hidden = torch.relu(model.fc1(X_data))
        y_hat = model.fc2(hidden)
        # Log probability for the observed data
        obs = pyro.sample('obs', dist.Normal(y_hat, 1.0), obs=y_data)

# Define the guide function in Pyro
def guide(X_data, y_data, model):
    # Variational approximation for posterior (we use normal distribution)
    priors = {
        'fc1.weight': pyro.param('fc1_weight_loc', torch.randn_like(model.fc1.weight)),
        'fc1.bias': pyro.param('fc1_bias_loc', torch.randn_like(model.fc1.bias)),
        'fc2.weight': pyro.param('fc2_weight_loc', torch.randn_like(model.fc2.weight)),
        'fc2.bias': pyro.param('fc2_bias_loc', torch.randn_like(model.fc2.bias))
    }

    # The guide is the variational distribution, we define it to sample from
    with pyro.plate('data', X_data.size(0)):  # Correctly batched
        hidden = torch.relu(model.fc1(X_data))
        y_hat = model.fc2(hidden)
        pyro.sample('obs', dist.Normal(y_hat, 1.0), obs=y_data)

# Function for training the model
def MLP_train(args):
    data_file_name = args.f
    batch_size = args.batch_size
    k = args.k
    dim = args.dim

    scalers_dict = {}

    print(f" -------------------------------- {data_file_name} ---------------------------------------")
    combined_data_path = os.path.join("..", "data", "processed", "combined", f"{data_file_name}.csv")
    
    combined_df = pd.read_csv(combined_data_path)
    combined_df["fantasy_points"] = combined_df["batting_fantasy_points"] + combined_df["bowling_fantasy_points"]
    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = "2010-01-01"
    split_date = "2023-10-05"
    end_date = "2023-11-19"
    
    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)

    y_test_id = test_df["match_id"].values

    X_train, _ = process(train_df, k)
    X_test, _ = process(test_df, k)

    y_train = train_df["fantasy_points"].values
    y_test = test_df["fantasy_points"].values
    
    X_train, y_train, scaler_X, scaler_y = normalise_data(X_train, y_train, MinMax=False)
    scalers_dict[f"x"] = scaler_X
    scalers_dict[f"y"] = scaler_y

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    num_input_features = X_train.shape[1]
    print(num_input_features)
    
    full_model = MLPModel(layer_sizes=[num_input_features, dim, 1]).to(device)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set up optimizer and SVI
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

    # Training loop with SVI
    num_epochs = 500
    for epoch in range(num_epochs):
        loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Perform one step of SVI
            loss += svi.step(X_batch, y_batch, full_model)  # Pass the model to guide and model

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss/len(train_loader)}")

    # Save the trained model's state dict
    torch.save(full_model.state_dict(), f"../model_artifacts/{data_file_name}_bnn_model.pth")

    # Evaluate the model
    loss, test_predictions = evaluate_model(full_model, test_loader, nn.MSELoss(), device=device, return_predictions=True)
    true_values = y_test

    test_predictions_scaled = test_predictions.numpy()

    true_values_original = test_df["fantasy_points"].values
    test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten().astype(int)
    print(test_predictions_original)
    
    mse_loss = mean_squared_error(true_values_original, test_predictions_original)
    mae_loss = mean_absolute_error(true_values_original, test_predictions_original)

    print(f"mse_loss_original_scale : {mse_loss}, mae_loss_original_scale : {mae_loss}")

    results = compute_overlap_true_test(true_values, test_predictions, y_test_id)
    print(f"Average Overlap: {results}")
    test_df.loc[:, "predicted_points"] = test_predictions_original
    MAE, MAPE = compute_loss(test_df[["match_id", "predicted_points", "fantasy_points"]])

    print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")
    
    # Save scalers for future use
    with open(f'../model_artifacts/{data_file_name}_scalers.pkl', 'wb') as file:
        pickle.dump(scalers_dict, file)




def main():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")
    parser.add_argument("-f", type=str, required=True, help="File name")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-dim", type=int, default=128, help="MLP layer")
    parser.add_argument("-batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("-lr", type=float, default=0.005, help="batch size")

    args = parser.parse_args()

    k = args.f.split("_")[0]
    args.k = int(k)

    MLP_train(args)

if __name__ == "__main__":
    main()

