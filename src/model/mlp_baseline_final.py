# python mlp_baseline_final.py -f "15_ODI" -k 15 -e 20 -dim 128 -batch_size 1024 -lr 0.005

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import os
from model_utils import MLPModel, train_model, test_model
import argparse
from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



# function for training the model
def MLP_train(args):
    data_file_name = args.f
    batch_size = args.batch_size
    k = args.k
    dim = args.dim

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
    scalers_dict[f"{data_file_name}_x"] = scaler_X
    scalers_dict[f"{data_file_name}_y"] = scaler_y

    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    input_features = X_train.shape[1]
    full_model = MLPModel(input_features=input_features, hidden_units=dim).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    test_model(full_model, test_loader, device=device)
    model = train_model(full_model, train_loader, test_loader, args, should_save_best_model=False,device=device)
    torch.save(model.state_dict(), f"../model_artifacts/{data_file_name}_model.pth")

    test_predictions = test_model(model, test_loader, device=device)
    true_values = y_test

    test_predictions_scaled = test_predictions.detach().numpy()

    true_values_original = test_df["fantasy_points"].values
    test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten().astype(int)
    # test_predictions_original = test_predictions_scaled
    mse_loss = mean_squared_error(true_values_original, test_predictions_original)

    print("mse_loss_original_scale : ", mse_loss)

    results = compute_overlap_true_test(true_values, test_predictions, y_test_id)
    print(f"Average Overlap: {results}")
    test_df["predicted_points"] = test_predictions_original
    MAE, MAPE = compute_loss(test_df[["match_id", "predicted_points", "fantasy_points"]])

    print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")
    
    with open(f'../model_artifacts/{data_file_name}_scalers.pkl', 'wb') as file:
        pickle.dump(scalers_dict, file)



def main():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")
    parser.add_argument("-f", type=str, required=True, help="File name")
    parser.add_argument("-k", type=int, required=True, help="The value of k")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-dim", type=int, default=128, help="MLP layer")
    parser.add_argument("-batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("-lr", type=float, default=0.005, help="batch size")

    args = parser.parse_args()


    MLP_train(args)

if __name__ == "__main__":
    main()

