# python mlp_baseline_final_himanshu.py -f 7_final -e 20 -dim 128 -batch_size 1024 -lr 0.005 -model_name test

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
current_dir = os.path.dirname(os.path.abspath(__file__))

def set_seeds(seed=0):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at the start
set_seeds()

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

    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = pd.to_datetime("2008-04-18")
    split_date = pd.to_datetime("2020-10-17")
    # split_date = pd.Timestamp.today()
    end_date = pd.to_datetime("2025-04-04")

    if split_date >= end_date or split_date >= pd.to_datetime("today"):
        test = False
    else:
        test = True

    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    

    X_train, _ = process(train_df, k)
    X_test, _ = process(test_df, k)

    y_train = train_df["fantasy_points"].values
    y_test = test_df["fantasy_points"].values
    
    X_train, y_train, scaler_X, scaler_y = normalise_data(X_train, y_train, MinMax=False)
    scalers_dict[f"x"] = scaler_X
    scalers_dict[f"y"] = scaler_y

    save_path = f'{current_dir}/../model_artifacts/{args.model_name}_d-{data_file_name}_sd-{start_date.strftime("%Y-%m-%d")}_ed-{split_date.strftime("%Y-%m-%d")}'
    with open(f'{save_path}_scalers.pkl', 'wb') as file:
        pickle.dump(scalers_dict, file)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if test:
        y_test_id = test_df["match_id"].values
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        true_values = y_test
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        test_loader = train_loader
        test_df = train_df
        true_values = y_train
        y_test_id = train_df["match_id"].values

    num_input_features = X_train.shape[1]
    print(num_input_features)
    full_model = MLPModel(layer_sizes=[num_input_features, dim, 1]).to(device)

    model = train_model(full_model, train_loader, test_loader, args, should_save_best_model=False, device=device, loss_fn=nn.MSELoss())
    torch.save(model.state_dict(), f"{save_path}_model.pth")

    loss, test_predictions = evaluate_model(model, test_loader, nn.MSELoss(), device=device, return_predictions=True)
    

    test_predictions_scaled = test_predictions.numpy()

    true_values_original = test_df["fantasy_points"].values
    test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten().astype(int)

    # test_predictions_original = test_predictions_scaled
    mse_loss = mean_squared_error(true_values_original, test_predictions_original)
    mae_loss = mean_absolute_error(true_values_original, test_predictions_original)

    print(f"mse_loss_original_scale : {mse_loss}, mae_loss_original_scale : {mae_loss}")

    results = compute_overlap_true_test(true_values, test_predictions, y_test_id)
    print(f"Average Overlap: {results}")
    test_df.loc[:, "predicted_points"] = test_predictions_original
    MAE, MAPE = compute_loss(test_df[["match_id", "predicted_points", "fantasy_points"]])

    print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")

def main():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")
    parser.add_argument("-f", type=str, required=True, help="File name")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-dim", type=int, default=128, help="MLP layer")
    parser.add_argument("-batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("-lr", type=float, default=0.005, help="batch size")
    parser.add_argument("-model_name", type=str, default="MLP", help="Model name")

    args = parser.parse_args()

    k = args.f.split("_")[0]
    args.k = int(k)

    MLP_train(args)

if __name__ == "__main__":
    main()

