import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from model_utils import MLPModel, train_model, evaluate_model as test_model, BattingLoss, BowlingLoss, WicketLoss, FieldingLoss
import argparse
from feature_utils import process_batting, compute_overlap_true_test, compute_loss, normalise_data2, process, process_bowling, process_wickets, process_field
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
current_dir = os.getcwd()

def set_seeds(seed=10):
    """Set seeds for reproducibility."""
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at the start
set_seeds()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def MLP_train_bat(args):
    # data_file_name = "../data/interim/ODI_all.csv"
    ouptut_data_path = "../data/interim/ODI_all.csv"
    file_name = "15_ODI"
    batch_size = args.batch_size
    k = args.k
    dim = args.dim
    data_file_name = args.f

    scalers_dict = {}

    print(f" -------------------------------- {file_name} ---------------------------------------")
    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = pd.to_datetime("2010-01-01").strftime("%Y-%m-%d")
    # split_date = pd.to_datetime("2025-02-05")
    split_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = pd.to_datetime("2025-03-05").strftime("%Y-%m-%d")
    

    if split_date >= end_date or split_date >= pd.Timestamp.today().strftime("%Y-%m-%d"):
        test = False
    else:
        test = True

    # print(pd.to_datetime("today"))
    
    print(test)

    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    # train_df = train_df[train_df["Total_matches_played_sum"] > 10]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    # print(test_df.head())
    output_df = pd.read_csv(ouptut_data_path)
    
    output_df = output_df[["match_id", "player_id", "Runs", "Fours", "Sixes", "Balls Faced"]]
    train_df = process_batting(train_df, k, return_tensor=False)
    test_df = process_batting(test_df, k, return_tensor=False)
    print(train_df.shape, test_df.shape)
    
    train_df = pd.merge(train_df, output_df, on=["match_id", "player_id"], how="inner")
    test_df = pd.merge(test_df, output_df, on=["match_id", "player_id"], how="inner")

    y_test_id = test_df["match_id"].values
    X_train = torch.tensor(train_df.drop(columns=["match_id","player_id", "Runs", "Fours", "Sixes", "Balls Faced"]).values, dtype=torch.float32)
    X_test = torch.tensor(test_df.drop(columns=["match_id", "player_id", "Runs", "Fours", "Sixes", "Balls Faced"]).values, dtype=torch.float32)

    y_train = torch.tensor(train_df[["Runs", "Fours", "Sixes", "Balls Faced"]].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[["Runs", "Fours", "Sixes", "Balls Faced"]].values, dtype=torch.float32)
    
    X_train, y_train, scaler_X, scaler_y = normalise_data2(X_train, y_train, MinMax=False)
    scalers_dict[f"{file_name}_x"] = scaler_X
    scalers_dict[f"{file_name}_y"] = scaler_y

    if test:
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

    num_input_features = X_train.shape[1]
    print(num_input_features)
    full_model = MLPModel(layer_sizes=[num_input_features, dim, 4]).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 4)
    # print(X_train_tensor.shape, y_train_tensor.shape)

    if test:    
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 4)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if test:
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if test:    
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print('Initial Loss:',test_model(full_model, train_loader, device=device))
    model = train_model(full_model, train_loader, train_loader, args, should_save_best_model=False, device=device, loss_fn=BattingLoss(penalty_weight=0.5))
    torch.save(model.state_dict(), f"../model_artifacts/{file_name}_bat_model.pth")

    if test:
        score, test_predictions = test_model(model, test_loader, device=device, return_predictions=True)
        print('Final Loss:',test_model(full_model, test_loader, device=device))
        true_values = y_test
        test_predictions_scaled = test_predictions.detach().numpy()

        true_values_original = test_df[["Runs", "Fours", "Sixes", "Balls Faced"]].values
        test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 4)).astype(int)
        # test_predictions_original = test_predictions_scaled
        mse_loss = mean_squared_error(true_values_original, test_predictions_original)

        print("mse_loss_original_scale : ", mse_loss)
        with open(f'../model_artifacts/{file_name}_bat_scalers.pkl', 'wb') as file:
            pickle.dump(scalers_dict, file)
        return test_predictions_original, true_values_original, test_loader
    
def MLP_train_bowl(args):
    # data_file_name = "../data/interim/ODI_all.csv"
    ouptut_data_path = "../data/interim/ODI_all.csv"
    file_name = "15_ODI"
    # input_features_path = "../data/processed/combined/15_ODI.csv"
    train_features_path = "../data/processed/train/15_ODI.csv"
    test_features_path = "../data/processed/test/15_ODI.csv"
    batch_size = args.batch_size
    k = args.k
    dim = args.dim
    data_file_name = args.f

    scalers_dict = {}

    print(f" -------------------------------- {file_name} ---------------------------------------")
    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = pd.to_datetime("2010-01-01").strftime("%Y-%m-%d")
    # split_date = pd.to_datetime("2025-02-05").strftime("%Y-%m-%d")
    split_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = pd.to_datetime("2025-03-05").strftime("%Y-%m-%d")

    if split_date >= end_date or split_date >= pd.Timestamp.today().strftime("%Y-%m-%d"):
        test = False
    else:
        test = True

    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    # train_df = train_df[train_df["Total_matches_played_sum"] > 10]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    # print(test_df.head())
    output_df = pd.read_csv(ouptut_data_path)
    
    output_df = output_df[["match_id", "player_id", "Balls Bowled", "Dot Balls Bowled", "Maiden Overs", "Runsgiven"]]
    train_df = process_bowling(train_df, k, return_tensor=False)
    test_df = process_bowling(test_df, k, return_tensor=False)
    
    
    train_df = pd.merge(train_df, output_df, on=["match_id", "player_id"], how="inner")
    test_df = pd.merge(test_df, output_df, on=["match_id", "player_id"], how="inner")

    y_test_id = test_df["match_id"].values
    X_train = torch.tensor(train_df.drop(columns=["match_id","player_id", "Balls Bowled", "Maiden Overs", "Dot Balls Bowled", "Runsgiven"]).values, dtype=torch.float32)
    X_test = torch.tensor(test_df.drop(columns=["match_id", "player_id", "Balls Bowled", "Maiden Overs", "Dot Balls Bowled", "Runsgiven"]).values, dtype=torch.float32)

    y_train = torch.tensor(train_df[["Balls Bowled", "Maiden Overs", "Dot Balls Bowled", "Runsgiven"]].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[["Balls Bowled", "Maiden Overs", "Dot Balls Bowled", "Runsgiven"]].values, dtype=torch.float32)
    
    X_train, y_train, scaler_X, scaler_y = normalise_data2(X_train, y_train, MinMax=False)
    scalers_dict[f"{file_name}_x"] = scaler_X
    scalers_dict[f"{file_name}_y"] = scaler_y

    if test:
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

    num_input_features = X_train.shape[1]
    print(num_input_features)
    full_model = MLPModel(layer_sizes=[num_input_features, dim, 4]).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 4)
    # print(X_train_tensor.shape, y_train_tensor.shape)

    if test:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 4)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if test:
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if test:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print('Initial Loss:',test_model(full_model, train_loader, device=device))
    model = train_model(full_model, train_loader, train_loader, args, should_save_best_model=False, device=device, loss_fn=BowlingLoss(penalty_weight=0.5))
    torch.save(model.state_dict(), f"../model_artifacts/{file_name}_bowl_model.pth")

    if test:
        score, test_predictions = test_model(model, test_loader, device=device, return_predictions=True)
        print('Final Loss:',test_model(full_model, test_loader, device=device))
        true_values = y_test

        test_predictions_scaled = test_predictions.detach().numpy()

        true_values_original = test_df[["Balls Bowled", "Maiden Overs", "Dot Balls Bowled", "Runsgiven"]].values
        test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 4)).astype(int)
        # test_predictions_original = test_predictions_scaled
        mse_loss = mean_squared_error(true_values_original, test_predictions_original)

        print("mse_loss_original_scale : ", mse_loss)
        with open(f'../model_artifacts/{file_name}_bowl_scalers.pkl', 'wb') as file:
            pickle.dump(scalers_dict, file)
        return test_predictions_original, true_values_original, test_loader
    
def MLP_train_wicket(args):
    # data_file_name = "../data/interim/ODI_all.csv"
    ouptut_data_path = "../data/interim/ODI_all.csv"
    file_name = "15_ODI"
    # input_features_path = "../data/processed/combined/15_ODI.csv"
    train_features_path = "../data/processed/train/15_ODI.csv"
    test_features_path = "../data/processed/test/15_ODI.csv"
    batch_size = args.batch_size
    k = args.k
    dim = args.dim
    data_file_name = args.f

    scalers_dict = {}

    print(f" -------------------------------- {file_name} ---------------------------------------")
    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = pd.to_datetime("2010-01-01").strftime("%Y-%m-%d")
    # split_date = pd.to_datetime("2025-02-05").strftime("%Y-%m-%d")
    split_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = pd.to_datetime("2025-03-05").strftime("%Y-%m-%d")

    if split_date >= end_date or split_date >= pd.Timestamp.today().strftime("%Y-%m-%d"):
        test = False
    else:
        test = True

    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    # train_df = train_df[train_df["Total_matches_played_sum"] > 10]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    # print(test_df.head())
    output_df = pd.read_csv(ouptut_data_path)
    
    output_df = output_df[["match_id", "player_id", "Wickets", "LBWs", "Bowleds"]]
    output_df['Bonus'] = output_df['LBWs'] + output_df['Bowleds']
    train_df = process_wickets(train_df, k, return_tensor=False)
    test_df = process_wickets(test_df, k, return_tensor=False)
    
    
    train_df = pd.merge(train_df, output_df, on=["match_id", "player_id"], how="inner")
    test_df = pd.merge(test_df, output_df, on=["match_id", "player_id"], how="inner")

    y_test_id = test_df["match_id"].values
    X_train = torch.tensor(train_df.drop(columns=["match_id","player_id", "Wickets", "LBWs", "Bowleds" ,"Bonus"]).values, dtype=torch.float32)
    X_test = torch.tensor(test_df.drop(columns=["match_id", "player_id", "Wickets", "LBWs", "Bowleds" ,"Bonus"]).values, dtype=torch.float32)

    y_train = torch.tensor(train_df[["Wickets", "Bonus"]].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[["Wickets", "Bonus"]].values, dtype=torch.float32)
    
    X_train, y_train, scaler_X, scaler_y = normalise_data2(X_train, y_train, MinMax=False)
    scalers_dict[f"{file_name}_x"] = scaler_X
    scalers_dict[f"{file_name}_y"] = scaler_y

    if test:
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

    num_input_features = X_train.shape[1]
    print(num_input_features)
    full_model = MLPModel(layer_sizes=[num_input_features, dim, 2]).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,2)
    # print(X_train_tensor.shape, y_train_tensor.shape)

    if test:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 2)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if test:
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if test:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print('Initial Loss:',test_model(full_model, train_loader, device=device))
    model = train_model(full_model, train_loader, train_loader, args, should_save_best_model=False, device=device, loss_fn=WicketLoss(penalty_weight=0.5))
    torch.save(model.state_dict(), f"../model_artifacts/{file_name}_wick_model.pth")

    if test:
        score, test_predictions = test_model(model, test_loader, device=device, return_predictions=True)
        print('Final Loss:',test_model(full_model, test_loader, device=device))
        true_values = y_test

        test_predictions_scaled = test_predictions.detach().numpy()

        true_values_original = test_df[["Wickets", "Bonus"]].values
        test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 2)).astype(int)
        # test_predictions_original = test_predictions_scaled
        mse_loss = mean_squared_error(true_values_original, test_predictions_original)

        print("mse_loss_original_scale : ", mse_loss)
        with open(f'../model_artifacts/{file_name}_wick_scalers.pkl', 'wb') as file:
            pickle.dump(scalers_dict, file)
        return test_predictions_original, true_values_original, test_loader
    

def MLP_train_field(args):
    # data_file_name = "../data/interim/ODI_all.csv"
    ouptut_data_path = "../data/interim/ODI_all.csv"
    file_name = "15_ODI"
    # input_features_path = "../data/processed/combined/15_ODI.csv"
    train_features_path = "../data/processed/train/15_ODI.csv"
    test_features_path = "../data/processed/test/15_ODI.csv"
    batch_size = args.batch_size
    k = args.k
    dim = args.dim
    data_file_name = args.f

    scalers_dict = {}

    print(f" -------------------------------- {file_name} ---------------------------------------")
    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = pd.to_datetime("2010-01-01").strftime("%Y-%m-%d")
    # split_date = pd.to_datetime("2025-02-05").strftime("%Y-%m-%d")
    split_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = pd.to_datetime("2025-03-05").strftime("%Y-%m-%d")

    if split_date >= end_date or split_date >= pd.Timestamp.today().strftime("%Y-%m-%d"):
        test = False
    else:
        test = True

    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    # train_df = train_df[train_df["Total_matches_played_sum"] > 10]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    # print(test_df.head())
    output_df = pd.read_csv(ouptut_data_path)
    
    output_df = output_df[["match_id", "player_id", "Catches", "Stumpings", "direct run_outs", "indirect run_outs"]]
    train_df = process_field(train_df, k, return_tensor=False)
    test_df = process_field(test_df, k, return_tensor=False)
    
    
    train_df = pd.merge(train_df, output_df, on=["match_id", "player_id"], how="inner")
    test_df = pd.merge(test_df, output_df, on=["match_id", "player_id"], how="inner")

    y_test_id = test_df["match_id"].values
    X_train = torch.tensor(train_df.drop(columns=["match_id","player_id", "Catches", "Stumpings", "direct run_outs", "indirect run_outs"]).values, dtype=torch.float32)
    X_test = torch.tensor(test_df.drop(columns=["match_id", "player_id", "Catches", "Stumpings", "direct run_outs", "indirect run_outs"]).values, dtype=torch.float32)
    
    y_train = torch.tensor(train_df[["Catches", "Stumpings", "direct run_outs", "indirect run_outs"]].values, dtype=torch.float32)
    y_test = torch.tensor(test_df[["Catches", "Stumpings", "direct run_outs", "indirect run_outs"]].values, dtype=torch.float32)
    
    X_train, y_train, scaler_X, scaler_y = normalise_data2(X_train, y_train, MinMax=False)
    scalers_dict[f"{file_name}_x"] = scaler_X
    scalers_dict[f"{file_name}_y"] = scaler_y

    if test:
        X_test = scaler_X.transform(X_test)
        y_test = scaler_y.transform(y_test)

    num_input_features = X_train.shape[1]
    print(num_input_features)
    full_model = MLPModel(layer_sizes=[num_input_features, dim, 4]).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,4)
    # print(X_train_tensor.shape, y_train_tensor.shape)

    if test:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 4)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if test:
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if test:
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print('Initial Loss:',test_model(full_model, train_loader, device=device))
    model = train_model(full_model, train_loader, train_loader, args, should_save_best_model=False, device=device, loss_fn=FieldingLoss(penalty_weight=0.5))
    torch.save(model.state_dict(), f"../model_artifacts/{file_name}_field_model.pth")

    if test:
        score, test_predictions = test_model(model, test_loader, device=device, return_predictions=True)
        print('Final Loss:',test_model(full_model, test_loader, device=device))
        true_values = y_test

        test_predictions_scaled = test_predictions.detach().numpy()

        true_values_original = test_df[["Catches", "Stumpings", "direct run_outs", "indirect run_outs"]].values
        test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 4)).astype(int)
        # test_predictions_original = test_predictions_scaled
        mse_loss = mean_squared_error(true_values_original, test_predictions_original)

        print("mse_loss_original_scale : ", mse_loss)
        with open(f'../model_artifacts/{file_name}_field_scalers.pkl', 'wb') as file:
            pickle.dump(scalers_dict, file)
        return test_predictions_original, true_values_original, test_loader
    
import pickle
import torch
import numpy as np

def get_batting_points(runs, fours, sixes, balls):
    sr = runs/balls*100
    if sr > 140:
        sr_points = 6
    elif sr > 120:
        sr_points = 4
    elif sr > 100:
        sr_points = 2
    elif sr<50:
        sr_points = -2
    elif sr < 40:
        sr_points = -4
    elif sr < 30:
        sr_points = -6
    else:
        sr_points = 0
    
    if balls < 20:
        sr_points = 0
    
    points = runs + fours*4 + sixes*6 + sr_points
    if runs >= 150:
        points += 24
    elif runs >= 125:
        points += 20
    elif runs >= 100:
        points += 16
    elif runs >= 75:
        points += 12
    elif runs >= 50:
        points += 8
    elif runs >= 25:
        points += 4
    return points

def get_ball_points(balls, maidens, dot_balls ,runs):
    econ = runs/balls*6
    if balls < 30:
        econ_points = 0
    else:
        if econ < 2.5:
            econ_points = 6
        elif econ < 3.5:
            econ_points = 4
        elif econ < 4.5:
            econ_points = 2
        elif econ > 7:
            econ_points = -2
        elif econ > 8:
            econ_points = -4
        elif econ > 9:
            econ_points = -6
        else:
            econ_points = 0
    points = econ_points + maidens*4 + dot_balls//3
    return points

def get_wicket_points(wickets, bonus):
    points = wickets*25 + bonus*8
    if wickets >= 6:
        points += 12
    elif wickets >= 5:
        points += 8
    elif wickets >= 4:
        points += 4
    
    return points
    
def get_fielding_points(catches, stumpings, direct_run_outs, indirect_run_outs):
    points = catches*8 + stumpings*12 + direct_run_outs*12 + indirect_run_outs*6
    if catches >=3:
        points += 4 
    return points

def load_and_test_models(args, device):

    # Load scalers
    with open('../model_artifacts/15_ODI_bat_scalers.pkl', 'rb') as file:
        bat_scalers = pickle.load(file)
    with open('../model_artifacts/15_ODI_bowl_scalers.pkl', 'rb') as file:
        bowl_scalers = pickle.load(file)
    with open('../model_artifacts/15_ODI_wick_scalers.pkl', 'rb') as file:
        wick_scalers = pickle.load(file)
    with open('../model_artifacts/15_ODI_field_scalers.pkl', 'rb') as file:
        field_scalers = pickle.load(file)
    data_file_name = args.f

    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    start_date = pd.to_datetime("2010-01-01").strftime("%Y-%m-%d")
    # split_date = pd.to_datetime("2025-02-05").strftime("%Y-%m-%d")
    split_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = pd.to_datetime("2025-03-05").strftime("%Y-%m-%d")

    if split_date >= end_date or split_date >= pd.Timestamp.today().strftime("%Y-%m-%d"):
        test = False
    else:
        test = True
    
    if test:
        return

    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    # train_df = train_df[train_df["Total_matches_played_sum"] > 10]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    true_values = test_df["fantasy_points"].values
    bat_df = process_batting(test_df, 7, return_tensor=False)
    bowl_df = process_bowling(test_df, 7, return_tensor=False)
    wick_df = process_wickets(test_df, 7, return_tensor=False)
    field_df = process_field(test_df, 7, return_tensor=False)
    y_test_id = test_df["match_id"].values
    
    bat_x_df = bat_df.drop(columns=["match_id", "player_id"])
    bowl_x_df = bowl_df.drop(columns=["match_id", "player_id"])
    wick_x_df = wick_df.drop(columns=["match_id", "player_id"])
    field_x_df = field_df.drop(columns=["match_id", "player_id"])
    
    bat_x = torch.tensor(bat_x_df.values, dtype=torch.float32)
    bowl_x = torch.tensor(bowl_x_df.values, dtype=torch.float32)
    wick_x = torch.tensor(wick_x_df.values, dtype=torch.float32)
    field_x = torch.tensor(field_x_df.values, dtype=torch.float32)
    
    # Load models
    bat_model = MLPModel(layer_sizes=[bat_x.shape[1], args.dim, 4]).to(device)
    bat_model.load_state_dict(torch.load("../model_artifacts/15_ODI_bat_model.pth"))
    bat_model.eval()

    bowl_model = MLPModel(layer_sizes=[bowl_x.shape[1], args.dim, 4]).to(device)
    bowl_model.load_state_dict(torch.load("../model_artifacts/15_ODI_bowl_model.pth"))
    bowl_model.eval()

    wick_model = MLPModel(layer_sizes=[wick_x.shape[1], args.dim, 2]).to(device)
    wick_model.load_state_dict(torch.load("../model_artifacts/15_ODI_wick_model.pth"))
    wick_model.eval()

    field_model = MLPModel(layer_sizes=[field_x.shape[1], args.dim, 4]).to(device)
    field_model.load_state_dict(torch.load("../model_artifacts/15_ODI_field_model.pth"))
    field_model.eval()
    
    # Test models
    def test_model(model, data_loader, scaler_X, device):
        model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                inputs = data
                inputs = inputs.reshape(1,-1)
                inputs = inputs.to(device)
                # Scale the inputs
                inputs = torch.tensor(scaler_X.transform(inputs.cpu().numpy()), dtype=torch.float32).to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions, axis=0)

    bat_predictions = test_model(bat_model, bat_x, bat_scalers['15_ODI_x'], device)
    bowl_predictions = test_model(bowl_model, bowl_x, bowl_scalers['15_ODI_x'], device)
    wick_predictions = test_model(wick_model, wick_x, wick_scalers['15_ODI_x'], device)
    field_predictions = test_model(field_model, field_x, field_scalers['15_ODI_x'], device)

    # Inverse transform predictions
    bat_predictions_original = bat_scalers['15_ODI_y'].inverse_transform(bat_predictions)
    bowl_predictions_original = bowl_scalers['15_ODI_y'].inverse_transform(bowl_predictions)
    wick_predictions_original = wick_scalers['15_ODI_y'].inverse_transform(wick_predictions)
    field_predictions_original = field_scalers['15_ODI_y'].inverse_transform(field_predictions)
    
    bat_points = [get_batting_points(*pred) for pred in bat_predictions_original]
    bowl_points = [get_ball_points(*pred) for pred in bowl_predictions_original]
    wick_points = [get_wicket_points(*pred) for pred in wick_predictions_original]
    field_points = [get_fielding_points(*pred) for pred in field_predictions_original]
    
    test_predictions = np.array(bat_points) + np.array(bowl_points) + np.array(wick_points) + np.array(field_points)

    results = compute_overlap_true_test(true_values, test_predictions, y_test_id)
    print(f"Average Overlap: {results}")
    test_df["predicted_points"] = test_predictions
    MAE, MAPE = compute_loss(test_df[["match_id", "predicted_points", "fantasy_points"]])
    print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")
    prediction_list = test_df["predicted_points"].values
    actual_list = test_df["fantasy_points"].values
    MAE = np.mean(np.abs(prediction_list - actual_list))
    MAPE = np.mean(np.abs((prediction_list - actual_list) / actual_list))
    
    

    print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")
    
    return test_df

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

    MLP_train_bat(args)
    MLP_train_bowl(args)
    MLP_train_wicket(args)
    MLP_train_field(args)

    load_and_test_models(args, device)

if __name__ == "__main__":
    main()