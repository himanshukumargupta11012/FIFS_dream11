import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from random import sample
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig, CheckpointConfig
from mlp_baseline_final import MLP_train, normalise_data
from model_utils import MLPModel, train_model, test_model
from feature_utils import compute_overlap_true_test
from sklearn.metrics import mean_squared_error


os.environ['TUNE_MAX_PENDING_TRIALS_PG'] = '10'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MLP_train(config):
    k = config["k"]
    dim = config["hidden_units"]
    lr = config["lr"]
    num_epochs = config["epochs"]
    columns = config["columns"]
    train_df = config["train_df"]
    test_df = config["test_df"]

    batch_size = 1024

    game_format = "OD"

    y_test_id = test_df["match_id"]

    X_train = train_df.loc[:, columns].values
    X_test = test_df.loc[:, columns].values

    y_train = train_df["fantasy_points"].values
    y_test = test_df["fantasy_points"].values
    
    X_train, y_train, scaler_X, scaler_y = normalise_data(X_train, y_train, MinMax=False)
    X_test, y_test, _, _ = normalise_data(X_test, y_test, MinMax=False)

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
    model = train_model(full_model, train_loader, test_loader, config, game_format, device=device)

    test_predictions = test_model(model, test_loader, device=device)
    true_values = y_test

    true_values_numpy = true_values
    test_predictions_numpy = test_predictions.detach().numpy()

    true_values_original = scaler_y.inverse_transform(true_values_numpy.reshape(-1, 1)).flatten()
    test_predictions_original = scaler_y.inverse_transform(test_predictions_numpy.reshape(-1, 1)).flatten()

    mse_loss = mean_squared_error(true_values_original, test_predictions_original)
    overlap_score = compute_overlap_true_test(true_values, test_predictions, y_test_id)

    tune.report(val_loss=mse_loss, overlap_score=overlap_score)

def sample_20_additional_features(all_features):
    mandatory_features = [
        'last_15_matches_Fours_sum',
        'last_15_matches_Sixes_sum',
        'last_15_matches_fantasy_points_sum',
        'last_15_matches_Innings Bowled_sum',
        'last_15_matches_Balls Bowled_sum',
        'last_15_matches_Runsgiven_sum',
        'last_15_matches_Dot Balls Bowled_sum',
        'venue_avg_runs',
        'venue_avg_wickets',
        'last_15_matches_opponent_Runs_sum',
        'last_15_matches_venue_Runs_sum',
        'last_15_matches_Wickets_sum',
        'last_15_matches_LBWs_sum',
        'last_15_matches_Maiden Overs_sum',
        'last_15_matches_Stumpings_sum',
        'last_15_matches_Catches_sum',
        'last_15_matches_direct run_outs_sum',
        'last_15_matches_opponent_Wickets_sum',
        'last_15_matches_venue_Wickets_sum',
        'last_15_matches_indirect run_outs_sum',
        'last_15_matches_match_type_Innings Batted_sum',
        'last_15_matches_match_type_Innings Bowled_sum',
        'match_type_total_matches',
        "batting_fantasy_points",
        "bowling_fantasy_points",
        "fielding_fantasy_points",
        'last_15_matches_Venue_Wickets_sum',
        'last_15_matches_Opposition_Innings Bowled_sum',
        'last_15_matches_match_type_Wickets_sum',
        'last_15_matches_Opposition_Wickets_sum',
        'last_15_matches_derived_Economy Rate',
        'last_15_matches_Venue_Innings Bowled_sum',
        'last_15_matches_lbw_bowled_sum',
        'last_15_matches_Bowleds_sum'
    ]
    optional_features = [col for col in all_features if col not in mandatory_features]
    sampled_features = sample(optional_features, 20)
    combined_features = mandatory_features + sampled_features
    return combined_features

def main():
    data_path_train = os.path.join("..", "data", "processed", "train", "15_OD.csv")
    data_path_test = os.path.join("..", "data", "processed", "test", "15_OD.csv")
    df_train= pd.read_csv(data_path_train)
    df_test = pd.read_csv(data_path_test)

    new_df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True).drop(columns=['player','team','opposition','date','venue'])
    new_df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True).drop(columns=['player','team','opposition','date','venue'])
    all_features = list(new_df_train.columns)

    config = {
        "k": 15,
        "hidden_units": tune.choice([64, 128, 256]),
        "lr": 0.005,
        "epochs": 25,
        "columns": tune.sample_from(lambda _: sample_20_additional_features(all_features)),
        "train_df": df_train,
        "test_df": df_test
    }

    scheduler = ASHAScheduler(
        max_t=50,
        grace_period=5,
        reduction_factor=2
    )

    tuner = Tuner(
        MLP_train, 
        param_space=config,
        tune_config=TuneConfig(
            num_samples=1,
            scheduler=scheduler,
            metric="val_loss",
            mode="min",
            max_concurrent_trials=255
        ),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=4,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min"
            )
        )
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config
    best_columns = [col for col, keep in best_config["columns"].items() if keep]

    # Output the best results
    print("Best val_loss: ", best_result.metrics["val_loss"])
    print("Best config: ", best_config)
    print("Best 40 features: ", best_columns)


if __name__ == "__main__":
    main()