# Description: This script trains the model on the given training data and tests it on the given test data. It then saves the results to a CSV file.
import pandas as pd
from .model_utils import MLPModel, train_model, test_model
import torch
from torch.utils.data import DataLoader, TensorDataset
from .feature_utils import process
import numpy as np
import argparse
import os

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
k = 15

def train_and_test_model(train_start_date, train_end_date, test_start_date, test_end_date, output_csv):

    combined_test_df = pd.DataFrame()

    # separately training and testing for different formats
    for format in ["OD", "T20", "Test"]:
        df = pd.read_csv(f"{current_dir}/../data/processed/combined/5_{format}.csv")
        df["match_id"] = df["match_id"].astype(str)

        train_df = df[(df['date'] >= train_start_date) & (df['date'] <= train_end_date)].copy()
        test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)].copy()

        # saving the training data
        train_df.to_csv(f"{current_dir}/../data/processed/training_data_{train_end_date}_{format}.csv", index=False)

        # Dataset creation
        X_train, _ = process(train_df, k)
        y_train = train_df['fantasy_points'].values
        X_test, _ = process(test_df, k)
        y_test = test_df['fantasy_points'].values

        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Create dataset
        train_dataset = TensorDataset(X_train, y_train_tensor)
        test_dataset = TensorDataset(X_test, y_test_tensor)
        batch_size = 32

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Define the model
        num_features = X_train.shape[1]
        model = MLPModel(num_features, 64)
        model.to(device)
        
        # Train the model
        args = argparse.Namespace(k=k, e=20, batch_size=32, lr=0.001)
        model = train_model(model, train_loader, test_loader, args, format, device=device)
        pred = test_model(model, test_loader, device = device)

        test_df['predicted_points'] = pred

        # Save the model
        torch.save(model.state_dict(), f"{current_dir}/../model_artifacts/model_{train_end_date}_{format}.pth")

        refined_test_df = test_df[['date', 'player', 'player_id', 'team', 'opposition', 'predicted_points', 'fantasy_points', 'match_id']]
        combined_test_df = pd.concat([combined_test_df, refined_test_df])


    # Initialize a list to store the output for each match
    output_rows = []
    combined_test_df["date"] = pd.to_datetime(combined_test_df["date"])
    combined_test_df = combined_test_df.sort_values(by='date')

    # Process each test match (assuming each match has 22 players)
    for _, match_data in combined_test_df.groupby('match_id'):

        # top 11 predicted and actual players
        top_predicted = match_data.sort_values(by='predicted_points', ascending=False).iloc[:11]
        top_actual = match_data.sort_values(by='fantasy_points', ascending=False).iloc[:11]

        # total points for top 11 predicted and actual players
        total_predicted_points = top_predicted['predicted_points'].sum()
        total_actual_points = top_actual['fantasy_points'].sum()

        # Preparing row for the match
        output_data = {
            "Match Date": match_data['date'].iloc[0],  # Assuming same date across match
            "Team": match_data['team'].iloc[0],  # Assuming same team across match
            "Team 2": match_data['opposition'].iloc[0],  # Assuming same team_2 across match
        }
        for i, row in top_predicted.reset_index(drop=True).iterrows():
            output_data[f"Predicted Player {i+1}"] = row['player']
            output_data[f"Predicted Player {i+1} Points"] = row['predicted_points']
        for i, row in top_actual.reset_index(drop=True).iterrows():
            output_data[f"Dream Team Player {i+1}"] = row['player']
            output_data[f"Dream Team Player {i+1} Points"] = row['fantasy_points']

        # Add the total predicted points, total actual points, and MAE
        output_data["Total Points Predicted"] = total_predicted_points
        output_data["Total Dream Team Points"] = total_actual_points
        # MAE
        output_data["Total Points MAE"] = np.abs(total_actual_points - total_predicted_points)

        output_rows.append(output_data)

    # If we have any match results to save
    if output_rows:
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        return output_df.to_json(orient="records")
    else:
        print("No valid matches processed.")
        return None