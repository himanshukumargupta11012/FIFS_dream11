# python predict_model.py --input_path data/processed/testing_data.csv --model_name transformer_transformer_test_d-7_IPL_sd-2010-01-01_ed-2023-10-19

# Description: This script is used to predict the scores of the players in a match.
import torch
from datetime import datetime
import pandas as pd
import sys
import os
import pickle
import argparse

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "data_processing"))
from model_utils import MLPModel, PlayerSelectorTransformer
from predict_utils import get_toss_result, forward_himanshu, forward_charan, teams_shortform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ['Royal Challengers Bangalore' 'Kolkata Knight Riders' 'Delhi Daredevils'
#  'Rajasthan Royals' 'Chennai Super Kings' 'Kings XI Punjab'
#  'Deccan Chargers' 'Mumbai Indians' 'Kochi Tuskers Kerala' 'Pune Warriors'
#  'Sunrisers Hyderabad' 'Rising Pune Supergiants' 'Gujarat Lions'
#  'Rising Pune Supergiant' 'Delhi Capitals' 'Punjab Kings'
#  'Lucknow Super Giants' 'Gujarat Titans' 'Royal Challengers Bengaluru']


ipl_matches_info = pd.read_csv(f"{current_dir}/../testing_data/ipl_2025_schedule.csv")
ipl_matches_info["Date"] = pd.to_datetime(ipl_matches_info["Date"])

model_dir = os.path.join(current_dir, "..", "model_artifacts")

squads_players = pd.read_csv(f"{current_dir}/../testing_data/IPL_squad_with_id.csv")
squads_players["key_cricinfo"] = squads_players["key_cricinfo"].astype("Int64")





# Define the function to load classical models
def load_classical_model(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"\nLoaded classical model from {file_path}")
    return data["model"], data["scaler"]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict player scores for a match')
    parser.add_argument('--input_path', type=str, help='Path to the input file containing player data', required=True)
    parser.add_argument('--model_name', type=str, help='Name of the model to be used for prediction')
    args = parser.parse_args()

    k = 7
    # test_players_file = "match2_squad.csv"
    playing_11 = True

    today = datetime.today().strftime("%Y-%m-%d")
    date = today
    date = "2025-04-19"
    

    match_infos = ipl_matches_info[ipl_matches_info["Date"] == date]
    
    ensemble = False

    
    if playing_11:
        # try:
        #     sheet_url = "https://docs.google.com/spreadsheets/d/1AQMIoxTcZnAA9Tyemp3gtUaxy0SBz4_W6RtEQu1MNq0/export?format=csv"
        #     players_df = pd.read_csv(sheet_url)
        # except:

        players_df = pd.read_csv(f"{current_dir}/../{args.input_path}")
        # players_df = pd.read_csv(args.input_path)
        
        curr_teams_shortform = players_df["Team"].unique()
        team1_cricbuzz = teams_shortform[curr_teams_shortform[0]]["cricbuzz"]
        for _, row in match_infos.iterrows():
            if row["Team1"] == team1_cricbuzz or row["Team2"] == team1_cricbuzz:
                match_info = row
                break
        match_info["teams"] = curr_teams_shortform
        venue = match_info["venue_new"]
        
        toss_result = get_toss_result(match_info)
        print(match_info)
        print(toss_result)
        players_df = players_df[players_df["IsPlaying"].isin(["PLAYING"])]

        new_players = players_df[~players_df["Player Name"].isin(squads_players["Player Name"])]
        if new_players.shape[0] > 0:
            print("New players found in the squad", new_players["Player Name"].values)
        
        players_df = pd.merge(players_df, squads_players[["Player Name", "identifier", "name"]], how="inner", on="Player Name")
    else:
        players_df = squads_players[squads_players["Team"].isin(match_info["teams"])]

    if ensemble:
        forward_charan(date, players_df, venue, toss_result, k, device)
    else:
        print(players_df.shape)
        # loading the model
        state_dict = torch.load(f"{model_dir}/{args.model_name}_model.pth", weights_only=True, map_location=device)
        num_features = 105
        # num_features = state_dict["model.0.weight"].shape[1]
        # model = MLPModel([num_features, 128, 1])
        model = PlayerSelectorTransformer(embed_dim=num_features, transformer_dim=num_features, num_heads=1, num_layers=2).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        with open(f'{model_dir}/{args.model_name}_scalers.pkl', 'rb') as file:
            scalers_dict = pickle.load(file)
        forward_himanshu(model, scalers_dict, date, players_df, venue, toss_result, k, device)

