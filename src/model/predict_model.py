# python predict_model.py --input_path data/processed/testing_data.csv --model_name transformer_transformer_test_d-7_IPL_sd-2010-01-01_ed-2023-10-19

# Description: This script is used to predict the scores of the players in a match.
import torch
import pandas as pd
import sys
import os
import pickle
import argparse
from datetime import datetime, time






current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "data_processing"))
from model_utils import MLPModel, PlayerSelectorTransformer
from predict_utils import get_toss_result, forward_himanshu, forward_charan, teams_shortform, get_players_info, get_shortform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imp_impact_players = ["Shivam Dube", "Ashutosh Sharma", "Karun Nair","Sherfane Rutherford","Prasidh Krishna","Angkrish Raghuvanshi","Ayush Badoni", "Ravi Bishnoi","Rohit Sharma", "Tilak Varma", "Karn Sharma", "Priyansh Arya", "Prabhsimran Singh", "Suyash Sharma", "Devdutt Padikkal", "Vaibhav Suryavanshi", "Kumar Kartikeya", "Rahul Chahar", "Travis Head"]

# ['Royal Challengers Bangalore' 'Kolkata Knight Riders' 'Delhi Daredevils'
#  'Rajasthan Royals' 'Chennai Super Kings' 'Kings XI Punjab'
#  'Deccan Chargers' 'Mumbai Indians' 'Kochi Tuskers Kerala' 'Pune Warriors'
#  'Sunrisers Hyderabad' 'Rising Pune Supergiants' 'Gujarat Lions'
#  'Rising Pune Supergiant' 'Delhi Capitals' 'Punjab Kings'
#  'Lucknow Super Giants' 'Gujarat Titans' 'Royal Challengers Bengaluru']


ipl_matches_info = pd.read_csv(f"{current_dir}/../testing_data/ipl_2025_schedule.csv")
ipl_matches_info["Date"] = pd.to_datetime(ipl_matches_info["Date"])

model_dir = os.path.join(current_dir, "..", "model_artifacts")

squads_players = pd.read_csv(f"{current_dir}/../testing_data/IPL_squad_with_id_old2.csv")
squads_players["key_cricinfo"] = squads_players["key_cricinfo"].astype("Int64")





# Define the function to load classical models
def load_classical_model(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"\nLoaded classical model from {file_path}")
    return data["model"], data["scaler"]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict player scores for a match')
    parser.add_argument('--input_path', type=str, help='Path to the input file containing player data', default=None)
    parser.add_argument('--model_name', type=str, help='Name of the model to be used for prediction')
    parser.add_argument('--ensemble', type=bool, default=False)
    args = parser.parse_args()

    threshold_time = time(18, 0)  # 6 PM

    k = 7
    playing_11 = True

    print(f"Is ensemble: {args.ensemble}")

    today = datetime.today().strftime("%Y-%m-%d")
    date = today
    # date = "2025-04-19"
    
    match_infos = ipl_matches_info[ipl_matches_info["Date"] == date]
    
    if args.input_path is not None:
        # try:
        #     sheet_url = "https://docs.google.com/spreadsheets/d/1AQMIoxTcZnAA9Tyemp3gtUaxy0SBz4_W6RtEQu1MNq0/export?format=csv"
        #     players_df = pd.read_csv(sheet_url)
        # except:
        #     print("Error reading the Google Sheets URL. Please check the URL and try again.")

        players_df = pd.read_csv(f"{current_dir}/../{args.input_path}")
        
        curr_teams_shortform = players_df["Team"].unique()
        team1_cricbuzz = teams_shortform[curr_teams_shortform[0]]["cricbuzz"]
        # match_infos["Team1"] = team1_cricbuzz
        # match_infos["Team2"] = teams_shortform[curr_teams_shortform[1]]["cricbuzz"]
        for _, row in match_infos.iterrows():
            if row["Team1"] == team1_cricbuzz or row["Team2"] == team1_cricbuzz:
                match_info = row
                break
        match_info["teams"] = curr_teams_shortform
        venue = match_info["venue_new"]
        
        toss_result = get_toss_result(match_info)
        print(match_info)
        print("Toss result", toss_result, "\n")

        impact_players_df = players_df[players_df["IsPlaying"].isin(["X_FACTOR_SUBSTITUTE"])]
        players_df = players_df[players_df["IsPlaying"].isin(["PLAYING"])]

        imp_impact_players_df = impact_players_df[impact_players_df["Player Name"].isin(imp_impact_players)]
        selected_impact_players = imp_impact_players_df.groupby('Team').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
        players_df = pd.concat([players_df, selected_impact_players], ignore_index=True)
        print("Selected impact players:\n", selected_impact_players)
        print("total players", len(players_df))


        new_players = players_df[~players_df["Player Name"].isin(squads_players["Player Name"])]
        if new_players.shape[0] > 0:
            print("New players found in the squad", new_players["Player Name"].values)
            new_players_with_id = get_players_info(new_players)
            squads_players = pd.concat([squads_players, new_players_with_id], ignore_index=True)
        
        players_df = pd.merge(players_df, squads_players[["Player Name", "identifier", "name"]], how="inner", on="Player Name")

        print("Player with null identifier:\n", players_df[players_df['identifier'].isna() | (players_df['identifier'] == '')], "\n")
        players_df = players_df[~players_df['identifier'].isna() & (players_df['identifier'] != '')]

    else:
        if match_infos.shape[0] == 2:
            # If there are two matches on the same day, select the match based on the current time
            now = datetime.now().time()
            if now < threshold_time:
                match_info = match_infos.iloc[0].copy()
            else:
                match_info = match_infos.iloc[1].copy()
        else:
            match_info = match_infos.iloc[0].copy()
        match_info["teams"] = [get_shortform(match_info["Team1"], origin="cricbuzz"), get_shortform(match_info["Team2"], origin="cricbuzz")]
        venue = match_info["venue_new"]
        toss_result = get_toss_result(match_info)
        print(match_info)
        print("Toss result", toss_result, "\n")
        players_df = squads_players[squads_players["Team"].isin(match_info["teams"])].copy()
        players_df["lineupOrder"] = 5

        print("Player with null identifier:\n", players_df[players_df['identifier'].isna() | (players_df['identifier'] == '')], "\n")
        players_df = players_df[~players_df['identifier'].isna() & (players_df['identifier'] != '')]

    if args.ensemble:
        forward_charan(date, players_df, venue, toss_result, k, device)
    else:
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

