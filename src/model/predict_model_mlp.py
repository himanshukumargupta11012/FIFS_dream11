# Description: This script predicts player scores using the MLP baseline model
import torch
import pandas as pd
import numpy as np
import sys
import os
import pickle
import requests
import random
from bs4 import BeautifulSoup
from datetime import datetime

# Set seeds for reproducibility
def set_seeds(seed=0):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at the start
set_seeds()

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "data_processing"))
from feature_engineering import calculate_player_stats

from model_utils import MLPModel
from feature_utils import process_batting, process_bowling, process_wickets, process_field
from final_selection import select_team

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

teams_shortform = {
    "AUS": "Australia",
    "BAN": "Bangladesh",
    "ENG": "England",
    "IND": "India",
    "NZ": "New Zealand",
    "PAK": "Pakistan",
    "SA": "South Africa",
    "AFG": "Afghanistan"
}
venue_list = ["National Stadium, Karachi", "Dubai International Cricket Stadium", "Gaddafi Stadium, Lahore", "Rawalpindi Cricket Stadium"]
venue_map = {
    "Gaddafi Stadium, Lahore": 2,
    "Dubai International Cricket Stadium, Dubai": 1,
    "National Stadium, Karachi": 0,
    "Rawalpindi Cricket Stadium, Rawalpindi": 3,
    "TBC, TBC": 1
}

CT_matches_info = {
    "2025-02-19": {"venue": 0, "link": "/112395/nz-vs-pak-1st-match-group-a-icc-champions-trophy-2025"},
    "2025-02-20": {"venue": 1, "link": "/112402/ban-vs-ind-2nd-match-group-a-icc-champions-trophy-2025"},
    "2025-02-21": {"venue": 0, "link": "/112409/rsa-vs-afg-3rd-match-group-b-icc-champions-trophy-2025"},
    "2025-02-22": {"venue": 2, "link": "/112413/eng-vs-aus-4th-match-group-b-icc-champions-trophy-2025"},
    "2025-02-23": {"venue": 1, "link": "/112420/pak-vs-ind-5th-match-group-a-icc-champions-trophy-2025"},
    "2025-02-24": {"venue": 3, "link": "/112427/ban-vs-nz-6th-match-group-a-icc-champions-trophy-2025"},
    "2025-02-25": {"venue": 3, "link": "/112430/aus-vs-rsa-7th-match-group-b-icc-champions-trophy-2025"},
    "2025-02-26": {"venue": 2, "link": "/112437/afg-vs-eng-8th-match-group-b-icc-champions-trophy-2025"},
    "2025-02-27": {"venue": 3, "link": "/112441/pak-vs-ban-9th-match-group-a-icc-champions-trophy-2025"},
    "2025-02-28": {"venue": 2, "link": "/112444/afg-vs-aus-10th-match-group-b-icc-champions-trophy-2025"},
    "2025-03-01": {"venue": 0, "link": "/112451/rsa-vs-eng-11th-match-group-b-icc-champions-trophy-2025"},
    "2025-03-02": {"venue": 1, "link": "/112455/nz-vs-ind-12th-match-group-a-icc-champions-trophy-2025"},
    "2025-03-04": {"venue": 1, "link": "/112462/tbc-vs-tbc-1st-semi-final-a1-v-b2-icc-champions-trophy-2025"},
    "2025-03-05": {"venue": 2, "link": "/112465/tbc-vs-tbc-2nd-semi-final-b1-v-a2-icc-champions-trophy-2025"},
    "2025-03-09": {"venue": "TBC", "link": "/112469/tbc-vs-tbc-final-icc-champions-trophy-2025"}
}

model_dir = os.path.join(current_dir, "..", "model_artifacts")
squads_players = pd.read_csv(f"{current_dir}/../data/processed/all_teams_squad_with_id.csv")
squads_players["key_cricinfo"] = squads_players["key_cricinfo"].astype("Int64")
interim_csv = f"{current_dir}/../data/interim/ODI_all.csv"
interim_df = pd.read_csv(interim_csv)
interim_df["date"] = pd.to_datetime(interim_df["date"])


def get_batting_points(runs, fours, sixes, balls):
    """Calculate batting fantasy points"""
    # Handle division by zero
    if balls <= 0:
        sr = 0
    else:
        sr = (runs/balls)*100
        
    if sr > 140:
        sr_points = 6
    elif sr > 120:
        sr_points = 4
    elif sr > 100:
        sr_points = 2
    elif sr < 50:
        sr_points = -2
    elif sr < 40:
        sr_points = -4
    elif sr < 30:
        sr_points = -6
    else:
        sr_points = 0
    
    if balls < 20:
        sr_points = 0
    
    points = runs + fours*1 + sixes*2 + sr_points

    if runs > 1000 or runs < 0:
        print(runs, "runs")
    if fours > 1000 or fours < 0:
        print(fours, "fours")
    if sixes > 1000 or sixes < 0:
        print(sixes, "sixes")
    
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


def get_bowling_points(balls, maidens, dot_balls, runs):
    """Calculate bowling fantasy points"""
    # Handle division by zero
    if balls == 0:
        econ = 0
    else:
        econ = (runs/balls)*6
        
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
    """Calculate wicket-taking fantasy points"""
    points = wickets*25 + bonus*8
    if wickets >= 6:
        points += 12
    elif wickets >= 5:
        points += 8
    elif wickets >= 4:
        points += 4
    
    return points


def get_fielding_points(catches, stumpings, direct_run_outs, indirect_run_outs):
    """Calculate fielding fantasy points"""
    points = catches*8 + stumpings*12 + direct_run_outs*12 + indirect_run_outs*6
    if catches >= 3:
        points += 4 
    return points


def get_toss_result(match_info):
    """Get toss result from Cricbuzz website"""
    link = match_info["link"]
    url = f"https://www.cricbuzz.com/live-cricket-scorecard{link}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        toss_prev_div = soup.find_all("div", string="Toss")
        if len(toss_prev_div) == 1:
            toss_result = toss_prev_div[0].find_next_sibling("div").text.strip()
            winner, choice = toss_result.split(" won the toss and opt to ")
            winner = [k for k in teams_shortform if teams_shortform[k] == winner][0]
            loser = match_info["teams"][0] if winner == match_info["teams"][1] else match_info["teams"][1]
            if choice == "bat":
                return {winner: 0, loser: 1}
            else:
                return {winner: 1, loser: 0}
 
        else:
            print("Can't find toss row in webpage. Using default value")
            return {match_info["teams"][0]: 0, match_info["teams"][1]: 1}
    
    else:
        print(f"Error fetching toss result page: {response.status_code}. Using default value")
        return {match_info["teams"][0]: 0, match_info["teams"][1]: 1}
    

def get_venue(link):
    """Get venue information from Cricbuzz website"""
    url = f"https://www.cricbuzz.com/live-cricket-scorecard{link}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        venue_prev_divs = soup.find_all("div", string="Venue")
        if len(venue_prev_divs) == 1:
            venue = venue_prev_divs[0].find_next_sibling("div").text.strip()
            if venue == "TBC, TBC":
                print("Venue info not available on website. Using default value")
            return venue_map[venue]
        else:
            print("Can't find venue row in webpage. Using default value")
            return 1
    
    else:
        print(f"Error fetching match result page: {response.status_code}. Using default value")
        return 1


def load_mlp_models(model_artifacts_dir="model_artifacts", file_prefix="15_ODI"):
    """Load MLP models and scalers for batting, bowling, wickets, and fielding"""
    bat_model = MLPModel(layer_sizes=[34, 128, 4]).to(device)
    bowl_model = MLPModel(layer_sizes=[36, 128, 4]).to(device)
    wick_model = MLPModel(layer_sizes=[29, 128, 2]).to(device)
    field_model = MLPModel(layer_sizes=[26, 128, 4]).to(device)
    
    # Load model weights
    bat_model.load_state_dict(torch.load(f"{model_artifacts_dir}/{file_prefix}_bat_model.pth", map_location=device))
    bowl_model.load_state_dict(torch.load(f"{model_artifacts_dir}/{file_prefix}_bowl_model.pth", map_location=device))
    wick_model.load_state_dict(torch.load(f"{model_artifacts_dir}/{file_prefix}_wick_model.pth", map_location=device))
    field_model.load_state_dict(torch.load(f"{model_artifacts_dir}/{file_prefix}_field_model.pth", map_location=device))
    
    # Set models to evaluation mode
    bat_model.eval()
    bowl_model.eval()
    wick_model.eval()
    field_model.eval()
    
    # Load scalers
    with open(f'{model_artifacts_dir}/{file_prefix}_bat_scalers.pkl', 'rb') as file:
        bat_scalers = pickle.load(file)
    with open(f'{model_artifacts_dir}/{file_prefix}_bowl_scalers.pkl', 'rb') as file:
        bowl_scalers = pickle.load(file)
    with open(f'{model_artifacts_dir}/{file_prefix}_wick_scalers.pkl', 'rb') as file:
        wick_scalers = pickle.load(file)
    with open(f'{model_artifacts_dir}/{file_prefix}_field_scalers.pkl', 'rb') as file:
        field_scalers = pickle.load(file)
    
    return {
        'bat': {'model': bat_model, 'scalers': bat_scalers},
        'bowl': {'model': bowl_model, 'scalers': bowl_scalers},
        'wick': {'model': wick_model, 'scalers': wick_scalers},
        'field': {'model': field_model, 'scalers': field_scalers}
    }


def predict_with_mlp(df, models, k=7):
    """Make predictions using the MLP models"""
    # Process dataframes
    bat_df = process_batting(df, k, return_tensor=False)
    bowl_df = process_bowling(df, k, return_tensor=False)
    wick_df = process_wickets(df, k, return_tensor=False)
    field_df = process_field(df, k, return_tensor=False)

    # Check for NaN values in each dataframe
    if bat_df.isnull().values.any():
        print("\nRows with NaN values in batting data:")
        print(bat_df[bat_df.isnull().any(axis=1)])
        
    if bowl_df.isnull().values.any():
        print("\nRows with NaN values in bowling data:")
        print(bowl_df[bowl_df.isnull().any(axis=1)])
        
    if wick_df.isnull().values.any():
        print("\nRows with NaN values in wickets data:")
        print(wick_df[wick_df.isnull().any(axis=1)])
        
    if field_df.isnull().values.any():
        print("\nRows with NaN values in fielding data:")
        print(field_df[field_df.isnull().any(axis=1)])
    
    # Extract features
    bat_x = torch.tensor(bat_df.drop(columns=["match_id", "player_id"]).values, dtype=torch.float32)
    bowl_x = torch.tensor(bowl_df.drop(columns=["match_id", "player_id"]).values, dtype=torch.float32)
    wick_x = torch.tensor(wick_df.drop(columns=["match_id", "player_id"]).values, dtype=torch.float32)
    field_x = torch.tensor(field_df.drop(columns=["match_id", "player_id"]).values, dtype=torch.float32)

    bat_x = torch.tensor(models['bat']['scalers'][f'15_ODI_x'].transform(bat_x), dtype=torch.float32).to(device)
    bowl_x = torch.tensor(models['bowl']['scalers'][f'15_ODI_x'].transform(bowl_x), dtype=torch.float32).to(device)
    wick_x = torch.tensor(models['wick']['scalers'][f'15_ODI_x'].transform(wick_x), dtype=torch.float32).to(device)
    field_x = torch.tensor(models['field']['scalers'][f'15_ODI_x'].transform(field_x), dtype=torch.float32).to(device)
    
    # Make predictions
    with torch.no_grad():
        bat_pred = models['bat']['model'](bat_x).cpu().numpy()
        bowl_pred = models['bowl']['model'](bowl_x).cpu().numpy()
        wick_pred = models['wick']['model'](wick_x).cpu().numpy()
        field_pred = models['field']['model'](field_x).cpu().numpy()
    
    # Transform predictions back to original scale
    bat_pred = models['bat']['scalers'][f'15_ODI_y'].inverse_transform(bat_pred).astype(int)
    bowl_pred = models['bowl']['scalers'][f'15_ODI_y'].inverse_transform(bowl_pred).astype(int)
    wick_pred = models['wick']['scalers'][f'15_ODI_y'].inverse_transform(wick_pred).astype(int)
    field_pred = models['field']['scalers'][f'15_ODI_y'].inverse_transform(field_pred).astype(int)
    

    # Print some sample predictions for verification
    print("\nSample predictions (first 3 players):")
    print(f"Batting (runs, fours, sixes, balls): {bat_pred[:3]}")
    print(f"Bowling (balls, maidens, dots, runs): {bowl_pred[:3]}")
    print(f"Wickets (wickets, bonus): {wick_pred[:3]}")
    print(f"Fielding (catches, stumpings, direct RO, indirect RO): {field_pred[:3]}")
    
    # Calculate fantasy points using the defined functions
    print("\nCalculating fantasy points...")
    # Calculate individual component points
    bat_points = np.array([get_batting_points(runs, fours, sixes, balls) 
                           for runs, fours, sixes, balls in bat_pred])
    
    bowl_points = np.array([get_bowling_points(balls, maidens, dots, runs) 
                            for balls, maidens, dots, runs in bowl_pred])
    
    wick_points = np.array([get_wicket_points(wickets, bonus) 
                            for wickets, bonus in wick_pred])
    
    field_points = np.array([get_fielding_points(catches, stumpings, direct, indirect) 
                             for catches, stumpings, direct, indirect in field_pred])
    
    # Sum up total fantasy points
    total_points = bat_points + bowl_points + wick_points + field_points
    
    # Create a results DataFrame with player IDs and detailed stats
    results_df = pd.DataFrame({
        'player_id': bat_df['player_id'],
        'match_id': bat_df['match_id'],
        'fantasy_points': total_points,
        'bat_points': bat_points,
        'bowl_points': bowl_points,
        'wick_points': wick_points,
        'field_points': field_points,
        'runs': bat_pred[:, 0],
        'fours': bat_pred[:, 1],
        'sixes': bat_pred[:, 2],
        'balls_faced': bat_pred[:, 3],
        'balls_bowled': bowl_pred[:, 0],
        'maidens': bowl_pred[:, 1],
        'dot_balls': bowl_pred[:, 2],
        'runs_given': bowl_pred[:, 3],
        'wickets': wick_pred[:, 0],
        'bonus_wickets': wick_pred[:, 1],
        'catches': field_pred[:, 0],
        'stumpings': field_pred[:, 1],
        'direct_runouts': field_pred[:, 2],
        'indirect_runouts': field_pred[:, 3]
    })
    
    # Print summary statistics of fantasy points
    print(f"\nFantasy points summary:")
    print(f"Mean: {results_df['fantasy_points'].mean():.2f}")
    print(f"Max: {results_df['fantasy_points'].max():.2f}")
    print(f"Min: {results_df['fantasy_points'].min():.2f}")
    print(results_df.columns)
    
    return results_df


def forward(date, players_df, venue, toss_result, k=7):
    """Main prediction function"""
    match_id = 1234567890
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    
    # Prepare dataframe
    df = pd.DataFrame()
    df["Players"] = players_df["name"]
    df["player_id"] = players_df["identifier"]
    players_df['player_id'] = players_df['identifier']
    df["Team"] = players_df["Team"].map(teams_shortform)
    unique_teams = df['Team'].unique()
    team_opposition_mapping = {unique_teams[0]: unique_teams[1], unique_teams[1]: unique_teams[0]}
    df['Opposition'] = df['Team'].map(team_opposition_mapping)
    df["date"] = date_obj
    df["Venue"] = venue
    df["match_id"] = match_id
    df["Inning"] = players_df["Team"].map(toss_result)
    
    # Combine with historical data
    interim_df2 = pd.concat([interim_df, df], ignore_index=True)
    processed_df2 = calculate_player_stats(interim_df2, 32, k)
    processed_df2 = processed_df2[processed_df2["match_id"] == match_id]
    
    
    # Load MLP models
    models = load_mlp_models(model_artifacts_dir="../model_artifacts")
    
    # Make predictions
    predictions_df = predict_with_mlp(processed_df2, models, k)
    
    # Merge predictions with player info
    result_df = pd.merge(players_df, predictions_df, on="player_id")
    
    print("\nTop players by fantasy points:")
    print(result_df[["Player Name", "Team", "fantasy_points", "runs", "wickets"]].sort_values(by="fantasy_points", ascending=False).head(15))
    
    # Save detailed predictions
    result_df.to_csv("mlp_predictions_detailed.csv", index=False)
    
    # Generate Dream11 team
    team = select_team("mlp_predictions_detailed.csv")
    output_df = team[["Player Name", "Team"]]
    output_df["C/VC"] = "NA"
    output_df.loc[0, "C/VC"] = "C"
    output_df.loc[1, "C/VC"] = "VC"
    
    print("\nSelected Dream11 Team:")
    print(output_df)
    output_df.to_csv("dream11_team_mlp.csv", index=False)
    
    return result_df


if __name__ == "__main__":
    # Set parameters
    k = 7  # Number of past matches to consider
    playing_11 = True
    test_players_file = "match2_squad.csv"
    
    # Set date for prediction
    today = datetime.today().strftime("%Y-%m-%d")
    date = "2025-02-23"  # You can change this to today for current predictions
    
    match_info = CT_matches_info[date]
    
    # Get venue information
    if match_info["venue"] == "TBC":
        venue = venue_list[get_venue(match_info["link"])]
    else:
        venue = venue_list[match_info["venue"]]
    
    # Get players data
    if playing_11:
        players_df = pd.read_csv(f"{current_dir}/../data/processed/{test_players_file}")
        print(f"Loaded player data from local file: {test_players_file}")
            
        match_info["teams"] = players_df["Team"].unique()
        toss_result = get_toss_result(match_info)
        players_df = players_df[players_df["IsPlaying"] == "PLAYING"]
        
        # Check for new players
        new_players = players_df[~players_df["Player Name"].isin(squads_players["Player Name"])]
        if new_players.shape[0] > 0:
            print("New players found in the squad:", new_players["Player Name"].values)
        
        # Merge with player identifiers
        players_df = pd.merge(players_df, squads_players[["Player Name", "identifier", "name"]], 
                             how="inner", on="Player Name")
    else:
        players_df = squads_players[squads_players["Team"].isin(match_info["teams"])]
    
    print(f"\nPredicting for match on {date}: {match_info['teams'][0]} vs {match_info['teams'][1]}")
    print(f"Venue: {venue}")
    print(f"Toss result: {toss_result}")
    
    # Make predictions
    forward(date, players_df, venue, toss_result, k) 