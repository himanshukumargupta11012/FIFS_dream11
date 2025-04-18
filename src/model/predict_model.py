# python predict_model.py --input_path data/processed/testing_data.csv --model_name transformer_transformer_test_d-7_IPL_sd-2010-01-01_ed-2023-10-19

# Description: This script is used to predict the scores of the players in a match.
import torch
from datetime import datetime
import pandas as pd
import sys
import os
import pickle
import requests
from bs4 import BeautifulSoup
import numpy as np
import argparse

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "data_processing"))
from feature_engineering import calculate_player_stats
from model_utils import MLPModel, PlayerSelectorTransformer
from feature_utils import process, classification_process
from final_selection import select_team

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ['Royal Challengers Bangalore' 'Kolkata Knight Riders' 'Delhi Daredevils'
#  'Rajasthan Royals' 'Chennai Super Kings' 'Kings XI Punjab'
#  'Deccan Chargers' 'Mumbai Indians' 'Kochi Tuskers Kerala' 'Pune Warriors'
#  'Sunrisers Hyderabad' 'Rising Pune Supergiants' 'Gujarat Lions'
#  'Rising Pune Supergiant' 'Delhi Capitals' 'Punjab Kings'
#  'Lucknow Super Giants' 'Gujarat Titans' 'Royal Challengers Bengaluru']

teams_shortform = {
    "CHE": {
        "cricsheet": "Chennai Super Kings",
        "cricbuzz": "Chennai Super Kings"
    },
    "DC": {
        "cricsheet": "Delhi Capitals",
        "cricbuzz": "Delhi Capitals"
    },
    "GT": {
        "cricsheet": "Gujarat Titans",
        "cricbuzz": "Gujarat Titans"
    },
    "KKR": {
        "cricsheet": "Kolkata Knight Riders",
        "cricbuzz": "Kolkata Knight Riders"
    },
    "LSG": {
        "cricsheet": "Lucknow Super Giants",
        "cricbuzz": "Lucknow Super Giants"
    },
    "MI": {
        "cricsheet": "Mumbai Indians",
        "cricbuzz": "Mumbai Indians"
    },
    "PBKS": {
        "cricsheet": "Kings XI Punjab",
        "cricbuzz": "Punjab Kings"
    },
    "RCB": {
        "cricsheet": "Royal Challengers Bangalore",
        "cricbuzz": "Royal Challengers Bengaluru"
    },
    "RR": {
        "cricsheet": "Rajasthan Royals",
        "cricbuzz": "Rajasthan Royals"
    },
    "SRH": {
        "cricsheet": "Sunrisers Hyderabad",
        "cricbuzz": "Sunrisers Hyderabad"
    }
}

venue_list = ["National Stadium, Karachi", "Dubai International Cricket Stadium", "Gaddafi Stadium, Lahore", "Rawalpindi Cricket Stadium"]

CT_matches_info = {
    # "2025-02-19": {"venue": 0, "link": "/112395/nz-vs-pak-1st-match-group-a-icc-champions-trophy-2025"},
    # "2025-02-20": {"venue": 1, "link": "/112402/ban-vs-ind-2nd-match-group-a-icc-champions-trophy-2025"},
    # "2025-02-21": {"venue": 0, "link": "/112409/rsa-vs-afg-3rd-match-group-b-icc-champions-trophy-2025"},
    # "2025-02-22": {"venue": 2, "link": "/112413/eng-vs-aus-4th-match-group-b-icc-champions-trophy-2025"},
    # "2025-02-23": {"venue": 1, "link": "/112420/pak-vs-ind-5th-match-group-a-icc-champions-trophy-2025"},
    # "2025-02-24": {"venue": 3, "link": "/112427/ban-vs-nz-6th-match-group-a-icc-champions-trophy-2025"},
    # "2025-02-25": {"venue": 3, "link": "/112430/aus-vs-rsa-7th-match-group-b-icc-champions-trophy-2025"},
    # "2025-02-26": {"venue": 2, "link": "/112437/afg-vs-eng-8th-match-group-b-icc-champions-trophy-2025"},
    # "2025-02-27": {"venue": 3, "link": "/112441/pak-vs-ban-9th-match-group-a-icc-champions-trophy-2025"},
    # "2025-02-28": {"venue": 2, "link": "/112444/afg-vs-aus-10th-match-group-b-icc-champions-trophy-2025"},
    # "2025-03-01": {"venue": 0, "link": "/112451/rsa-vs-eng-11th-match-group-b-icc-champions-trophy-2025"},
    # "2025-03-02": {"venue": 1, "link": "/112455/nz-vs-ind-12th-match-group-a-icc-champions-trophy-2025"},
    # "2025-03-04": {"venue": 1, "link": "/112462/tbc-vs-tbc-1st-semi-final-a1-v-b2-icc-champions-trophy-2025"},
    # "2025-03-05": {"venue": 2, "link": "/112465/tbc-vs-tbc-2nd-semi-final-b1-v-a2-icc-champions-trophy-2025"},
    # "2025-03-09": {"venue": "TBC", "link": "/112469/tbc-vs-tbc-final-icc-champions-trophy-2025"}
    "2025-04-18": {"venue": 0, "link": "/115174/rcb-vs-pbks-34th-match-indian-premier-league-2025"}
}

top_players = [
    "Shubman Gill", "Virat Kohli", "Rohit Sharma", "Hardik Pandya", "Kuldeep Yadav", "Mohammed Shami", "Lokesh Rahul",
    "Babar Azam", "Mohammed Rizwan", "Khushdil Shah", "Salman Agha", "Shaheen Afridi", "Naseem Shah",
    "Aiden Markram", "Rassie van der-Dussen", "Marco Jansen", "Kagiso Rabada", "Heinrich Klaasen", "Ryan Rickelton",
    "Marnus Labuschagne", "Travis Head", "Glenn Maxwell", "Alex Carey", "Josh Inglis", "Adam Zampa", "Steven Smith",
    "Jos Buttler", "Joe Root", "Ben Duckett", "Liam Livingstone", "Philip Salt", "Jofra Archer",
    "Mohammad Nabi", "Rashid-Khan", "Rahmat Shah", "Azmatullah Omarzai", "Rahmanullah Gurbaz",
    "Kane Williamson", "Will Young", "Daryl Mitchell", "Rachin Ravindra", "Glenn Phillips", "Tom Latham",
    "Michael Bracewell", "Devon Conway"
]


model_dir = os.path.join(current_dir, "..", "model_artifacts")

squads_players = pd.read_csv(f"{current_dir}/../data/processed/all_teams_squad_with_id.csv")
squads_players["key_cricinfo"] = squads_players["key_cricinfo"].astype("Int64")
interim_csv = f"{current_dir}/../data/interim/T20_all.csv"

interim_df = pd.read_csv(interim_csv)
interim_df["date"] = pd.to_datetime(interim_df["date"])


def get_toss_result(match_info):
    link = match_info["link"]
    url = f"https://www.cricbuzz.com/live-cricket-scorecard{link}"
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        toss_prev_div = soup.find_all("div", string="Toss")
        if len(toss_prev_div) == 1:
            toss_result = toss_prev_div[0].find_next_sibling("div").text.strip()
            winner, choice = toss_result.split(" won the toss and opt to ")
            winner = [k for k in teams_shortform if teams_shortform[k]["cricbuzz"] == winner][0]
            loser = match_info["teams"][0] if winner == match_info["teams"][1] else match_info["teams"][1]
            if choice == "bat":

                return {winner: 0, loser: 1}
            else:
                return {winner: 1, loser: 0}
 
        else:
            print(f"Can't find toss row in webpage. Using default value")
            return {match_info["teams"][0]: 0, match_info["teams"][1]: 1}
    
    else:
        print(f"Error fetching toss result page: {response.status_code}. Using default value")
        return {match_info["teams"][0]: 0, match_info["teams"][1]: 1}


# venue_map = {
#     "Gaddafi Stadium, Lahore": 2,
#     "Dubai International Cricket Stadium, Dubai": 1,
#     "National Stadium, Karachi": 0,
#     "Rawalpindi Cricket Stadium, Rawalpindi": 3,
#     "TBC, TBC": 1
# } 

# def get_venue(link):
#     url = f"https://www.cricbuzz.com/live-cricket-scorecard{link}"
#     response = requests.get(url)

#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
#         venue_prev_divs = soup.find_all("div", string="Venue")
#         if len(venue_prev_divs) == 1:
#             venue = venue_prev_divs[0].find_next_sibling("div").text.strip()
#             if venue == "TBC, TBC":
#                 print("Venue info not available on website. Using default value")
#             return venue_map[venue]
#         else:
#             print(f"Can't find venue row in webpage. Using default value")
#             return 1
    
#     else:
#         print(f"Error fetching match result page: {response.status_code}. Using default value")
#         return 1




# Define the function to load classical models
def load_classical_model(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"\nLoaded classical model from {file_path}")
    return data["model"], data["scaler"]

# function for predicting the scores and generating explanations
def forward(date, players_df, venue, toss_result, k):
    match_id = 1234567890
    date = datetime.strptime(date, "%Y-%m-%d")
    df = pd.DataFrame()
    df["Players"] = players_df["name"]
    df["type"] = "IPL"
    df["player_id"] = players_df["identifier"]
    df["Team"] = players_df["Team"].map(teams_shortform).map(lambda x: x["cricbuzz"])
    unique_teams = df['Team'].unique()
    team_opposition_mapping = {unique_teams[0]: unique_teams[1], unique_teams[1]: unique_teams[0]}
    df['Opposition'] = df['Team'].map(team_opposition_mapping)
    df["date"] = date
    df["batting_order"] = players_df["lineupOrder"]
    df["Venue"] = venue
    df["match_id"] = match_id
    df["Inning"] = players_df["Team"].map(toss_result)


    interim_df2 = pd.concat([interim_df, df], ignore_index=True)

    processed_df2 = calculate_player_stats(interim_df2, 16, k)
    processed_df2 = processed_df2[processed_df2["match_id"] == match_id]

    test_data, columns = classification_process(processed_df2, k, train=False)

    if len(test_data.shape) == 3:
        test_data_2d = test_data.reshape(-1, test_data.shape[-1])
        test_data_2d = scalers_dict["x"].transform(test_data_2d)
        test_data = test_data_2d.reshape(test_data.shape)
    else:
        test_data = scalers_dict["x"].transform(test_data)

    test_data = torch.from_numpy(test_data).float().to(device)
    output = model(test_data).cpu().detach().numpy().flatten()
    print(output)
    players_df["fantasy_points"] = output

    players_df["prev_fantasy_points"] = processed_df2[f"last_{k}_matches_fantasy_points_sum"].values
    team = select_team(None, players_df)
    team = pd.merge(team, players_df[["Player Name", "prev_fantasy_points"]], how="left", on="Player Name")
    curr_top_players = team[team["Player Name"].isin(top_players)]
    
    curr_top_players["prev_fantasy_points_prob"] = curr_top_players["prev_fantasy_points"] / curr_top_players["prev_fantasy_points"].sum()
    selected_player = np.random.choice(curr_top_players["Player Name"], p=curr_top_players["prev_fantasy_points_prob"], size=2, replace=False)

    output_df = team[["Player Name", "Team"]]
    output_df["C/VC"] = "NA"

    if len(curr_top_players) >= 2:
        # c_vc = curr_top_players.sample(n=2).reset_index(drop=True)
        output_df.loc[output_df["Player Name"] == selected_player[0], "C/VC"] = "C"
        output_df.loc[output_df["Player Name"] == selected_player[1], "C/VC"] = "VC"
    else:
        output_df.loc[0, "C/VC"] = "C"
        output_df.loc[1, "C/VC"] = "VC"

    output_df.to_csv("hack-et_keepers_output.csv", index=False)

    print(output_df)


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
    date = "2025-04-18"
    

    match_info = CT_matches_info[date]

    venue = venue_list[match_info["venue"]]

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
    
    if playing_11:
        # try:
        #     sheet_url = "https://docs.google.com/spreadsheets/d/1AQMIoxTcZnAA9Tyemp3gtUaxy0SBz4_W6RtEQu1MNq0/export?format=csv"
        #     players_df = pd.read_csv(sheet_url)
        # except:
        players_df = pd.read_csv(f"{current_dir}/../{args.input_path}")
        match_info["teams"] = players_df["Team"].unique()
        toss_result = get_toss_result(match_info)
        players_df = players_df[players_df["IsPlaying"].isin(["PLAYING", "X_FACTOR_SUBSTITUTE"])]
        print(len(players_df))

        new_players = players_df[~players_df["Player Name"].isin(squads_players["Player Name"])]
        if new_players.shape[0] > 0:
            print("New players found in the squad", new_players["Player Name"].values)
        
        players_df = pd.merge(players_df, squads_players[["Player Name", "identifier", "name"]], how="inner", on="Player Name")
    else:
        players_df = squads_players[squads_players["Team"].isin(match_info["teams"])]

    print(len(players_df))



    forward(date, players_df, venue, toss_result, k)
