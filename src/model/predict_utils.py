import requests
from bs4 import BeautifulSoup
import pandas as pd
import torch
from datetime import datetime
import os, sys
from argparse import Namespace
from googlesearch import search

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "data_processing"))
sys.path.append(os.path.join(current_dir, ".."))

from feature_engineering import calculate_player_stats
from feature_engineering_charan import calculate_player_stats as calculate_player_stats_charan
from feature_utils import classification_process
from ensemble_final import load_and_test_model



def get_espncricinfo_link(row):
    if row["Player Name"] == "Andre Siddharth":
        row["Player Name"] = "Andre Siddarth"
    if row["Team"] == "CHE":
        row["Team"] = "CSK"

    query = f"{row['Player Name']} profile ESPNcricinfo {row['Team']}"
    try:
        search_result = search(query, sleep_interval=20, num_results=10, lang="en", timeout=10)
        for link in search_result:
            if "espncricinfo.com" in link:
                return link
            
        print(f"Could not find ESPNcricinfo link for {row['Player Name']}.")
        return None

    except Exception as e:
        print(f"Error searching for {row['Player Name']}: {e}. Please try again.")
        return None
    
def get_espncricinfo_number(link):
    return int(link.split("-")[-1]) if link else None

def get_players_info(df, progress=False):
    new_df = df.copy()
    if progress:
        new_df["ESPNcricinfo Link"] = df.progress_apply(get_espncricinfo_link, axis=1)
    else:
        new_df["ESPNcricinfo Link"] = new_df.apply(get_espncricinfo_link, axis=1)

    new_df['key_cricinfo'] = new_df['ESPNcricinfo Link'].apply(get_espncricinfo_number)
    new_df["key_cricinfo"] = new_df["key_cricinfo"].astype("Int64")

    all_players_df = pd.read_csv(f"{current_dir}/../data/raw/cricsheet/people.csv").dropna(subset=["key_cricinfo"])
    players_df = pd.merge(new_df[["Credits", "Player Type", "Player Name", "Team", "key_cricinfo"]], all_players_df[["identifier", "key_cricinfo", "name"]], on="key_cricinfo", how="left")

    return players_df


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

def get_shortform(name, origin="cricbuzz"):
    for short, sources in teams_shortform.items():
        if sources[origin] == name:
            return short


def get_toss_result(match_info):
    link = match_info["link"]
    url = f"https://www.cricbuzz.com/live-cricket-scorecard/{link}"
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
    

def knapsack(players, budget):
    """
    Solve the knapsack problem to maximize fantasy points while staying within the Credits limit.
    """
    n = len(players)
    dp = [[0] * (int(budget * 2) + 1) for _ in range(n + 1)]
    selected = [[[] for _ in range(int(budget * 2) + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        player = players.iloc[i - 1]
        cost, value = int(player['Credits'] * 2), player['fantasy_points']
        
        for j in range(int(budget * 2) + 1):
            if cost > j:
                dp[i][j] = dp[i - 1][j]
                selected[i][j] = selected[i - 1][j]
            else:
                if dp[i - 1][j] > dp[i - 1][j - cost] + value:
                    dp[i][j] = dp[i - 1][j]
                    selected[i][j] = selected[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j - cost] + value
                    selected[i][j] = selected[i - 1][j - cost] + [player['identifier']]
    return selected[n][int(budget * 2)]  # Return list of selected identifiers


def select_team(csv_file, optional_df, feature1):
    """
    The columns of optional_df are expected to be
    'identifier', 'Player Name', 'Player Type', 'Team', 'fantasy_points', 'Credits', 'feature1'
    """
    if optional_df is None:
        df = pd.read_csv(csv_file)
    else:
        df = optional_df

    # Step 1: Select top players by type with tie-breaker using 'feature1'
    selected_players = []
    remaining_players = df.copy()
    total_credits_used = 0

    for player_type in ['BAT', 'BOWL', 'WK', 'ALL']:
        candidates = df[df['Player Type'] == player_type]
        if not candidates.empty:
            top_player = candidates.sort_values(by=['fantasy_points', feature1], ascending=False).head(1)
            selected_players.append(top_player.iloc[0].to_dict())
            remaining_players = remaining_players[remaining_players['identifier'] != top_player.iloc[0]['identifier']]
            total_credits_used += top_player.iloc[0]['Credits']

    # Step 2: Use Knapsack for the remaining 7 players
    budget_left = 100 - total_credits_used
    knapsack_selected_IDS = knapsack(remaining_players, budget_left)
    knapsack_selected = remaining_players[remaining_players['identifier'].isin(knapsack_selected_IDS)]

    # Tie-break in knapsack selection
    knapsack_selected = knapsack_selected.sort_values(by=['fantasy_points', feature1], ascending=False).head(7)

    selected_players.extend(knapsack_selected.to_dict('records'))

    # Step 3: Ensure at least one player from each team
    selected_df = pd.DataFrame(selected_players)
    teams_covered = set(selected_df['Team'])
    missing_teams = set(df['Team']) - teams_covered

    for team in missing_teams:
        team_players = df[df['Team'] == team]
        worst_player = selected_df.nsmallest(1, 'fantasy_points')  # or customize to remove specific role
        remaining_credits = 100 - selected_df['Credits'].sum() + worst_player['Credits'].values[0]
        best_player = team_players[team_players['Credits'] <= remaining_credits].sort_values(by=['fantasy_points', feature1], ascending=False).head(1)
        
        selected_df = selected_df[selected_df['identifier'] != worst_player['identifier'].values[0]]
        selected_df = pd.concat([selected_df, best_player], ignore_index=True)

    return selected_df[['identifier', 'Player Type', "Player Name", 'Team', 'fantasy_points', 'Credits', feature1]]


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



# function for predicting the scores and generating explanations
def forward_himanshu(model, scalers_dict, date, players_df, venue, toss_result, k, device):

    interim_csv = f"{current_dir}/../data/interim/ipl/all.csv"

    interim_df = pd.read_csv(interim_csv)
    interim_df["date"] = pd.to_datetime(interim_df["date"])

    match_id = 1234567890123
    date = datetime.strptime(date, "%Y-%m-%d")
    df = pd.DataFrame()
    df["Players"] = players_df["name"]
    df["type"] = "IPL"
    df["player_id"] = players_df["identifier"]
    df["Team"] = players_df["Team"].map(teams_shortform).map(lambda x: x["cricsheet"])
    unique_teams = df['Team'].unique()
    team_opposition_mapping = {unique_teams[0]: unique_teams[1], unique_teams[1]: unique_teams[0]}
    df['Opposition'] = df['Team'].map(team_opposition_mapping)
    df["date"] = date
    df["batting_order"] = players_df["lineupOrder"]
    df["Venue"] = venue
    df["match_id"] = match_id
    df["Inning"] = players_df["Team"].map(toss_result)

    df.to_csv("test.csv", index=False)

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
    players_df["fantasy_points"] = output

    players_df["prev_fantasy_points"] = processed_df2[f"last_{k}_matches_fantasy_points_sum"].values
    team = select_team(None, players_df, "prev_fantasy_points")

    top_players = team.sort_values(by='prev_fantasy_points', ascending=False).head(2)

    # Assign roles
    captain = top_players.iloc[0]
    vice_captain = top_players.iloc[1]


    output_df = team[["Player Name", "Team"]].copy()
    output_df["C/VC"] = "NA"

    # Assign roles
    output_df.loc[output_df["Player Name"] == captain["Player Name"], "C/VC"] = "C"
    output_df.loc[output_df["Player Name"] == vice_captain["Player Name"], "C/VC"] = "VC"

    output_df.to_csv("yorker_yodas_output.csv", index=False)
    print(output_df)




# function for predicting the scores and generating explanations
def forward_charan(date, players_df, venue, toss_result, k, device):

    interim_csv = f"{current_dir}/../data_charan/interim/T20_all.csv"

    interim_df = pd.read_csv(interim_csv)
    interim_df["date"] = pd.to_datetime(interim_df["date"])

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


    args = Namespace(mlp_model_path=f"{current_dir}/../mlp_fixed_output/mlp_k{k}_final_model.pth", lgbm_model_path=f"{current_dir}/../lgbm_fixed_output/lgbm_k{k}_final_model.joblib", scalers_path=f"{current_dir}/../mlp_fixed_output/mlp_k{k}_scalers.pkl", feature_names_path=f"{current_dir}/../lgbm_fixed_output/lgbm_k{k}_feature_names.pkl", k=k, device=device, mlp_weight=0.5, lgbm_weight=0.5)
    output = load_and_test_model(args, processed_df2, device)

    players_df["fantasy_points"] = output

    players_df["prev_fantasy_points"] = processed_df2[f"last_7_matches_fantasy_points_sum"].values

    team = select_team(None, players_df, "prev_fantasy_points")

    top_players = team.sort_values(by='prev_fantasy_points', ascending=False).head(2)

    # Assign roles
    captain = top_players.iloc[0]
    vice_captain = top_players.iloc[1]


    output_df = team[["Player Name", "Team"]].copy()
    output_df["C/VC"] = "NA"

    # Assign roles
    output_df.loc[output_df["Player Name"] == captain["Player Name"], "C/VC"] = "C"
    output_df.loc[output_df["Player Name"] == vice_captain["Player Name"], "C/VC"] = "VC"

    output_df.to_csv("reinforced_strikers_output.csv", index=False)

    print(output_df)