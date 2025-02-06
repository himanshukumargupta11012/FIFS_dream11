# python feature_engineering.py --input ../data/interim/Test.csv --output_dir ../data/processed --window 15 --threads 8


import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import argparse
import numpy as np
from tqdm import tqdm

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

FEATURES = [
    "Innings Batted", "Runs", "Fours", "Sixes", "Outs", "Dot Balls", "Balls Faced",
    # "Bowled Outs", "LBW Outs", "Hitwicket Outs", "Caught Outs", "Stumped Outs", "Run Outs",   "Singles"
    # "Caught and Bowled Outs", 
    # "totalstos", "totalstosopp", 
    "Innings Bowled", "Balls Bowled","Wickets", "LBWs", "Bowleds", "Extras", "Maiden Overs", "Runsgiven", "Dot Balls Bowled", "Foursgiven", "Sixesgiven",
    # "Singlesgiven", 
    # "No Balls", "Wides", 
    # "Hitwickets", "Caughts", "Stumpeds", "Caught and Bowleds", 
    
    # "totalstosgiven", "totalstosgivenopp",
    "Stumpings", "Catches", "direct run_outs", "indirect run_outs"
]

specifics_features = ["Runs", "Wickets", "Innings Batted", "Innings Bowled"]
specifics = ["Venue", "Opposition", "match_type"]

cumulative_columns = [
    ("cumulative_derived_Dot Ball%", "cumulative_Dot Balls_sum", "cumulative_Balls Faced_sum"),
    ("cumulative_derived_Batting Strike Rate", "cumulative_Runs_sum", "cumulative_Balls Faced_sum"),
    ("cumulative_derived_Batting Avg", "cumulative_Runs_sum", "cumulative_Outs_sum"),
    ("cumulative_derived_Mean Score", "cumulative_Runs_sum", "cumulative_Innings Batted_sum"),
    ("cumulative_derived_Boundary%", "cumulative_Fours_sum", "cumulative_Sixes_sum", "cumulative_Balls Faced_sum"),
    ("cumulative_derived_Mean Balls Faced", "cumulative_Balls Faced_sum", "cumulative_Innings Batted_sum"),
    ("cumulative_derived_Dismissal Rate", "cumulative_Balls Faced_sum", "cumulative_Outs_sum"),
    ("cumulative_derived_Economy Rate", "cumulative_Runsgiven_sum", "cumulative_Balls Bowled_sum"),
    ("cumulative_derived_Bowling Dot Ball%", "cumulative_Dot Balls Bowled_sum", "cumulative_Balls Bowled_sum"),
    ("cumulative_derived_Boundary Given%", "cumulative_Foursgiven_sum", "cumulative_Sixesgiven_sum", "cumulative_Balls Bowled_sum"),
    ("cumulative_derived_Bowling Avg", "cumulative_Runsgiven_sum", "cumulative_Wickets_sum"),
    ("cumulative_derived_Bowling Strike Rate", "cumulative_Balls Bowled_sum", "cumulative_Wickets_sum")
]
last_k_columns = []

def calculate_opponent_venue_rolling_avgs(player_data, new_player_data, features, specifics, k):
    for specific in specifics:
        specific_group = player_data.groupby(['player', specific])
        new_player_data[f'{specific}_total_matches_sum'] = specific_group["Games"].shift(1).cumsum()

        for feature in features:
            new_player_data[f'cumulative_{specific}_{feature}_sum'] = (specific_group[feature].shift(1).cumsum())

            new_player_data[f'last_{k}_matches_{specific}_{feature}_sum'] = (
                specific_group[feature].apply(lambda x: x.shift(1).rolling(k, min_periods=1).sum())
            ).reset_index(level=[0, 1], drop=True)
   
    return new_player_data



def calculate_player_stats_for_group(player_data, k=10):
    global last_k_columns
    # Initialize cumulative and last k matches columns

    new_player_data = pd.DataFrame()
    new_player_data["player"] = player_data["player"]
    new_player_data["team"] = player_data["Team"]
    new_player_data["opposition"] = player_data["Opposition"]
    new_player_data["date"] = player_data["Date"]
    new_player_data["venue"] = player_data["Venue"]
    new_player_data["match_type"] = player_data["match_type"]
    new_player_data["match_id"] = player_data["match_id"]
    new_player_data["player_id"] = player_data["player_id"]
    new_player_data["fantasy_points"] = player_data["fielding_fantasy_points"] + player_data["batting_fantasy_points"] + player_data["bowling_fantasy_points"]
    # new_player_data["batting_fantasy_points"] = player_data["batting_fantasy_points"]
    # new_player_data["bowling_fantasy_points"] = player_data["bowling_fantasy_points"]
    # new_player_data["fielding_fantasy_points"] = player_data["fielding_fantasy_points"]
    new_player_data["Total_matches_played_sum"] = player_data["Games"].shift(1).cumsum()
    for col in FEATURES:
        shifted_data = player_data[col].shift(1)
        new_player_data[f'cumulative_{col}_sum'] = shifted_data.cumsum()
        new_player_data[f'last_{k}_matches_{col}_sum'] = shifted_data.rolling(k, min_periods=1).sum()
    
    new_player_data[f"last_{k}_matches_lbw_bowled_sum"] = (new_player_data[f"last_{k}_matches_LBWs_sum"] + new_player_data[f"last_{k}_matches_Bowleds_sum"])
    
    new_player_data[f"last_{k}_matches_centuries_sum"] = (player_data["Runs"] >= 100).astype(int).shift(1).rolling(k, min_periods=1).sum()
    new_player_data[f"last_{k}_matches_half_centuries_sum"] = (player_data["Runs"] >= 50).astype(int).shift(1).rolling(k, min_periods=1).sum()

    new_player_data[f"last_{k}_matches_duck_outs_sum"] = ((player_data["Runs"] == 0) & (player_data["Innings Batted"] == 1)).astype(int).shift(1).rolling(10, min_periods=1).sum()
    new_player_data[f"last_{k}_matches_3wickets_sum"] = ((player_data["Wickets"] >= 3) & (player_data["Wickets"] < 4)).astype(int).shift(1).rolling(k, min_periods=1).sum()
    new_player_data[f"last_{k}_matches_4wickets_sum"] = ((player_data["Wickets"] >= 4) & (player_data["Wickets"] < 5)).astype(int).shift(1).rolling(k, min_periods=1).sum()
    new_player_data[f"last_{k}_matches_5wickets_sum"] = (player_data["Wickets"] >= 5).astype(int).shift(1).rolling(k, min_periods=1).sum()

    # Set the date as the index
    player_data['Date'] = pd.to_datetime(player_data['Date'])
    # player_data.set_index('Date', inplace=True)


    for feature in ["Runs", "Wickets"]:
        # Calculate rolling one-year averages for both columns
        # new_player_data[f'last_year_avg_{feature}'] = (
        #     player_data[feature]
        #     .rolling('365D', closed='left', min_periods=1)  # 365-day rolling window
        #     .mean()
        # )

        new_player_data[f'last_year_avg_{feature}'] = player_data.apply(
            lambda row: player_data[
                (player_data['Date'] < row['Date']) & 
                (player_data['Date'] > row['Date'] - pd.DateOffset(years=1))
            ][feature].mean(),
            axis=1
        )

    # Reset the index if needed
    # player_data.reset_index(inplace=True)




    # List to collect new columns
    new_columns_cumulative = []

    # Create the new columns for cumulative data
    for col_name, *columns in cumulative_columns:
        if len(columns) == 2:
            new_columns_cumulative.append((new_player_data[columns[0]] / new_player_data[columns[1]]).rename(col_name))
        elif len(columns) == 3:
            new_columns_cumulative.append(((new_player_data[columns[0]] + new_player_data[columns[1]]) / new_player_data[columns[2]] * 100).rename(col_name))

    # Define the list for the last k columns
    last_k_columns = [
        (f"last_{k}_matches_derived_Dot Ball%", f"last_{k}_matches_Dot Balls_sum", f"last_{k}_matches_Balls Faced_sum"),
        (f"last_{k}_matches_derived_Batting Strike Rate", f"last_{k}_matches_Runs_sum", f"last_{k}_matches_Balls Faced_sum"),
        (f"last_{k}_matches_derived_Batting Avg", f"last_{k}_matches_Runs_sum", f"last_{k}_matches_Outs_sum"),
        (f"last_{k}_matches_derived_Mean Score", f"last_{k}_matches_Runs_sum", f"last_{k}_matches_Innings Batted_sum"),
        (f"last_{k}_matches_derived_Boundary%", f"last_{k}_matches_Fours_sum", f"last_{k}_matches_Sixes_sum", f"last_{k}_matches_Balls Faced_sum"),
        (f"last_{k}_matches_derived_Mean Balls Faced", f"last_{k}_matches_Balls Faced_sum", f"last_{k}_matches_Innings Batted_sum"),
        (f"last_{k}_matches_derived_Dismissal Rate", f"last_{k}_matches_Balls Faced_sum", f"last_{k}_matches_Outs_sum"),
        (f"last_{k}_matches_derived_Economy Rate", f"last_{k}_matches_Runsgiven_sum", f"last_{k}_matches_Balls Bowled_sum"),
        (f"last_{k}_matches_derived_Bowling Dot Ball%", f"last_{k}_matches_Dot Balls Bowled_sum", f"last_{k}_matches_Balls Bowled_sum"),
        (f"last_{k}_matches_derived_Boundary Given%", f"last_{k}_matches_Foursgiven_sum", f"last_{k}_matches_Sixesgiven_sum", f"last_{k}_matches_Balls Bowled_sum"),
        (f"last_{k}_matches_derived_Bowling Avg", f"last_{k}_matches_Runsgiven_sum", f"last_{k}_matches_Wickets_sum"),
        (f"last_{k}_matches_derived_Bowling Strike Rate", f"last_{k}_matches_Balls Bowled_sum", f"last_{k}_matches_Wickets_sum")
    ]

    # List to collect new columns for last k matches data
    new_columns_last_k = []

    # Create the new columns for last k matches data
    for col_name, *columns in last_k_columns:
        if len(columns) == 2:
            new_columns_last_k.append((new_player_data[columns[0]] / new_player_data[columns[1]]).rename(col_name))

        elif len(columns) == 3:
            new_columns_last_k.append(((new_player_data[columns[0]] + new_player_data[columns[1]]) / new_player_data[columns[2]] * 100).rename(col_name))


    # Combine all the new columns (cumulative and last k matches) into the DataFrame at once
    new_player_data = pd.concat([new_player_data] + new_columns_cumulative + new_columns_last_k, axis=1)

    # Create a copy of the updated DataFrame
    new_player_data = new_player_data.copy()

    new_player_data = calculate_opponent_venue_rolling_avgs(player_data, new_player_data, specifics_features, specifics, k)

    return new_player_data

def calculate_player_stats(df, num_threads, k):
    df.rename(columns={df.columns[0]: 'player'}, inplace=True)
    
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        
        groups = df.groupby('player_id')
        results = list(tqdm(executor.map(calculate_player_stats_for_group, [group for _, group in groups], [k] * len(groups)), 
                        total=len(groups), 
                        desc="Processing Players"))

    new_df = pd.concat(results, axis=0)
    new_df = processing(new_df.copy(), k)

    # adding venue stats
    venue_avg_runs = df.groupby("Venue")['Runs'].mean() * 11
    venue_avg_wickets = df.groupby("Venue")['Wickets'].mean() * 11
    new_df["venue_avg_runs_sum"] = new_df["venue"].map(venue_avg_runs)
    new_df["venue_avg_wickets_sum"] = new_df["venue"].map(venue_avg_wickets)

    # adding league stats
    league_avg_runs = df.groupby("match_type")['Runs'].mean() * 11
    league_avg_wickets = df.groupby("match_type")['Wickets'].mean() * 11
    new_df["league_avg_runs_sum"] = new_df["match_type"].map(league_avg_runs)
    new_df["league_avg_wickets_sum"] = new_df["match_type"].map(league_avg_wickets)

    return new_df


def processing(df, k):

    data_start = df.columns.get_loc('Total_matches_played_sum')
    sum_columns = [col for col in df.columns if col.endswith('sum')]
    df[sum_columns] = df[sum_columns].fillna(0)

    # Define the list for the last k columns
    last_k_columns = [
        (f"last_{k}_matches_derived_Dot Ball%", f"last_{k}_matches_Dot Balls_sum", f"last_{k}_matches_Balls Faced_sum"),
        (f"last_{k}_matches_derived_Batting Strike Rate", f"last_{k}_matches_Runs_sum", f"last_{k}_matches_Balls Faced_sum"),
        (f"last_{k}_matches_derived_Batting Avg", f"last_{k}_matches_Runs_sum", f"last_{k}_matches_Outs_sum"),
        (f"last_{k}_matches_derived_Mean Score", f"last_{k}_matches_Runs_sum", f"last_{k}_matches_Innings Batted_sum"),
        (f"last_{k}_matches_derived_Boundary%", f"last_{k}_matches_Fours_sum", f"last_{k}_matches_Sixes_sum", f"last_{k}_matches_Balls Faced_sum"),
        (f"last_{k}_matches_derived_Mean Balls Faced", f"last_{k}_matches_Balls Faced_sum", f"last_{k}_matches_Innings Batted_sum"),
        (f"last_{k}_matches_derived_Dismissal Rate", f"last_{k}_matches_Balls Faced_sum", f"last_{k}_matches_Outs_sum"),
        (f"last_{k}_matches_derived_Economy Rate", f"last_{k}_matches_Runsgiven_sum", f"last_{k}_matches_Balls Bowled_sum"),
        (f"last_{k}_matches_derived_Bowling Dot Ball%", f"last_{k}_matches_Dot Balls Bowled_sum", f"last_{k}_matches_Balls Bowled_sum"),
        (f"last_{k}_matches_derived_Boundary Given%", f"last_{k}_matches_Foursgiven_sum", f"last_{k}_matches_Sixesgiven_sum", f"last_{k}_matches_Balls Bowled_sum"),
        (f"last_{k}_matches_derived_Bowling Avg", f"last_{k}_matches_Runsgiven_sum", f"last_{k}_matches_Wickets_sum"),
        (f"last_{k}_matches_derived_Bowling Strike Rate", f"last_{k}_matches_Balls Bowled_sum", f"last_{k}_matches_Wickets_sum")
    ]

    value = -1
    
    for col_name, *columns in cumulative_columns:
        if len(columns) == 2:
            df.loc[(df[columns[1]] == 0.0) & (df[columns[0]] == 0.0) , col_name] = value
            df.loc[(df[columns[1]] == 0.0) & (df[columns[0]] != 0.0) , col_name] = df[columns[0]] * 1
        elif len(columns) == 3:
            df.loc[df[columns[2]] == 0, col_name] = value

    # Create the new columns for last k matches data
    for col_name, *columns in last_k_columns:
        if len(columns) == 2:
            df.loc[(df[columns[1]] == 0.0) & (df[columns[0]] == 0.0) , col_name] = value
            df.loc[(df[columns[1]] == 0.0) & (df[columns[0]] != 0.0) , col_name] = df[columns[0]] * 1
        elif len(columns) == 3:
            df.loc[df[columns[2]] == 0, col_name] = value

    for param1 in specifics_features:
        for window in ["cumulative", f"last_{k}_matches"]:
            columns = [f"{window}_{specific}_{param1}_sum" for specific in specifics]
            for col in columns:
                df[col] = df[col].fillna(df[f"{window}_{param1}_sum"], axis=0)

    for specific in specifics:
        df[f"{specific}_total_matches_sum"] = df[f"{specific}_total_matches_sum"].fillna(0)

    for feature in ["Runs", "Wickets"]:
        df[f'last_year_avg_{feature}'] = df[f'last_year_avg_{feature}'].fillna(value)

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")

    return df


def main(input_file_path, output_dir, k, num_threads):
    df = pd.read_csv(input_file_path)
    new_df = calculate_player_stats(df, num_threads, k)
    input_file_name = os.path.splitext(os.path.basename(input_file_path))[0] + '.csv'

    train_df = new_df[new_df['date'] <= '2024-06-30']
    test_df = new_df[(new_df['date'] > '2024-06-30') & (new_df['date'] <= '2024-11-10')]    
    
    for subfolder in ['train', 'test', 'test_after10nov', 'combined']:
        os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)

    # features for testing after 10th Nov
    new_df = new_df.sort_values(by=['player_id', 'date'], ascending=[True, False])
    # latest_matches_df = new_df.drop_duplicates(subset='player_id', keep='first').reset_index(drop=True)

    train_df.to_csv(os.path.join(output_dir, 'train',str(k) + "_" + input_file_name ), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test', str(k) + "_" + input_file_name), index=False)
    new_df[new_df['date'] <= '2024-11-10'].to_csv(os.path.join(output_dir, 'combined', str(k) + "_" + input_file_name), index=False)
    # latest_matches_df.to_csv(os.path.join(output_dir, 'test_after10nov', str(k) + "_" + input_file_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate player stats with rolling averages")
    parser.add_argument(
        '-i', '--input', type=str, required=True, 
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, required=True, 
        help='Path to the output directory'
    )
    parser.add_argument(
        '-k', '--window', type=int, default=10, 
        help='Window size for rolling averages (default: 10)'
    )
    parser.add_argument(
        '-t', '--threads', type=int, default=4, 
        help='Number of threads to use for parallel processing (default: 4)'
    )
    args = parser.parse_args()
    main(args.input, args.output_dir, args.window, args.threads)