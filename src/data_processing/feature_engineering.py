# python feature_engineering.py --input_dir ../data/interim/ipl --output_dir ../data/processed --window 7 --threads 24 --output_file IPL
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
import numpy as np
from tqdm import tqdm

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

pvp_df = None
main_df = None

specifics_features = ["Runs", "Wickets"
, "Innings Batted", "Innings Bowled"
]
specifics = ["Venue", "Opposition", "Inning"]

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


def getPVPData(row, pvp_stats, type, main_df):
    if type == "batsman_id":
        opposition_type = "bowler_id"
    else:
        opposition_type = "batsman_id"

    opposition_players = main_df[(main_df["match_id"] == row["match_id"]) & (main_df["Team"] == row["opposition"])]["player_id"]

    filtered_df = pvp_stats[(pvp_stats[type] == row['player_id']) & (pvp_stats['date'] < row['date']) & (pvp_stats[opposition_type].isin(opposition_players))]

    filtered_df = filtered_df.groupby(opposition_type)[['runs', 'wickets', 'balls']].mean()

    if filtered_df.empty:
        return 0, 0, 0
    return filtered_df['runs'].mean(), filtered_df['wickets'].mean(), filtered_df['balls'].mean()

def calculate_group_type_rolling_avgs(player_data, new_player_data, features, specifics, k):
    for specific in specifics:
        specific_group = player_data.groupby(['player', specific])
        new_player_data[f'{specific}_total_matches_sum'] = specific_group["player_id"].shift(1).expanding().count()

        for feature in features:
            new_player_data[f'cumulative_{specific}_{feature}_sum'] = (specific_group[feature].shift(1).expanding().mean())

            new_player_data[f'last_{k}_matches_{specific}_{feature}_sum'] = (
                specific_group[feature].apply(lambda x: x.shift(1).rolling(k, min_periods=1).mean())
            ).reset_index(level=[0, 1], drop=True)
   
    return new_player_data



def calculate_player_stats_for_group(player_data, k=10):
    global last_k_columns
    # Initialize cumulative and last k matches columns

    player_data.sort_values(by='date', inplace=True)

    new_player_data = pd.DataFrame()
    new_player_data["player"] = player_data["player"]
    new_player_data["team"] = player_data["Team"]
    new_player_data["opposition"] = player_data["Opposition"]
    new_player_data["type"] = player_data["type"]
    new_player_data["date"] = player_data["date"]
    new_player_data["venue"] = player_data["Venue"]
    new_player_data["match_id"] = player_data["match_id"]
    new_player_data["player_id"] = player_data["player_id"]
    new_player_data["fantasy_points"] = player_data["fielding_fantasy_points"] + player_data["batting_fantasy_points"] + player_data["bowling_fantasy_points"]
    new_player_data["batting_fantasy_points"] = player_data["batting_fantasy_points"]
    new_player_data["bowling_fantasy_points"] = player_data["bowling_fantasy_points"]
    new_player_data["fielding_fantasy_points"] = player_data["fielding_fantasy_points"]
    new_player_data["Total_matches_played_sum"] = player_data["player_id"].shift(1).expanding().count()
    new_player_data[f"last_{k}_matches_fantasy_points_sum"] = new_player_data["fantasy_points"].shift(1).rolling(k, min_periods=1).mean()
    new_player_data[f"last_{k}_matches_fantasy_points_var"] = new_player_data["fantasy_points"].shift(1).rolling(k, min_periods=1).var(ddof=0)
    new_player_data[f"date_diff"] = new_player_data["date"].diff().dt.days
    new_player_data["batting_order"] = player_data["batting_order"]

    new_player_data["k_value"] = k

    # new_player_data["year"] = player_data["Date"].dt.year - 2025


    # new_player_data[['batting_pvp_runs', 'batting_pvp_wickets', 'batting_pvp_balls']] = new_player_data.apply(getPVPData, axis=1, args=(pvp_df, "batsman_id", main_df)).apply(pd.Series)
    # new_player_data[['bowling_pvp_runs', 'bowling_pvp_wickets', 'bowling_pvp_balls']] = new_player_data.apply(getPVPData, axis=1, args=(pvp_df, "bowler_id", main_df)).apply(pd.Series)

    for col in FEATURES:
        shifted_data = player_data[col].shift(1)
        new_player_data[f'cumulative_{col}_sum'] = shifted_data.expanding().mean()
        new_player_data[f'last_{k}_matches_{col}_sum'] = shifted_data.rolling(k, min_periods=1).mean()
    
    new_player_data[f"last_{k}_matches_lbw_bowled_sum"] = (new_player_data[f"last_{k}_matches_LBWs_sum"] + new_player_data[f"last_{k}_matches_Bowleds_sum"])
    
    new_player_data[f"last_{k}_matches_centuries_sum"] = (player_data["Runs"] >= 100).astype(int).shift(1).rolling(k, min_periods=1).mean()
    new_player_data[f"last_{k}_matches_half_centuries_sum"] = (player_data["Runs"] >= 50).astype(int).shift(1).rolling(k, min_periods=1).mean()

    new_player_data[f"last_{k}_matches_duck_outs_sum"] = ((player_data["Runs"] == 0) & (player_data["Innings Batted"] == 1)).astype(int).shift(1).rolling(10, min_periods=1).mean()
    
    new_player_data[f"last_{k}_matches_4wickets_sum"] = ((player_data["Wickets"] >= 4) & (player_data["Wickets"] < 5)).astype(int).shift(1).rolling(k, min_periods=1).mean()
    new_player_data[f"last_{k}_matches_5wickets_sum"] = ((player_data["Wickets"] >= 5) & (player_data["Wickets"] < 6)).astype(int).shift(1).rolling(k, min_periods=1).mean()
    new_player_data[f"last_{k}_matches_6wickets_sum"] = (player_data["Wickets"] >= 6).astype(int).shift(1).rolling(k, min_periods=1).mean()


    for feature in ["Runs", "Wickets"]:
        # Calculate rolling one-year averages for both columns
        # new_player_data[f'last_year_avg_{feature}'] = (
        #     player_data[feature]
        #     .rolling('365D', closed='left', min_periods=1)  # 365-day rolling window
        #     .mean()
        # )

        new_player_data[f'last_year_avg_{feature}'] = player_data.apply(
            lambda row: player_data[
                (player_data['date'] < row['date']) & 
                (player_data['date'] > row['date'] - pd.DateOffset(years=1))
            ][feature].mean(),
            axis=1
        )




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
    new_player_data = new_player_data.copy()

    new_player_data = calculate_group_type_rolling_avgs(player_data, new_player_data, specifics_features, specifics, k)

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
    venue_total_avg_runs = venue_avg_runs.mean()
    venue_avg_wickets = df.groupby("Venue")['Wickets'].mean() * 11
    venue_total_avg_wickets = venue_avg_wickets.mean()
    new_df["venue_avg_runs_sum"] = new_df["venue"].map(venue_avg_runs).fillna(venue_total_avg_runs)
    new_df["venue_avg_wickets_sum"] = new_df["venue"].map(venue_avg_wickets).fillna(venue_total_avg_wickets)

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

    df[f"last_{k}_matches_fantasy_points_var"] = df[f"last_{k}_matches_fantasy_points_var"].fillna(0)

    for specific in specifics:
        df[f"{specific}_total_matches_sum"] = df[f"{specific}_total_matches_sum"].fillna(0)

    for feature in ["Runs", "Wickets"]:
        df[f'last_year_avg_{feature}'] = df[f'last_year_avg_{feature}'].fillna(value)

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        null_columns = list(df.columns[df.isnull().any()])
        print(f"There are NaN or infinite values in these columns {null_columns}")

    return df


def main(input_dir, output_dir, k, output_file_name, num_threads):
    global pvp_df, main_df
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pvp_df = pd.read_csv(os.path.join(input_dir, "pvp.csv"))
    pvp_df["date"] = pd.to_datetime(pvp_df["date"])

    main_df = pd.read_csv(os.path.join(input_dir, "all.csv"))
    main_df['date'] = pd.to_datetime(main_df['date'])
    
    new_df = calculate_player_stats(main_df, num_threads, k)

    # train_df = new_df[(start_date <= new_df['date']) & (new_df['date'] <= mid_date)]
    # test_df = new_df[(new_df['date'] > mid_date) & (new_df['date'] <= end_date)]    
    
    # Saving the data
    # for subfolder in ['train', 'test', 'combined']:
    subfolder = "combined"
    os.makedirs(os.path.join(output_dir, subfolder), exist_ok=True)

    full_output_file_name = f"{k}_{output_file_name}.csv"
    # train_df.to_csv(os.path.join(output_dir, 'train', full_output_file_name), index=False)
    # test_df.to_csv(os.path.join(output_dir, 'test', full_output_file_name), index=False)
    new_df.to_csv(os.path.join(output_dir, 'combined', full_output_file_name), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate player stats with rolling averages")
    parser.add_argument(
        '-i', '--input_dir', type=str, required=True, 
        help='Path to the data folder'
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
    parser.add_argument(
        '-of', '--output_file', type=str, required=True, 
        help='Output file name'
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.window, args.output_file, args.threads)