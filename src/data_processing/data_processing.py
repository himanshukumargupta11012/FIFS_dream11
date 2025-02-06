# python data_processing.py ../data/raw/cricsheet/ ../data/interim/ 6
import os
from cricstats import cricketstats
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import glob
from tqdm import tqdm
pd.set_option('display.max_columns', None)


# Check if the correct number of arguments are passed
if len(sys.argv) != 4:
    print("Usage: python3 data_processing.py <file_path> <output_dir> <num_threads")
    sys.exit(1)  # Exit the script with an error code

# # Get the arguments
_, data_path, output_dir, num_threads = sys.argv
num_threads = int(num_threads)


def generate_df(file_path):
    
    id = file_path.split("/")[-1].split(".")[0]
    
    df, match_format = search.stats2(file_path)

    df["format"] = match_format
    df["match_id"] = int(file_path.split("/")[-1].split(".")[0])

    df = calculate_fantasy_points(df, match_format)
    return df


def calculate_fantasy_points(df, match_type):
    # Define scoring rules for different match types
    scoring_rules = {
        "T20": {
            "run": 1,
            "fours": 1,
            "sixes": 2,
            "run_milestones": [(100, 16), (50, 8), (30, 4)],
            "duck_penalty": -2,
            "wicket": 25,
            "lbw_bowled_bonus": 8,
            "wicket_bonuses": [(5, 16), (4, 8), (3, 4)],
            "maiden_over": 12,
            "catch": 8,
            "three_catch_bonus": 4,
            "stumping": 12,
            "direct_run_out": 12,
            "non_direct_run_out": 6,
            "starting_11_bonus": 4,
            "economy_rate_bonus": {
                "min_overs": 2,
                "thresholds": [(5, 6), (5.99, 4), (7, 2), (10, 0), (11, -2), (12, -4)]
            },
            "strike_rate_bonus": {
                "min_balls": 10,
                "thresholds": [(170, 6), (150.01, 4), (130, 2), (70, 0), (60, -2), (50, -4)]
            }
        },
        "ODI": {
            "run": 1,
            "fours": 1,
            "sixes": 2,
            "run_milestones": [(100, 8), (50, 4)],
            "duck_penalty": -3,
            "wicket": 25,
            "lbw_bowled_bonus": 8,
            "wicket_bonuses": [(5, 8), (4, 4)],
            "maiden_over": 4,
            "catch": 8,
            "three_catch_bonus": 4,
            "stumping": 12,
            "direct_run_out": 12,
            "non_direct_run_out": 6,
            "starting_11_bonus": 4,
            "economy_rate_bonus": {
                "min_overs": 5,
                "thresholds": [(2.5, 6), (3.49, 4), (4.5, 2), (7, 0), (8.01, -2), (9, -4)]
            },
            "strike_rate_bonus": {
                "min_balls": 20,
                "thresholds": [(140, 6), (120.01, 4), (100, 2), (50, 0), (40, -2), (30, -4)]
            }
        },
        "Test": {
            "run": 1,
            "fours": 1,
            "sixes": 2,
            "run_milestones": [(100, 8), (50, 4)],
            "duck_penalty": -4,
            "wicket": 16,
            "lbw_bowled_bonus": 8,
            "wicket_bonuses": [(5, 8), (4, 4)],
            "maiden_over": 0,  # No points for maiden overs in Test matches
            "catch": 8,
            "three_catch_bonus": 0,  # No 3-catch bonus in Test matches
            "stumping": 12,
            "direct_run_out": 12,
            "non_direct_run_out": 6,
            "starting_11_bonus": 4,
        }
    }

    # Get the rules for the given match type
    rules = scoring_rules.get(match_type)

    # Calculate fantasy points
    batting_points = (
        df['Runs'] * rules['run'] +
        df['Fours'] * rules['fours'] +
        df['Sixes'] * rules['sixes']
    )
    bowling_points = (
        df['Wickets'] * rules['wicket'] +
        (df['Bowleds'] + df['LBWs']) * rules['lbw_bowled_bonus'] +
        df['Maiden Overs'] * rules['maiden_over']
    )
    fielding_points = (
        df['Catches'] * rules['catch'] +
        df['Stumpings'] * rules['stumping'] +
        df['direct run_outs'] * rules['direct_run_out'] +
        df['indirect run_outs'] * rules['non_direct_run_out']
    )

    # Apply run milestones bonuses
    batting_points += df['Runs'].apply(
        lambda runs: max([points for milestone, points in rules['run_milestones'] if runs >= milestone], default=0)
    )

    # Apply duck penalty
    batting_points += df.apply(
        lambda row: rules['duck_penalty'] if row['Runs'] == 0 and row['Outs'] == True else 0, axis=1
    )
    # Apply wicket bonuses
    bowling_points += df['Wickets'].apply(
        lambda w: max([points for milestone, points in rules['wicket_bonuses'] if w >= milestone], default=0)
    )

    # Apply 3-catch bonus (only for T20 matches)
    if rules.get('three_catch_bonus', 0) > 0:
        fielding_points += df['Catches'].apply(lambda x: rules['three_catch_bonus'] if x >= 3 else 0)

    # Apply economy rate points (only for T20 matches)
    if match_type in ["T20", "OD"]:
        def calculate_economy_points(balls_bowled, Runs_conceded):
            if round((balls_bowled / 6),1) < rules['economy_rate_bonus']['min_overs']:
                return 0
            economy_rate = round((Runs_conceded / balls_bowled) * 6, 3)
            for threshold, points in rules['economy_rate_bonus']['thresholds']:
                if economy_rate <= threshold:
                    return points
            return -6
        
        bowling_points += df.apply(
            lambda row: calculate_economy_points(row['Balls Bowled'], row['Runsgiven']), axis=1
        )

    # Apply strike rate points (only for T20 matches)
        def calculate_strike_rate_points(Runs, balls_faced):
            if balls_faced < rules['strike_rate_bonus']['min_balls']:
                return 0
            strike_rate = round((Runs / balls_faced) * 100,3)
            for threshold, points in rules['strike_rate_bonus']['thresholds']:
                if strike_rate >= threshold:
                    return points
            return -6
        
        batting_points += df.apply(
            lambda row: calculate_strike_rate_points(row['Runs'], row['Balls Faced']), axis=1
        )

    df["batting_fantasy_points"] = batting_points
    df["bowling_fantasy_points"] = bowling_points
    df["fielding_fantasy_points"] = fielding_points
    
    return df


file_paths = glob.glob(f"{data_path}/all_json/*.json")
search = cricketstats.search(allplayers=True)

with ProcessPoolExecutor(max_workers=num_threads) as executor:
    dfs = list(tqdm(executor.map(generate_df, file_paths), total=len(file_paths)))
combined_df = pd.concat(dfs, ignore_index=True)

# creating readme data
info_df = pd.read_csv(f"{data_path}/matches_info.csv")
info_df['match_type'] = info_df['type'] + "-" + info_df['format']

# ----------------- Filtering -----------------
info_df = info_df[info_df['gender'] == 'male']
# ---------------------------------------------


info_df.to_csv(f"{output_dir}/matches_info.csv", index=False)

# Select only the 'name', 'age', and 'address_email' columns from df2
info_df = info_df[['match_id', 'match_type']]

# Merge the dataframes on 'name'
combined_df = pd.merge(combined_df, info_df, on='match_id', how='right')

combined_df = combined_df.sort_values(by='Date')
grouped = combined_df.groupby('format')

# Create separate DataFrames for each category
for name, group in grouped:
    df = grouped.get_group(name)
    df.to_csv(f"{output_dir}/{name}.csv", index=False)


