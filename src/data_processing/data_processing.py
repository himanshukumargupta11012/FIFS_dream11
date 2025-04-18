# python data_processing.py --input_dir ../data/raw/cricsheet/ --output_dir ../data/interim/ --num_threads 24
import os
from cricstats import cricketstats
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import glob
from tqdm import tqdm
import argparse


# Define scoring rules for different match types
scoring_rules = {
    "ODI": {
        "run": 1,
        "fours": 4,
        "sixes": 6,
        "run_milestones": [(150, 24), (125, 20), (100, 16), (75, 12), (50, 8), (25, 4)],
        "duck_penalty": -3,
        "wicket": 25,
        "three_dot_ball": 1,
        "lbw_bowled_bonus": 8,
        "wicket_bonuses": [(6, 12), (5, 8), (4, 4)],
        "maiden_over": 4,
        "catch": 8,
        "three_catch_bonus": 4,
        "stumping": 12,
        "direct_run_out": 12,
        "non_direct_run_out": 6,
        "starting_11_bonus": 4,
        "economy_rate_bonus": {
            "min_overs": 5,
            "thresholds": [(2.49, 6), (3.49, 4), (4.5, 2), (6.99, 0), (8, -2), (9, -4)]
        },
        "strike_rate_bonus": {
            "min_balls": 20,
            "thresholds": [(140, 6), (120.01, 4), (100, 2), (50.01, 0), (40, -2), (30, -4)]
        }
    },
    "T20": {
        "run": 1,
        "fours": 4,
        "sixes": 6,
        "run_milestones": [ (100, 16), (75, 12), (50, 8), (25, 4)],
        "duck_penalty": -2,
        "wicket": 25,
        # "three_dot_ball": 1,
        "dot_balls" : 1,
        "lbw_bowled_bonus": 8,
        "wicket_bonuses": [(5, 12), (4, 8), (3, 4)],
        "maiden_over": 12,
        "catch": 8,
        "three_catch_bonus": 4,
        "stumping": 12,
        "direct_run_out": 12,
        "non_direct_run_out": 6,
        "starting_11_bonus": 4,
        "economy_rate_bonus": {
            "min_overs": 2,
            "thresholds": [(4.99, 6), (5.99, 4), (7, 2), (9.99, 0), (11, -2), (12, -4)]
        },
        "strike_rate_bonus": {
            "min_balls": 10,
            "thresholds": [(170, 6), (150.01,4), (130, 2), (70.01, 0), (60, -2), (50, -4)]
        }
    }
}

def generate_df(file_path, search=None):
    match_id = int(file_path.split("/")[-1].split(".")[0])
    if search is None:
        search = cricketstats.search(allplayers=True)
    try:
        df, pvp, match_format = search.stats2(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()
    df, pvp, match_format = search.stats2(file_path)

    df["format"] = match_format
    pvp["format"] = match_format
    df["match_id"] = match_id
    pvp["match_id"] = match_id

    df = calculate_fantasy_points(df, match_format)
    return df, pvp


def calculate_fantasy_points(df, match_format):

    # Get the rules for the given match type
    rules = scoring_rules[match_format]

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

    if match_format == "ODI":
        bowling_points += df['Dot Balls Bowled'].apply(lambda x: rules['three_dot_ball'] * x // 3)
    elif match_format == "T20":
        bowling_points += df['Dot Balls Bowled'].apply(lambda x: rules['dot_balls'] * x)

    fielding_points += df['Catches'].apply(lambda x: rules['three_catch_bonus'] if x >= 3 else 0)

    # Apply economy rate points (only for T20 matches)
    def calculate_economy_points(balls_bowled, Runs_conceded):
        if round((balls_bowled / 6), 1) < rules['economy_rate_bonus']['min_overs']:
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
    df["fielding_fantasy_points"] = fielding_points + 4    
    
    return df


def process_dfs(df, type="all"):
    df = pd.merge(df, info_df[['match_id', 'date', 'type']], on='match_id', how='right')
    df = df.sort_values(by='date')
    grouped_df = df.groupby('format')

    for name, group in grouped_df:
        group.to_csv(f"{output_dir}/{name}_{type}.csv", index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process data from cricsheet.org')
    parser.add_argument('--input_dir', type=str, help='Path to the data folder', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory to store the data', required=True)
    parser.add_argument('--num_threads', type=int, help='Number of threads to use for processing', default=4)

    args = parser.parse_args()

    data_path = args.input_dir
    output_dir = args.output_dir
    num_threads = args.num_threads

    os.makedirs(output_dir, exist_ok=True)

    file_paths = glob.glob(f"{data_path}/all_json/*.json")
    # search = cricketstats.search(allplayers=True)

    dfs = []
    pvp_stats = []
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for df, batting in tqdm(executor.map(generate_df, file_paths), total=len(file_paths)):
            dfs.append(df)
            pvp_stats.append(batting)
    
    dfs = pd.concat(dfs, ignore_index=True)
    pvp_stats = pd.concat(pvp_stats, ignore_index=True)

    # creating readme data
    info_df = pd.read_csv(f"{data_path}/matches_info.csv")
    info_df['match_type'] = info_df['level'] + "-" + info_df['type']
    info_df = pd.merge(info_df, dfs[["match_id", "format"]].drop_duplicates(subset="match_id"), on='match_id', how='left')

    # ----------------- Filtering -----------------
    incomplete_matchid = [291360, 1083447, 64889, 582191, 366709, 1059713, 1336084, 66356, 744683, 1208346, 757505, 65273, 267710, 464990, 1251950, 250670, 1388394, 1275260, 1336130, 433577, 473924, 296918, 1243930, 1276912, 760889, 224044, 1144497, 1140379, 602475, 377316, 749785, 366626, 1185311, 754757, 1408106, 1152841, 1033361, 1348328, 1375867, 295785, 749775, 566943, 238213, 1239536, 1355718, 1272380, 562441, 325576, 914213, 473315, 534228, 1188624, 1456445, 291346, 256665, 932861, 66364, 1322343, 1395700, 1031669, 1378192, 430890, 65652, 66375, 1322279, 66296, 648655, 534214, 295788, 1398272, 1098206, 415281, 1450707, 1368849]
    info_df = info_df[~info_df['match_id'].isin(incomplete_matchid)]
    # ---------------------------------------------

    info_df.to_csv(f"{output_dir}/matches_info.csv", index=False)

    process_dfs(dfs)
    process_dfs(pvp_stats, type="pvp")



