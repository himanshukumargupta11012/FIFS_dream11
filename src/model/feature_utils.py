import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def process(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")
        
    new_df = df.iloc[:, data_start:]
    # new_df = new_df.loc[:, ~new_df.columns.str.startswith('cumulative')]
    # new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'last_{k}_matches_derived')]
    new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'cumulative_derived_')]
   
    new_df = new_df.drop(
        [   
            'batting_order',
            'year',
            'cumulative_Innings Batted_sum',
            'cumulative_Outs_sum',
       'cumulative_Dot Balls_sum',
            # 'Total Innings Played',
        #     f'last_{k}_matches_Fours_sum',
        #    f'last_{k}_matches_Sixes_sum',
            f'last_{k}_matches_Outs_sum',
            # f'last_{k}_matches_fantasy_points_sum',
            f'last_{k}_matches_Dot Balls_sum',  
            f'last_{k}_matches_Balls Faced_sum',
            f'last_{k}_matches_Innings Bowled_sum',
            f'last_{k}_matches_Balls Bowled_sum',
            'Opponent_total_matches',
            'Venue_total_matches',
          #   f'last_{k}_matches_Runsgiven_sum',
          #   f'last_{k}_matches_Dot Balls Bowled_sum',
            f'last_{k}_matches_Foursgiven_sum',
            f'last_{k}_matches_Sixesgiven_sum',
        #     'venue_avg_runs',
        #    'venue_avg_wickets',
          f'last_{k}_matches_Extras_sum',
            f'last_{k}_matches_centuries_sum',
          f'last_{k}_matches_half_centuries_sum',
        #    f'last_{k}_matches_opponent_Runs_sum',
        #  f'last_{k}_matches_venue_Runs_sum',
        f'last_{k}_matches_Wickets_sum', f'last_{k}_matches_LBWs_sum',
          #  f'last_{k}_matches_Maiden Overs_sum', f'last_{k}_matches_Stumpings_sum',
          #  f'last_{k}_matches_Catches_sum', f'last_{k}_matches_direct run_outs_sum',
          #  f'last_{k}_matches_opponent_Wickets_sum',
          #  f'last_{k}_matches_venue_Wickets_sum'
      #  f'last_{k}_matches_direct run_outs_sum',
      #     f'last_{k}_matches_indirect run_outs_sum',
      #  f'last_{k}_matches_match_type_Innings Batted_sum',
      #     f'last_{k}_matches_match_type_Innings Bowled_sum',
          #  'match_type_total_matches',
          #  "batting_fantasy_points",
          #  "bowling_fantasy_points",
          #  "fielding_fantasy_points",
          #  f'last_{k}_matches_Venue_Wickets_sum',
          #  f'last_{k}_matches_Opposition_Innings Bowled_sum',
          #  f'last_{k}_matches_match_type_Wickets_sum',
          #  f'last_{k}_matches_Opposition_Wickets_sum',
          f'last_{k}_matches_derived_Economy Rate',
          #  f'last_{k}_matches_Venue_Innings Bowled_sum',
          f'last_{k}_matches_lbw_bowled_sum',
          f'last_{k}_matches_Bowleds_sum',
          f'last_{k}_matches_duck_outs_sum',
          f'last_{k}_matches_6wickets_sum',
          f'last_{k}_matches_4wickets_sum',
            f'last_{k}_matches_5wickets_sum',
            'date_diff'
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    # print(new_df.columns)
    if not return_tensor:
        return new_df
        
    X_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_tensor, new_df.columns



def classification_process(df, k, train=True, is_mlp=False):
    num_players = 22
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")

    if train:
        df = df.sort_values(['match_id', 'fantasy_points'], ascending=[True, False]).groupby('match_id').head(22)
    else:
        num_players = df.shape[0]

    df.sort_values(by=['match_id', 'player_id'], inplace=True)
    if train:
        fantasy_points = df['fantasy_points'].values
    new_df = df.iloc[:, data_start:]
    # new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'cumulative_derived_')]
    new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'hist_')]
   
    new_df = new_df.drop(
        [   
    #         'batting_order',
    #         'year',
    #         'cumulative_Innings Batted_sum',
    #         'cumulative_Outs_sum',
    #    'cumulative_Dot Balls_sum',
            # 'Total Innings Played',
        #     f'last_{k}_matches_Fours_sum',
        #    f'last_{k}_matches_Sixes_sum',
            # f'last_{k}_matches_Outs_sum',
            # f'last_{k}_matches_fantasy_points_sum',
            # f'last_{k}_matches_Dot Balls_sum',  
            # f'last_{k}_matches_Balls Faced_sum',
            # f'last_{k}_matches_Innings Bowled_sum',
            # f'last_{k}_matches_Balls Bowled_sum',
            # 'Opponent_total_matches',
            # 'Venue_total_matches',
          #   f'last_{k}_matches_Runsgiven_sum',
          #   f'last_{k}_matches_Dot Balls Bowled_sum',
            # f'last_{k}_matches_Foursgiven_sum',
            # f'last_{k}_matches_Sixesgiven_sum',
        #     'venue_avg_runs',
        #    'venue_avg_wickets',
          f'last_{k}_matches_Extras_sum',
            f'last_{k}_matches_centuries_sum',
          f'last_{k}_matches_half_centuries_sum',
        #    f'last_{k}_matches_opponent_Runs_sum',
        #  f'last_{k}_matches_venue_Runs_sum',
        f'last_{k}_matches_Wickets_sum', f'last_{k}_matches_LBWs_sum',
          #  f'last_{k}_matches_Maiden Overs_sum', f'last_{k}_matches_Stumpings_sum',
          #  f'last_{k}_matches_Catches_sum', f'last_{k}_matches_direct run_outs_sum',
          #  f'last_{k}_matches_opponent_Wickets_sum',
          #  f'last_{k}_matches_venue_Wickets_sum'
      #  f'last_{k}_matches_direct run_outs_sum',
      #     f'last_{k}_matches_indirect run_outs_sum',
      #  f'last_{k}_matches_match_type_Innings Batted_sum',
      #     f'last_{k}_matches_match_type_Innings Bowled_sum',
          #  'match_type_total_matches',
          #  "batting_fantasy_points",
          #  "bowling_fantasy_points",
          #  "fielding_fantasy_points",
          #  f'last_{k}_matches_Venue_Wickets_sum',
          #  f'last_{k}_matches_Opposition_Innings Bowled_sum',
          #  f'last_{k}_matches_match_type_Wickets_sum',
          #  f'last_{k}_matches_Opposition_Wickets_sum',
          f'last_{k}_matches_derived_Economy Rate',
          #  f'last_{k}_matches_Venue_Innings Bowled_sum',
        #   f'last_{k}_matches_lbw_bowled_sum',
        #   f'last_{k}_matches_Bowleds_sum',
        #   f'last_{k}_matches_duck_outs_sum',
        #   f'last_{k}_matches_6wickets_sum',
        #   f'last_{k}_matches_4wickets_sum',
        #     f'last_{k}_matches_5wickets_sum',
            'date_diff'

        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )

    if new_df.isnull().values.any() or np.isinf(new_df.values).any():
        null_columns = list(new_df.columns[new_df.isnull().any()])
        print(f"There are NaN or infinite values in these columns {null_columns}")

       
    X_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    X_tensor = X_tensor.view(-1, num_players, X_tensor.shape[-1])
    if is_mlp:
        X_tensor = X_tensor.flatten(start_dim=1)

    if train:
        y_tensor = torch.tensor(fantasy_points)
        y_tensor = y_tensor.view(-1, num_players)
        topk_indices = torch.topk(y_tensor, k=11, dim=1).indices

        y_tensor = torch.zeros_like(y_tensor)
        y_tensor.scatter_(1, topk_indices, 1)

        y_tensor = y_tensor.to(torch.float32)

        return X_tensor, y_tensor, new_df.columns
    
    else:
        return X_tensor, new_df.columns
    



import pandas as pd
def compute_overlap_true_test(true_tensor, pred_tensor, pred_match_id):
    
    df = pd.DataFrame({'match_id': pred_match_id, 'predicted_points': pred_tensor, 'fantasy_points': true_tensor})

    common_indices = []
    for match_id, match_data in df.groupby('match_id'):
        top_predicted_indices = match_data["predicted_points"].nlargest(11).index
        top_actual_indices = match_data["fantasy_points"].nlargest(11).index

        matching_indices = set(top_predicted_indices).intersection(top_actual_indices)
        common_indices.append(len(matching_indices))

    return sum(common_indices) / len(common_indices)





# Function to compute MAE and MAPE loss
def compute_loss(df):
    prediction_list = []
    actual_list = []
    for match_id, match_data in df.groupby('match_id'):
        top_predicted = match_data.sort_values(by='predicted_points', ascending=False).iloc[:11]['predicted_points']
        top_actual = match_data.sort_values(by='fantasy_points', ascending=False).iloc[:11]['fantasy_points']

        top_predicted += 4
        top_predicted.iloc[0] *= 2
        top_predicted.iloc[1] = int(top_predicted.iloc[1] * 1.5)

        top_actual += 4
        top_actual.iloc[0] *= 2
        top_actual.iloc[1] = int(top_actual.iloc[1] * 1.5)

        predicted_sum = top_predicted.sum()
        actual_sum = top_actual.sum()

        prediction_list.append(predicted_sum)
        actual_list.append(actual_sum)

    prediction_list = np.array(prediction_list)
    actual_list = np.array(actual_list)

    MAE = np.mean(np.abs(prediction_list - actual_list))
    MAPE = np.mean(np.abs((prediction_list - actual_list) / actual_list))

    return MAE, MAPE

# Function to compute MAE and MAPE loss for dream team top 11
def compute_loss2(df):
    prediction_list = []
    actual_list = []
    for match_id, match_data in df.groupby('match_id'):
        # top_predicted = match_data.sort_values(by='predicted_points', ascending=False).iloc[:11]['predicted_points']
        top_predicted = match_data.sort_values(by='predicted_points', ascending=False).iloc[:11]['fantasy_points']
        top_actual = match_data.sort_values(by='fantasy_points', ascending=False).iloc[:11]['fantasy_points']

        top_predicted += 4
        top_predicted.iloc[0] *= 2
        top_predicted.iloc[1] = int(top_predicted.iloc[1] * 1.5)

        top_actual += 4
        top_actual.iloc[0] *= 2
        top_actual.iloc[1] = int(top_actual.iloc[1] * 1.5)

        predicted_sum = top_predicted.sum()
        actual_sum = top_actual.sum()

        prediction_list.append(predicted_sum)
        actual_list.append(actual_sum)

    prediction_list = np.array(prediction_list)
    actual_list = np.array(actual_list)

    MAE = np.mean(np.abs(prediction_list - actual_list))
    MAPE = np.mean(np.abs((prediction_list - actual_list) / actual_list))

    return MAE, MAPE

# function for normalizing the data
def normalise_data(X, y, MinMax=True):
    if MinMax:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    else:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Reshape y to 2D before scaling

    return X_scaled, y_scaled, scaler_X, scaler_y


def normalise_data2(X, y, MinMax=True):
    if MinMax:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    else:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y) # Reshape y to 2D before scaling

    return X_scaled, y_scaled, scaler_X, scaler_y

def process_batting(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")
        
    new2_df = df.iloc[:, data_start:]
    new_df = new2_df.loc[:, ~new2_df.columns.str.startswith('cumulative')]
    new_df = new_df.drop(columns=['date_diff'], errors='ignore')
    new_df['match_id'] = df['match_id']
    new_df['player_id'] = df['player_id']
    new_df = new_df.drop(
        [   
            'batting_order',
            # 'year',
            # 'Total Innings Played',
        #     f'last_{k}_matches_Fours_sum',
        #    f'last_{k}_matches_Sixes_sum',
            # f'last_{k}_matches_Outs_sum',
            f'last_{k}_matches_fantasy_points_sum',
            # f'last_{k}_matches_Dot Balls_sum',  
            # f'last_{k}_matches_Balls Faced_sum',
            f'last_{k}_matches_Innings Bowled_sum',
            f'last_{k}_matches_Balls Bowled_sum',
            f'last_{k}_matches_derived_Dot Ball%',
            # f'last_{k}_matches_derived_Batting Strike Rate',
            # f'last_{k}_matches_derived_Batting Avg',
            # f'last_{k}_matches_derived_Mean Score',
            # f'last_{k}_matches_derived_Boundary%',
            # f'last_{k}_matches_derived_Mean Balls Faced',
            # f'last_{k}_matches_derived_Dismissal Rate',
            f'last_{k}_matches_derived_Bowling Dot Ball%',
            f'last_{k}_matches_derived_Boundary Given%',
            f'last_{k}_matches_derived_Bowling Avg',
            f'last_{k}_matches_derived_Bowling Strike Rate',
            'Opponent_total_matches',
            'Venue_total_matches',
            f'last_{k}_matches_Runsgiven_sum',
            f'last_{k}_matches_Dot Balls Bowled_sum',
            f'last_{k}_matches_Foursgiven_sum',
            f'last_{k}_matches_Sixesgiven_sum',
        #     'venue_avg_runs',
        #    'venue_avg_wickets',
          f'last_{k}_matches_Extras_sum',
            # f'last_{k}_matches_centuries_sum',
        #   f'last_{k}_matches_half_centuries_sum',
        #    f'last_{k}_matches_opponent_Runs_sum',
        #  f'last_{k}_matches_venue_Runs_sum',
        f'last_{k}_matches_Wickets_sum', f'last_{k}_matches_LBWs_sum',
           f'last_{k}_matches_Maiden Overs_sum', f'last_{k}_matches_Stumpings_sum',
           f'last_{k}_matches_Catches_sum', f'last_{k}_matches_direct run_outs_sum',
           f'last_{k}_matches_opponent_Wickets_sum',
           f'last_{k}_matches_venue_Wickets_sum'
       f'last_{k}_matches_direct run_outs_sum',
          f'last_{k}_matches_indirect run_outs_sum',
      #  f'last_{k}_matches_match_type_Innings Batted_sum',
          f'last_{k}_matches_match_type_Innings Bowled_sum',
          #  'match_type_total_matches',
           "batting_fantasy_points",
           "bowling_fantasy_points",
           "fielding_fantasy_points",
           f'last_{k}_matches_Venue_Wickets_sum',
           f'last_{k}_matches_Opposition_Innings Bowled_sum',
           f'last_{k}_matches_match_type_Wickets_sum',
           f'last_{k}_matches_Opposition_Wickets_sum',
          f'last_{k}_matches_derived_Economy Rate',
           f'last_{k}_matches_Venue_Innings Bowled_sum',
          f'last_{k}_matches_lbw_bowled_sum',
          f'last_{k}_matches_Bowleds_sum',
        #   f'last_{k}_matches_duck_outs_sum',
          f'last_{k}_matches_6wickets_sum',
          f'last_{k}_matches_4wickets_sum',
            f'last_{k}_matches_5wickets_sum'
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    if not return_tensor:
        return new_df
    
        
    X_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_tensor, new_df.columns

def process_bowling(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")
        
    new2_df = df.iloc[:, data_start:]
    new_df = new2_df.loc[:, ~new2_df.columns.str.startswith('cumulative')]
    new_df = new_df.drop(columns=['date_diff'], errors='ignore')
    new_df['match_id'] = df['match_id']
    new_df['player_id'] = df['player_id']
    new_df = new_df.drop(
        [   
            'batting_order',
            # 'year',
            # 'Total Innings Played',
            f'last_{k}_matches_Fours_sum',
           f'last_{k}_matches_Sixes_sum',
            f'last_{k}_matches_Outs_sum',
            f'last_{k}_matches_fantasy_points_sum',
            f'last_{k}_matches_Dot Balls_sum',  
            f'last_{k}_matches_Balls Faced_sum',
            # f'last_{k}_matches_Innings Bowled_sum',
            # f'last_{k}_matches_Balls Bowled_sum',
            # f'last_{k}_matches_derived_Dot Ball%',
            f'last_{k}_matches_derived_Batting Strike Rate',
            f'last_{k}_matches_derived_Batting Avg',
            f'last_{k}_matches_derived_Mean Score',
            f'last_{k}_matches_derived_Boundary%',
            f'last_{k}_matches_derived_Mean Balls Faced',
            f'last_{k}_matches_derived_Dismissal Rate',
            # f'last_{k}_matches_derived_Bowling Dot Ball%',
            # f'last_{k}_matches_derived_Boundary Given%',
            # f'last_{k}_matches_derived_Bowling Avg',
            # f'last_{k}_matches_derived_Bowling Strike Rate',
            'Opponent_total_matches',
            'Venue_total_matches',
            # f'last_{k}_matches_Runsgiven_sum',
            # f'last_{k}_matches_Dot Balls Bowled_sum',
            # f'last_{k}_matches_Foursgiven_sum',
            # f'last_{k}_matches_Sixesgiven_sum',
        #     'venue_avg_runs',
        #    'venue_avg_wickets',
        #   f'last_{k}_matches_Extras_sum',
            f'last_{k}_matches_centuries_sum',
          f'last_{k}_matches_half_centuries_sum',
           f'last_{k}_matches_opponent_Runs_sum',
         f'last_{k}_matches_venue_Runs_sum',
        f'last_{k}_matches_Wickets_sum', f'last_{k}_matches_LBWs_sum',
        #    f'last_{k}_matches_Maiden Overs_sum', 
        f'last_{k}_matches_Stumpings_sum',
           f'last_{k}_matches_Catches_sum', f'last_{k}_matches_direct run_outs_sum',
           f'last_{k}_matches_opponent_Wickets_sum',
           f'last_{k}_matches_venue_Wickets_sum'
       f'last_{k}_matches_direct run_outs_sum',
          f'last_{k}_matches_indirect run_outs_sum',
       f'last_{k}_matches_match_type_Innings Batted_sum',
        #   f'last_{k}_matches_match_type_Innings Bowled_sum',
          #  'match_type_total_matches',
           "batting_fantasy_points",
           "bowling_fantasy_points",
           "fielding_fantasy_points",
           f'last_{k}_matches_Venue_Wickets_sum',
        #    f'last_{k}_matches_Opposition_Innings Bowled_sum',
           f'last_{k}_matches_match_type_Wickets_sum',
           f'last_{k}_matches_Opposition_Wickets_sum',
        #   f'last_{k}_matches_derived_Economy Rate',
        #    f'last_{k}_matches_Venue_Innings Bowled_sum',
          f'last_{k}_matches_lbw_bowled_sum',
          f'last_{k}_matches_Bowleds_sum',
          f'last_{k}_matches_duck_outs_sum',
          f'last_{k}_matches_6wickets_sum',
          f'last_{k}_matches_4wickets_sum',
            f'last_{k}_matches_5wickets_sum'
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    if not return_tensor:
        return new_df
        
    X_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_tensor, new_df.columns

def process_wickets(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")
        
    new2_df = df.iloc[:, data_start:]
    new_df = new2_df.loc[:, ~new2_df.columns.str.startswith('cumulative')]
    new_df = new_df.drop(columns=['date_diff'], errors='ignore')
    new_df['match_id'] = df['match_id']
    new_df['player_id'] = df['player_id']
    new_df = new_df.drop(
        [   
            'batting_order',
            # 'year',
            # 'Total Innings Played',
            f'last_{k}_matches_Fours_sum',
           f'last_{k}_matches_Sixes_sum',
            f'last_{k}_matches_Outs_sum',
            f'last_{k}_matches_fantasy_points_sum',
            f'last_{k}_matches_Dot Balls_sum',  
            f'last_{k}_matches_Balls Faced_sum',
            f'last_{k}_matches_Innings Bowled_sum',
            f'last_{k}_matches_Balls Bowled_sum',
            f'last_{k}_matches_derived_Dot Ball%',
            f'last_{k}_matches_derived_Batting Strike Rate',
            f'last_{k}_matches_derived_Batting Avg',
            f'last_{k}_matches_derived_Mean Score',
            f'last_{k}_matches_derived_Boundary%',
            f'last_{k}_matches_derived_Mean Balls Faced',
            f'last_{k}_matches_derived_Dismissal Rate',
            f'last_{k}_matches_derived_Bowling Dot Ball%',
            f'last_{k}_matches_derived_Boundary Given%',
            f'last_{k}_matches_derived_Bowling Avg',
            f'last_{k}_matches_derived_Bowling Strike Rate',
            'Opponent_total_matches',
            'Venue_total_matches',
            f'last_{k}_matches_Runsgiven_sum',
            f'last_{k}_matches_Dot Balls Bowled_sum',
            f'last_{k}_matches_Foursgiven_sum',
            f'last_{k}_matches_Sixesgiven_sum',
            'venue_avg_runs',
        #    'venue_avg_wickets',
          f'last_{k}_matches_Extras_sum',
            f'last_{k}_matches_centuries_sum',
          f'last_{k}_matches_half_centuries_sum',
           f'last_{k}_matches_opponent_Runs_sum',
         f'last_{k}_matches_venue_Runs_sum',
        # f'last_{k}_matches_Wickets_sum', f'last_{k}_matches_LBWs_sum',
           f'last_{k}_matches_Maiden Overs_sum', 
        f'last_{k}_matches_Stumpings_sum',
           f'last_{k}_matches_Catches_sum', f'last_{k}_matches_direct run_outs_sum',
        #    f'last_{k}_matches_opponent_Wickets_sum',
        #    f'last_{k}_matches_venue_Wickets_sum'
       f'last_{k}_matches_direct run_outs_sum',
          f'last_{k}_matches_indirect run_outs_sum',
       f'last_{k}_matches_match_type_Innings Batted_sum',
          f'last_{k}_matches_match_type_Innings Bowled_sum',
        #    'match_type_total_matches',
           "batting_fantasy_points",
           "bowling_fantasy_points",
           "fielding_fantasy_points",
        #    f'last_{k}_matches_Venue_Wickets_sum',
           f'last_{k}_matches_Opposition_Innings Bowled_sum',
        #    f'last_{k}_matches_match_type_Wickets_sum',
        #    f'last_{k}_matches_Opposition_Wickets_sum',
          f'last_{k}_matches_derived_Economy Rate',
           f'last_{k}_matches_Venue_Innings Bowled_sum',
        #   f'last_{k}_matches_lbw_bowled_sum',
        #   f'last_{k}_matches_Bowleds_sum',
          f'last_{k}_matches_duck_outs_sum',
        #   f'last_{k}_matches_4wickets_sum',
        #   f'last_{k}_matches_5wickets_sum',
        #     f'last_{k}_matches_6wickets_sum'
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    if not return_tensor:
        return new_df
        
    X_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_tensor, new_df.columns

def process_field(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")
        
    new2_df = df.iloc[:, data_start:]
    new_df = new2_df.loc[:, ~new2_df.columns.str.startswith('cumulative')]
    new_df = new_df.drop(columns=['date_diff'], errors='ignore')
    new_df['match_id'] = df['match_id']
    new_df['player_id'] = df['player_id']
    new_df = new_df.drop(
        [   
            'batting_order',
            # 'year',
            # 'Total Innings Played',
            f'last_{k}_matches_Fours_sum',
           f'last_{k}_matches_Sixes_sum',
            f'last_{k}_matches_Outs_sum',
            f'last_{k}_matches_fantasy_points_sum',
            f'last_{k}_matches_Dot Balls_sum',  
            f'last_{k}_matches_Balls Faced_sum',
            f'last_{k}_matches_Innings Bowled_sum',
            f'last_{k}_matches_Balls Bowled_sum',
            f'last_{k}_matches_derived_Dot Ball%',
            f'last_{k}_matches_derived_Batting Strike Rate',
            f'last_{k}_matches_derived_Batting Avg',
            f'last_{k}_matches_derived_Mean Score',
            f'last_{k}_matches_derived_Boundary%',
            f'last_{k}_matches_derived_Mean Balls Faced',
            f'last_{k}_matches_derived_Dismissal Rate',
            f'last_{k}_matches_derived_Bowling Dot Ball%',
            f'last_{k}_matches_derived_Boundary Given%',
            f'last_{k}_matches_derived_Bowling Avg',
            f'last_{k}_matches_derived_Bowling Strike Rate',
            'Opponent_total_matches',
            'Venue_total_matches',
            f'last_{k}_matches_Runsgiven_sum',
            f'last_{k}_matches_Dot Balls Bowled_sum',
            f'last_{k}_matches_Foursgiven_sum',
            f'last_{k}_matches_Sixesgiven_sum',
            'venue_avg_runs',
        #    'venue_avg_wickets',
          f'last_{k}_matches_Extras_sum',
            f'last_{k}_matches_centuries_sum',
          f'last_{k}_matches_half_centuries_sum',
           f'last_{k}_matches_opponent_Runs_sum',
         f'last_{k}_matches_venue_Runs_sum',
        f'last_{k}_matches_Wickets_sum', f'last_{k}_matches_LBWs_sum',
           f'last_{k}_matches_Maiden Overs_sum', 
        # f'last_{k}_matches_Stumpings_sum',
        #    f'last_{k}_matches_Catches_sum', f'last_{k}_matches_direct run_outs_sum',
        #    f'last_{k}_matches_opponent_Wickets_sum',
        #    f'last_{k}_matches_venue_Wickets_sum'
    #    f'last_{k}_matches_direct run_outs_sum',
        #   f'last_{k}_matches_indirect run_outs_sum',
       f'last_{k}_matches_match_type_Innings Batted_sum',
          f'last_{k}_matches_match_type_Innings Bowled_sum',
        #    'match_type_total_matches',
           "batting_fantasy_points",
           "bowling_fantasy_points",
           "fielding_fantasy_points",
        #    f'last_{k}_matches_Venue_Wickets_sum',
           f'last_{k}_matches_Opposition_Innings Bowled_sum',
        #    f'last_{k}_matches_match_type_Wickets_sum',
        #    f'last_{k}_matches_Opposition_Wickets_sum',
          f'last_{k}_matches_derived_Economy Rate',
           f'last_{k}_matches_Venue_Innings Bowled_sum',
          f'last_{k}_matches_lbw_bowled_sum',
          f'last_{k}_matches_Bowleds_sum',
          f'last_{k}_matches_duck_outs_sum',
          f'last_{k}_matches_6wickets_sum',
          f'last_{k}_matches_4wickets_sum',
            f'last_{k}_matches_5wickets_sum'
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    if not return_tensor:
        return new_df
        
    X_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_tensor, new_df.columns



def data_augmentation(X, y):
    high_points_x = X[y > 0.5]
    high_points_y = y[y > 0.5]

