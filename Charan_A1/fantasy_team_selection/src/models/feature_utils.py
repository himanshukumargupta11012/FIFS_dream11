import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def process(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")

    new_df = df.iloc[:, data_start:]
    if "identifier" in new_df.columns :
        new_df = new_df.drop(['identifier'],axis = 1)
    # new_df = new_df.loc[:, ~new_df.columns.str.startswith('cumulative')]
    # new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'last_{k}_matches_derived')]
    new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'cumulative_derived_')]
    new_df = new_df.loc[:,~new_df.columns.str.startswith(f'hist_')]
   
    new_df = new_df.drop(
        [   
            # 'batting_order',
            'year',
            'Credit',
            'player_role',
            'in_oracle_team',
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
    #     'Total_matches_played_sum', 'last_7_matches_fantasy_points_sum',                                                                                                            
    #    'last_7_matches_fantasy_points_var', 'date_diff',                                                                                                                           
    #    'last_7_matches_Innings Batted_sum', 'last_7_matches_Runs_sum',                                                                                                             
    #    'last_7_matches_Fours_sum', 'last_7_matches_Sixes_sum',                                                                                                                     
    #    'last_7_matches_Maiden Overs_sum', 'last_7_matches_Runsgiven_sum',                                                                                                          
    #    'last_7_matches_Dot Balls Bowled_sum', 'last_7_matches_Stumpings_sum',                                                                                                      
    #    'last_7_matches_Catches_sum', 'last_7_matches_direct run_outs_sum',                                                                                                         
    #    'last_7_matches_indirect run_outs_sum', 'last_year_avg_Runs',                                                                                                               
    #    'last_year_avg_Wickets', 'Venue_total_matches_sum',                                                                                                                         
    #    'last_7_matches_Venue_Runs_sum', 'last_7_matches_Venue_Wickets_sum',                                                                                                        
    #    'last_7_matches_Venue_Innings Batted_sum',                                                                                                                                  
    #    'last_7_matches_Venue_Innings Bowled_sum',                                                                                                                                  
    #    'Opposition_total_matches_sum', 'last_7_matches_Opposition_Runs_sum',                                                                                                       
    #    'last_7_matches_Opposition_Wickets_sum',                                                                                                                                    
    #    'last_7_matches_Opposition_Innings Batted_sum',                                                                                                                             
    #    'last_7_matches_Opposition_Innings Bowled_sum',                                                                                                                             
    #    'match_type_total_matches_sum', 'last_7_matches_match_type_Runs_sum',                                                                                                       
    #    'last_7_matches_match_type_Wickets_sum',                                                                                                                                    
    #    'last_7_matches_match_type_Innings Batted_sum',                                                                                                                             
    #    'last_7_matches_match_type_Innings Bowled_sum',                                                                                                                             
    #    'Inning_total_matches_sum', 'last_7_matches_Inning_Runs_sum',                                                                                                               
    #    'last_7_matches_Inning_Wickets_sum',                                                                                                                                        
    #    'last_7_matches_Inning_Innings Batted_sum',                                                                                                                                 
    #    'last_7_matches_Inning_Innings Bowled_sum', 'venue_avg_runs_sum',                                                                                                           
    #    'venue_avg_wickets_sum'
    
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    # print(new_df.columns)
    if not return_tensor:
        return new_df
    
    new_df = new_df.fillna(0) # Filling the NaN values with 0 
    for col,dtype in new_df.dtypes.items() :
            if dtype == np.object_ : 
                print(col)
    
    X_train_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_train_tensor, new_df.columns

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

def compute_overlap_robust(true_tensor, pred_tensor, pred_match_id):
    """
    Computes overlap score handling ties at the 11th position boundary.

    Instead of taking strictly the top 11 indices, it finds the score
    threshold of the 11th player and includes all players scoring at
    or above that threshold for both predicted and actual points.

    Args:
        true_tensor: Numpy array of actual fantasy points.
        pred_tensor: Numpy array of predicted fantasy points.
        pred_match_id: Numpy array of corresponding match IDs.

    Returns:
        The average overlap count across all matches.
    """
    df = pd.DataFrame({
        'match_id': pred_match_id,
        'predicted_points': pred_tensor,
        'fantasy_points': true_tensor
    })

    overlap_counts = []
    for match_id, group in df.groupby('match_id'):
        n_players = len(group)
        # Ensure there are enough players to form a team of 11
        if n_players < 11:
            # print(f"Skipping match {match_id}: Only {n_players} players.")
            continue

        # --- Predicted Top Set ---
        # Sort by predicted points (descending), keep index
        group_sorted_pred = group.sort_values('predicted_points', ascending=False, kind='mergesort') # Use stable sort
        # Get the score of the 11th player (index 10)
        pred_cutoff_score = group_sorted_pred['predicted_points'].iloc[10]
        # Select indices of all players with score >= cutoff score
        predicted_top_indices = set(group_sorted_pred[group_sorted_pred['predicted_points'] >= pred_cutoff_score].index)

        # --- Actual Top Set ---
        # Sort by actual points (descending), keep index
        group_sorted_actual = group.sort_values('fantasy_points', ascending=False, kind='mergesort') # Use stable sort
        # Get the score of the 11th player (index 10)
        actual_cutoff_score = group_sorted_actual['fantasy_points'].iloc[10]
        # Select indices of all players with score >= cutoff score
        actual_top_indices = set(group_sorted_actual[group_sorted_actual['fantasy_points'] >= actual_cutoff_score].index)

        # --- Calculate Overlap ---
        # Find the intersection of the two sets of indices
        common_indices = predicted_top_indices.intersection(actual_top_indices)
        overlap_counts.append(len(common_indices))

        # --- Optional: Diagnostic Output ---
        # if len(predicted_top_indices) > 11 or len(actual_top_indices) > 11:
        #    print(f"Match {match_id}: Ties detected. Pred set size: {len(predicted_top_indices)}, Actual set size: {len(actual_top_indices)}, Overlap: {len(common_indices)}")

    if not overlap_counts:
        print("Warning: No matches with >= 11 players found for overlap calculation.")
        return 0.0

    # Return the average overlap size across eligible matches
    return sum(overlap_counts) / len(overlap_counts)




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
def compute_overlap_segmented(true_tensor, pred_tensor, pred_match_id):
    """
    Calculates overlap metrics specifically for the top and bottom halves
    of players based on *actual* performance within each match.

    Returns:
        tuple: (overlap_top_half, overlap_bottom_half) - average overlap scores
               within each segment across matches.
    """
    df = pd.DataFrame({
        'match_id': pred_match_id,
        'predicted_points': pred_tensor,
        'fantasy_points': true_tensor
    })

    overlaps_top = []
    overlaps_bottom = []

    for match_id, group in df.groupby('match_id'):
        n_players = len(group)
        if n_players < 2: continue # Need at least 2 players to split

        # Sort by ACTUAL points to define segments
        group_sorted_actual = group.sort_values('fantasy_points', ascending=False, kind='mergesort')

        # Define split point (handle odd numbers)
        mid_point = n_players // 2
        # Use indices from the sorted group
        top_half_actual_indices = set(group_sorted_actual.head(mid_point).index)
        # Ensure bottom half includes the middle element if n_players is odd
        bottom_half_actual_indices = set(group_sorted_actual.iloc[mid_point:].index)


        # Find players predicted to be in the top half (based on prediction score)
        group_sorted_pred = group.sort_values('predicted_points', ascending=False, kind='mergesort')
        top_half_pred_indices = set(group_sorted_pred.head(mid_point).index)
        # Bottom half predicted indices
        bottom_half_pred_indices = set(group_sorted_pred.iloc[mid_point:].index)

        # Calculate overlap WITHIN each segment
        overlap_top_count = len(top_half_actual_indices.intersection(top_half_pred_indices))
        overlap_bottom_count = len(bottom_half_actual_indices.intersection(bottom_half_pred_indices))

        # Normalize overlap by segment size (mid_point or n_players - mid_point)
        # Or just report raw counts averaged? Let's average raw counts for simplicity
        overlaps_top.append(overlap_top_count)
        overlaps_bottom.append(overlap_bottom_count)


    avg_overlap_top = np.mean(overlaps_top) if overlaps_top else 0.0
    avg_overlap_bottom = np.mean(overlaps_bottom) if overlaps_bottom else 0.0

    return avg_overlap_top, avg_overlap_bottom
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
    
        
    X_train_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_train_tensor, new_df.columns

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
        [   'Credits',
            # 'in_oracle_team'
            'batting_order',
            'player_role',
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
    
    X_train_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_train_tensor, new_df.columns

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
        
    X_train_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_train_tensor, new_df.columns

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
        
    X_train_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_train_tensor, new_df.columns