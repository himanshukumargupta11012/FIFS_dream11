import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def classify_player_role_heuristic(row):
    """
    Assigns a heuristic player role based on cumulative historical stats in the row.
    Uses RELAXED thresholds to identify more Bowlers and Allrounders in T20 context.
    """
    # --- RELAXED Thresholds ---
    WK_STUMPING_THRESHOLD = 0       # Keep this - any stumping means WK

    # Lowered Bowler thresholds
    BOWLER_PERC_MATCHES_THRESH = 0.60 # Was 0.65 - bowl in slightly fewer games %
    BOWLER_AVG_BALLS_THRESH = 10    # Was 12 - Avg ~1.5 overs instead of 2

    # Slightly adjust Batter thresholds (optional)
    BATTER_PERC_MATCHES_THRESH = 0.75 # Keep similar
    BATTER_AVG_BALLS_THRESH = 8     # Was 10 - slightly lower avg balls faced

    # Significantly Lowered Moderate thresholds to qualify for Allrounder checks
    MODERATE_BOWLING_PERC = 0.15    # Was 0.25 - Bowls in >= 15% of matches
    MODERATE_BOWLING_BALLS = 4      # Was 6 - Avg >= 4 balls / match
    MODERATE_BATTING_PERC = 0.20    # Was 0.30 - Bats in >= 20% of matches
    MODERATE_BATTING_BALLS = 4      # Was 5 - Avg >= 4 balls faced / match

    # Adjusted Allrounder Distinction (Make middle category slightly wider)
    ALLROUNDER_HIGH_RATIO = 1.6  # Ratio for clear Batting/Bowling Allrounder (was 1.5)
    ALLROUNDER_LOW_RATIO = 1 / ALLROUNDER_HIGH_RATIO # ~0.625

    # --- Access Data Safely ---
    total_matches = row.get('Total_matches_played_sum', 0)

    # --- Handle players with very little history ---
    if total_matches < 5: # Increased minimum matches slightly for better pattern
        batting_order = row.get('batting_order', 99)
        # Still a weak guess, but leans more towards role based on order
        return 'Wicket Keeper' if row.get('cumulative_Stumpings_sum', 0) > 0 else \
               'Batter' if batting_order <= 6 else \
               'Bowler' if batting_order >= 9 else \
               'Allrounder' # Default AR for mid-order low history
        # Previous default: return 'Batter'

    # --- Get Cumulative Stats from the row ---
    stumpings = row.get('cumulative_Stumpings_sum', 0)
    innings_bowled = row.get('cumulative_Innings Bowled_sum', 0)
    balls_bowled = row.get('cumulative_Balls Bowled_sum', 0)
    innings_batted = row.get('cumulative_Innings Batted_sum', 0)
    balls_faced = row.get('cumulative_Balls Faced_sum', 0)

    # --- 1. Identify Wicket Keeper ---
    if stumpings > WK_STUMPING_THRESHOLD:
        return "Wicket Keeper" # Keep WK as highest priority check

    # --- 2. Calculate Significance Metrics (Based on cumulative stats) ---
    perc_matches_bowled = innings_bowled / total_matches
    avg_balls_bowled = balls_bowled / total_matches
    perc_matches_batted = innings_batted / total_matches
    avg_balls_faced = balls_faced / total_matches

    # --- 3. Check for Moderate Contributions FIRST (for Allrounders) ---
    is_moderate_bowler = (perc_matches_bowled >= MODERATE_BOWLING_PERC) or (avg_balls_bowled >= MODERATE_BOWLING_BALLS)
    is_moderate_batter = (perc_matches_batted >= MODERATE_BATTING_PERC) or (avg_balls_faced >= MODERATE_BATTING_BALLS)

    if is_moderate_bowler and is_moderate_batter:
        # Potential Allrounder - check contribution balance
        bat_contribution = perc_matches_batted + (avg_balls_faced / 50) # Combine freq and volume (heuristic weighting)
        bowl_contribution = perc_matches_bowled + (avg_balls_bowled / 24) # Combine freq and volume

        if bowl_contribution == 0 and bat_contribution == 0: # Edge case
             ratio = 1.0
        elif bat_contribution == 0: # Avoid division by zero if only bowls
             ratio = float('inf') # Definitely bowling leaning
        else:
             ratio = bowl_contribution / bat_contribution

        if ratio > ALLROUNDER_HIGH_RATIO:
            return 'Bowling Allrounder'
        elif ratio < ALLROUNDER_LOW_RATIO:
            return 'Batting Allrounder'
        else:
            # Contributions are relatively balanced
            return 'Allrounder'

    # --- 4. If not WK or AR, check for Pure Roles ---
    is_significant_bowler = (perc_matches_bowled >= BOWLER_PERC_MATCHES_THRESH) or \
                            (avg_balls_bowled >= BOWLER_AVG_BALLS_THRESH and perc_matches_bowled > 0.4) # Slightly relaxed frequency req for high volume
    is_significant_batter = (perc_matches_batted >= BATTER_PERC_MATCHES_THRESH) or \
                            (avg_balls_faced >= BATTER_AVG_BALLS_THRESH and perc_matches_batted > 0.4)

    # Assign pure role only if they lack even moderate contribution in the other discipline
    if is_significant_bowler and not is_moderate_batter: return 'Bowler'
    if is_significant_batter and not is_moderate_bowler: return 'Batter'

    # --- 5. Fallbacks based on any contribution ---
    # If they weren't AR, but have *some* contribution in both, default based on dominance
    if is_moderate_bowler and is_moderate_batter: # Should have been caught by AR logic, but as safety
         bat_contribution = perc_matches_batted; bowl_contribution = perc_matches_bowled
         return 'Bowling Allrounder' if bowl_contribution > bat_contribution else 'Batting Allrounder'

    if is_moderate_bowler: return 'Bowler' # If only moderate bowling -> Bowler
    if is_moderate_batter: return 'Batter' # If only moderate batting -> Batter

    # --- 6. Ultimate Default ---
    # If minimal history led to no classification, use batting order again or default
    batting_order = row.get('batting_order', 99)
    return 'Batter' if batting_order <= 7 else 'Bowler' if batting_order >= 9 else 'Allrounder' # Default AR if mid order

# --- Function to Assign HEURISTIC Roles and MAP TO STANDARD ---
def assign_heuristic_player_roles(feature_df):
    """
    Assigns a player role based purely on heuristic from last occurrence stats,
    then maps it to the standard 4 categories ('BAT', 'BOWL', 'WK', 'AR').
    Creates or OVERWRITES the 'player_role' column with the standard roles.

    Args:
        feature_df (pd.DataFrame): DataFrame with features including 'player_id', 'date',
                                   and cumulative stats needed by the heuristic.

    Returns:
        pd.DataFrame: DataFrame with a consistent 'player_role' column containing
                      only 'BAT', 'BOWL', 'WK', 'AR'.
    """
    # --- Check required columns ---
    required_cols = ['player_id', 'date', 'Total_matches_played_sum', 'cumulative_Stumpings_sum',
                     'cumulative_Innings Bowled_sum', 'cumulative_Balls Bowled_sum',
                     'cumulative_Innings Batted_sum', 'cumulative_Balls Faced_sum']
    missing_req_cols = [col for col in required_cols if col not in feature_df.columns]
    if missing_req_cols:
        print(f"Error: Missing required columns for heuristic: {missing_req_cols}")
        # Optionally add default role column before returning
        if 'player_role' not in feature_df.columns: feature_df['player_role'] = 'BAT'
        return feature_df

    # --- 1. Find last occurrence ---
    print("Finding last match occurrence for each player...")
    last_occurrence_df = feature_df.sort_values(['player_id', 'date'], ascending=[True, False]) \
                                   .drop_duplicates(subset='player_id', keep='first')
    if last_occurrence_df.empty:
        print("Warning: Could not determine last occurrences.")
        if 'player_role' not in feature_df.columns: feature_df['player_role'] = 'BAT'
        return feature_df

    # --- 2. Apply DETAILED Heuristic ---
    print("Applying detailed role heuristic based on last occurrence stats...")
    last_occurrence_df['detailed_heuristic_role'] = last_occurrence_df.apply(classify_player_role_heuristic, axis=1)

    # --- 3. Create Player-to-DETAILED-Role Mapping ---
    detailed_role_map = last_occurrence_df.set_index('player_id')['detailed_heuristic_role'].to_dict()

    # --- 4. Assign DETAILED Heuristic Role Temporarily ---
    feature_df['temp_detailed_role'] = feature_df['player_id'].map(detailed_role_map)

    # --- 5. Define Mapping to Standard Roles ---
    standard_role_map = {
        'Batter': 'BAT',
        'Wicket Keeper': 'WK',
        'Bowler': 'BOWL',
        'Allrounder': 'AR',
        'Batting Allrounder': 'AR',
        'Bowling Allrounder': 'AR'
    }

    # --- 6. Map to Standard Roles and Create/Overwrite 'player_role' ---
    print(f"Mapping detailed roles to standard roles (BAT, BOWL, WK, AR)...")
    # Apply the standard map to the temporary detailed role column
    # Use .map() and fill any NaNs (from missing players in map or unmapped detailed roles) with a default
    feature_df['player_role'] = feature_df['temp_detailed_role'].map(standard_role_map).fillna('BAT') # Default to BAT if mapping fails

    # --- 7. Clean up temporary column ---
    feature_df.drop(columns=['temp_detailed_role'], inplace=True)

    print("Finished assigning standard heuristic player roles.")
    print("Final Standard Role Distribution:\n", feature_df['player_role'].value_counts())

    return feature_df

# --- How to Apply ---
# Assuming 'feature_df' is your dataframe AFTER calculating all cumulative/rolling features
# Ensure it contains 'player_id', 'date', and all cumulative columns used by the heuristic.

# Example:
feature_df = pd.read_csv("/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data_charan/processed/combined/10_IPL.csv")
feature_df = assign_heuristic_player_roles(feature_df) # Now contains standard roles

feature_df.to_csv("/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data_charan/processed/combined/10_IPL_roles.csv",index = False)

# # Proceed with one-hot encoding
# feature_df = pd.get_dummies(feature_df, columns=['player_role'], prefix='role', dummy_na=False)


print("\n--- Role Assignment Verification ---")

# List some well-known players expected to fall into different categories
players_to_check = {
    'Bowlers': ['JJ Bumrah', 'Rashid Khan', 'YS Chahal', 'Mohammed Shami', 'TA Boult', 'K Rabada'],
    'Allrounders': ['RA Jadeja', 'HH Pandya', 'AD Russell', 'MP Stoinis', 'GJ Maxwell', 'MM Ali', 'Shakib Al Hasan'],
    'Batters': ['V Kohli', 'RG Sharma', 'DA Warner', 'KL Rahul', 'JC Buttler', 'F du Plessis'], # Excluding WKs initially
    'WicketKeepers': ['MS Dhoni', 'RR Pant', 'Q de Kock', 'Ishan Kishan', 'SV Samson'] # Should be caught by WK check mostly
}

# Create a dataframe containing only the LAST row for each player (most representative stats)
verification_df = feature_df.sort_values(['player_id', 'date'], ascending=[True, False]) \
                           .drop_duplicates(subset='player_id', keep='first')

# print(verification_df.columns)
# Display assigned role for each player in the lists
# for category, names in players_to_check.items():
#     print(f"\nChecking Category: {category}")
#     # Use 'Player Name' column if it exists, otherwise use 'player_name'
#     name_col = 'Player' if 'Player' in verification_df.columns else 'player_name'
#     if name_col not in verification_df.columns:
#          print(f"  Cannot verify - Missing name column ('{name_col}')")
#          continue

#     players_found_df = verification_df[verification_df[name_col].isin(names)][[name_col, 'player_role']]
#     if not players_found_df.empty:
#         print(players_found_df.to_string(index=False))
#     else:
#         print(f"  None of the example players found in the data for category: {category}")

# print("-" * 30)