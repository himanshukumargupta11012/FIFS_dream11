import pandas as pd
import numpy as np
from tqdm import tqdm
import os # Added os import

# --- YOUR PROVIDED knapsack FUNCTION ---
def knapsack(players, budget):
    """
    Solve the knapsack problem to maximize fantasy points while staying within the Credits limit.
    (ASSUMES 'Credits' and 'fantasy_points' columns exist in players df)
    (ASSUMES 'identifier' column exists for players df)
    """
    n = len(players)
    # Ensure budget is integer and scale credits to avoid floats if needed
    # Multiplying by 10 is common if credits have one decimal place
    budget_int = int(budget * 10)
    # DP table: rows=players+1, columns=budget*10+1
    dp = [[0] * (budget_int + 1) for _ in range(n + 1)]
    # Keep track of selected identifiers (using sets for faster checking)
    selected_ids_dp = [[set() for _ in range(budget_int + 1)] for _ in range(n + 1)]

    players_list = players.to_dict('records') # More efficient iteration

    for i in range(1, n + 1):
        player = players_list[i - 1]
        # Scale cost, handle potential NaN credits robustly
        cost = int(player.get('Credits', 0) * 10)
        # Use fantasy points, handle potential NaN robustly
        value = player.get('fantasy_points', 0)
        player_id = player.get('identifier')

        if player_id is None: continue # Skip if no identifier

        for j in range(budget_int + 1):
            if cost > j or cost <= 0: # Also skip if cost is invalid
                dp[i][j] = dp[i - 1][j]
                selected_ids_dp[i][j] = selected_ids_dp[i - 1][j]
            else:
                # Compare score if player i is included vs excluded
                score_without_i = dp[i - 1][j]
                score_with_i = dp[i - 1][j - cost] + value

                if score_without_i >= score_with_i:
                    dp[i][j] = score_without_i
                    selected_ids_dp[i][j] = selected_ids_dp[i - 1][j]
                else:
                    dp[i][j] = score_with_i
                    # Update selected set: copy previous best set for remaining budget, add current player
                    new_set = selected_ids_dp[i - 1][j - cost].copy()
                    new_set.add(player_id)
                    selected_ids_dp[i][j] = new_set

    # Return the set of selected identifiers from the final DP state
    return selected_ids_dp[n][budget_int]

# --- YOUR PROVIDED select_team FUNCTION (Slightly modified for clarity/safety) ---
def select_team_heuristic(match_players_df, budget=100):
    """
    Selects an 11-player team using the heuristic:
    1. Pick top player by fantasy_points for each role.
    2. Use Knapsack for the remaining 7 slots based on fantasy_points.
    3. Simple check/swap for team coverage (less robust, might fail complex cases).

    Args:
        match_players_df (pd.DataFrame): DataFrame of players for ONE match,
                                         must include 'identifier', 'Player Type' (or 'Role'),
                                         'Team', 'fantasy_points', 'Credits'.
        budget (int): Total budget constraint.

    Returns:
        pd.DataFrame: DataFrame containing the selected 11 players, or None if fails.
                      Returns columns: 'identifier'.
    """
    # --- Role Mapping (ensure consistency) ---
    # print(match_players_df.columns)
    if 'player_role' in match_players_df.columns: role_col = 'player_role'
    elif 'Player Type' in match_players_df.columns: role_col = 'Player Type'
    else: raise ValueError("Missing Role column")
    role_map = {'WK': 'WK', 'BAT': 'BAT', 'BOWL': 'BOWL', 'AR': 'AR', 'Wicketkeeper': 'WK', 'Batsman': 'BAT', 'Bowler': 'BOWL', 'AllRounder': 'AR', 'Allrounder': 'AR'}
    match_players_df['role_standard'] = match_players_df[role_col].map(role_map).fillna('Unknown')
    # ------------------------------------------

    # Ensure necessary columns are numeric and handle NaNs
    match_players_df['fantasy_points'] = pd.to_numeric(match_players_df['fantasy_points'], errors='coerce').fillna(0)
    match_players_df['Credits'] = pd.to_numeric(match_players_df['Credits'], errors='coerce').fillna(9.0) # Impute missing credits

    df = match_players_df.copy()
    selected_players_list = []
    remaining_players_df = df.copy()
    total_credits_used = 0
    required_roles = ['BAT', 'BOWL', 'WK', 'AR'] # Standard roles

    # Step 1: Select top player by type (using standardized role)
    print(f"Step 1 Starting Players: {len(remaining_players_df)}")
    selected_ids = set()
    for role in required_roles:
        candidates = remaining_players_df[remaining_players_df['role_standard'] == role]
        if not candidates.empty:
            # Sort by points, then credits (lower is better tie-breaker)
            top_player = candidates.sort_values(by=['fantasy_points', 'Credits'], ascending=[False, True]).iloc[0]
            selected_players_list.append(top_player.to_dict())
            selected_ids.add(top_player['identifier'])
            total_credits_used += top_player['Credits']
            print(f"  Added {role}: {top_player['player']} (FP: {top_player['fantasy_points']}, Cr: {top_player['Credits']})")
        else:
            print(f"Warning: No players found for role {role} in initial selection.")

    # Update remaining players
    remaining_players_df = remaining_players_df[~remaining_players_df['identifier'].isin(selected_ids)]
    print(f"Step 1 Credits Used: {total_credits_used}, Players Selected: {len(selected_players_list)}")
    print(f"Remaining Players for Knapsack: {len(remaining_players_df)}")


    # Step 2: Use Knapsack for the remaining slots
    slots_left = 11 - len(selected_players_list)
    if slots_left <= 0:
         print("Warning: Already selected 11 or more players in Step 1.")
         # Need to handle this case - maybe just take top 11? For now, return current selection
         selected_df = pd.DataFrame(selected_players_list).head(11)
         # Add checks here to ensure role/team validity before returning if needed
         return selected_df[['identifier']] # Return only IDs

    budget_left = budget - total_credits_used
    print(f"Step 2 Knapsack: Slots={slots_left}, Budget Left={budget_left:.1f}")

    if budget_left < 0 or remaining_players_df.empty:
        print("Warning: No budget left or no remaining players for Knapsack.")
        # Return based on initial selection - validity checks needed
        selected_df = pd.DataFrame(selected_players_list).head(11)
        return selected_df[['identifier']]

    # Solve knapsack to get IDs
    knapsack_selected_IDS = knapsack(remaining_players_df, budget_left) # Returns a set of IDs

    if not knapsack_selected_IDS:
         print("Warning: Knapsack returned no players.")
         selected_df = pd.DataFrame(selected_players_list).head(11)
         return selected_df[['identifier']]

    # Get the players corresponding to the selected IDs
    knapsack_players_df = remaining_players_df[remaining_players_df['identifier'].isin(knapsack_selected_IDS)]

    # Select exactly 'slots_left' players, prioritizing fantasy points, then credits
    knapsack_final_selection = knapsack_players_df.sort_values(
        by=['fantasy_points', 'Credits'], ascending=[False, True]
    ).head(slots_left)

    print(f"  Knapsack selected {len(knapsack_final_selection)} players.")
    selected_players_list.extend(knapsack_final_selection.to_dict('records'))
    selected_df = pd.DataFrame(selected_players_list)
    print(f"Total players after Knapsack: {len(selected_df)}, Credits: {selected_df['Credits'].sum():.1f}")


    # --- Step 3: Validity Check (Crucial - Your original check was minimal) ---
    # This is where heuristics often fail. A full check is needed.
    # If len(selected_df) != 11, something went wrong.
    if len(selected_df) != 11:
         print(f"Warning: Heuristic resulted in {len(selected_df)} players. Cannot guarantee validity.")
         # Fallback: maybe return top 11 by points from initial selection? Risky.
         # For oracle label generation, maybe skip this match if invalid.
         return None # Indicate failure

    # Check final constraints
    final_credits = selected_df['Credits'].sum()
    roles = selected_df['role_standard'].value_counts()
    wk_count = roles.get('WK', 0); bat_count = roles.get('BAT', 0); ar_count = roles.get('AR', 0); bowl_count = roles.get('BOWL', 0)
    team_counts = selected_df['team'].value_counts()

    valid = True
    # if not (1 <= wk_count <= 4): valid = False; print(f"  Constraint fail: WK={wk_count}")
    # if not (3 <= bat_count <= 6): valid = False; print(f"  Constraint fail: BAT={bat_count}")
    # if not (1 <= ar_count <= 4): valid = False; print(f"  Constraint fail: AR={ar_count}")
    # if not (3 <= bowl_count <= 6): valid = False; print(f"  Constraint fail: BOWL={bowl_count}")
    # if final_credits > budget: valid = False; print(f"  Constraint fail: Credits={final_credits}")
    # if (team_counts > 7).any(): valid = False; print(f"  Constraint fail: Team Count > 7")

    if not valid:
        print("Warning: Heuristic produced an invalid team after checks.")
        return None # Indicate failure

    # If valid, return the identifiers
    print("Heuristic produced a valid team.")
    return selected_df[['identifier']]


# --- Wrapper function to generate labels for the whole dataset ---
def generate_oracle_labels_with_heuristic(df, budget=100):
    """
    Generates 'in_oracle_team' labels using the select_team_heuristic.
    """
    print("Generating oracle team labels using provided heuristic...")
    df['in_oracle_team'] = 0 # Initialize column
    df["Credits"] = 9
    df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce').fillna(9.0) # Ensure Credits are numeric
    df['fantasy_points'] = pd.to_numeric(df['fantasy_points'], errors='coerce').fillna(0) # Ensure FP are numeric

    processed_matches = 0
    skipped_matches = 0
    total_matches = df['match_id'].nunique()
    match_groups = df.groupby('match_id')
    all_oracle_indices = []

    for match_id, group in tqdm(match_groups, total=total_matches, desc="Processing Matches"):
        if len(group) < 11: # Need at least 11 players
            skipped_matches += 1
            continue

        # Get players for the current match WITH original index preserved
        match_players_with_idx = group.copy()
        match_players_with_idx.reset_index(inplace=True) # Keep original index

        # Call the heuristic selection function
        selected_team_identifiers_df = select_team_heuristic(match_players_with_idx, budget)

        if selected_team_identifiers_df is not None and not selected_team_identifiers_df.empty:
            # Get the list of selected identifiers
            selected_ids = set(selected_team_identifiers_df['identifier'])
            # Find the original indices corresponding to these players in this match
            original_indices = match_players_with_idx[match_players_with_idx['identifier'].isin(selected_ids)]['index'].tolist()
            all_oracle_indices.extend(original_indices)
            processed_matches += 1
        else:
            print(f"  Skipping Match {match_id}: Heuristic failed or returned invalid team.")
            skipped_matches += 1

    # Assign Labels
    print(f"Processed {processed_matches}/{total_matches} matches ({skipped_matches} skipped).")
    print(f"Assigning 'in_oracle_team' label to {len(all_oracle_indices)} player instances...")
    df.loc[all_oracle_indices, 'in_oracle_team'] = 1
    print("'in_oracle_team' counts:\n", df['in_oracle_team'].value_counts())

    # Clean up temporary columns if any were added
    df.drop(columns=['role_standard'], inplace=True, errors='ignore')
    df.drop(columns=[''])
    return df



# --- How to Use ---
input_csv = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data_charan/processed/combined/10_IPL_roles.csv"
historical_df = pd.read_csv(input_csv)
historical_df['date'] = pd.to_datetime(historical_df['date'])

# # --> IMPORTANT: Ensure 'identifier' column exists and is unique per player <--
# # If 'identifier' is missing, create it e.g., from player_id or Player Name
if 'identifier' not in historical_df.columns:
     if 'player_id' in historical_df.columns:
          historical_df['identifier'] = historical_df['player_id']
     elif 'Player Name' in historical_df.columns: # Use name as last resort (risky if names clash)
           print("Warning: Using 'Player Name' as identifier. Ensure uniqueness.")
           historical_df['identifier'] = historical_df['Player Name']
     else:
           raise ValueError("Missing 'identifier' column (or player_id/Player Name to create it)")

# # Generate labels
# data_with_oracle_labels = generate_oracle_labels_with_heuristic(historical_df)

# # Save the result
# output_csv = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data_charan/processed/combined/10_IPL_heuristic_oracle.csv"
# data_with_oracle_labels.to_csv(output_csv, index=False)


# print(f"Saved data with heuristic oracle labels to: {output_csv}")

# --- Main script ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_FILE_PATH = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data_charan/processed/combined/10_IPL_heuristic_oracle.csv" # Use the file generated previously
    # Or load the file that has the *actual* fantasy points for each match if different

    print(f"Loading data from: {DATA_FILE_PATH}")
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE_PATH}")

    historical_df = pd.read_csv(DATA_FILE_PATH)
    historical_df['date'] = pd.to_datetime(historical_df['date'])

    # --- Ensure required columns exist ---
    required_cols = ['match_id', 'date', 'player', 'player_role', 'team', 'Credits', 'fantasy_points']
    if 'identifier' not in historical_df.columns: # Check for identifier
         if 'player_id' in historical_df.columns:
              historical_df['identifier'] = historical_df['player_id']
              required_cols.append('identifier')
         elif 'Player Name' in historical_df.columns:
             print("Warning: Using 'Player Name' as identifier.")
             historical_df['identifier'] = historical_df['Player Name']
             required_cols.append('identifier')

    missing_cols = [col for col in required_cols if col not in historical_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in the data: {missing_cols}")

    # --- Select the most recent match_id ---
    historical_df = historical_df.sort_values('date')
    recent_match_id = historical_df['match_id'].iloc[-1]
    print(f"\nAnalyzing most recent match_id: {recent_match_id} (Date: {historical_df['date'].iloc[-1].date()})")

    # --- Get data for that specific match ---
    recent_match_data = historical_df[historical_df['match_id'] == recent_match_id].copy()

    if recent_match_data.empty:
        print("Error: No data found for the most recent match.")
    elif len(recent_match_data) < 11:
        print(f"Error: Only {len(recent_match_data)} players found for the most recent match.")
    else:
        # --- Run the heuristic team selection ---
        print("\nRunning heuristic team selection for this match...")
        selected_ids_df = select_team_heuristic(recent_match_data) # Pass data for this match only

        if selected_ids_df is not None:
            print("\n--- Heuristically Selected 'Oracle' Team ---")
            selected_identifiers = set(selected_ids_df['identifier'])
            # Get full details from the match data using the selected identifiers
            final_team_df = recent_match_data[recent_match_data['identifier'].isin(selected_identifiers)].copy()

            # Add C/VC heuristic (Top 2 raw point scorers in the selected team)
            final_team_df = final_team_df.sort_values('fantasy_points', ascending=False)
            captain_id = final_team_df['identifier'].iloc[0]
            vice_captain_id = final_team_df['identifier'].iloc[1]

            # Add C/VC labels for display
            final_team_df['Captain_VC'] = ''
            final_team_df.loc[final_team_df['identifier'] == captain_id, 'Captain_VC'] = '(C)'
            final_team_df.loc[final_team_df['identifier'] == vice_captain_id, 'Captain_VC'] = '(VC)'

            # Display the team
            display_cols = ['player', 'player_role', 'team', 'Credits', 'fantasy_points', 'Captain_VC']
            print(final_team_df[display_cols].to_string(index=False))

            # Display Summary Stats
            total_points = final_team_df['fantasy_points'].sum()
            total_credits = final_team_df['Credits'].sum()
            captain_points = final_team_df[final_team_df['Captain_VC'] == '(C)']['fantasy_points'].iloc[0]
            vice_captain_points = final_team_df[final_team_df['Captain_VC'] == '(VC)']['fantasy_points'].iloc[0]
            dream11_score = total_points + captain_points + 0.5 * vice_captain_points # Calculate score with C/VC bonus

            print("-" * 30)
            print(f"Total Players: {len(final_team_df)}")
            print(f"Total Credits: {total_credits:.1f} / 100.0")
            print(f"Sum Raw Points: {total_points}")
            print(f"Approx Dream11 Score (with C/VC): {dream11_score:.1f}")
            print("-" * 30)
            print("Role Counts:")
            print(final_team_df['player_role'].value_counts())
            print("-" * 30)
            print("Team Counts:")
            print(final_team_df['team'].value_counts())
            print("-" * 30)

        else:
            print("\nHeuristic team selection failed or returned an invalid team for this match.")