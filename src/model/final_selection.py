
import pandas as pd
from itertools import combinations

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

def select_team(csv_file, optional_df=None):
    """
    The columns of optional_df are expected to be
    'identifier', 'Player Name', 'Player Type', 'Team', 'fantasy_points', 'Credits'
    """
    if optional_df is None:
        df = pd.read_csv(csv_file)
    else:
        df = optional_df

    # Step 1: Select top players by type
    selected_players = []
    remaining_players = df.copy()
    total_credits_used = 0
    
    for player_type in ['BAT', 'BOWL', 'WK', 'ALL']:
        top_player = df[df['Player Type'] == player_type].nlargest(1, 'fantasy_points')
        if not top_player.empty:
            selected_players.append(top_player.iloc[0].to_dict())
            remaining_players = remaining_players[remaining_players['identifier'] != top_player.iloc[0]['identifier']]
            total_credits_used += top_player.iloc[0]['Credits']

    # Step 2: Use Knapsack for the remaining 7 players
    budget_left = 100 - total_credits_used
    knapsack_selected_IDS = knapsack(remaining_players, budget_left)
    knapsack_selected = remaining_players[remaining_players['identifier'].isin(knapsack_selected_IDS)]
    if len(knapsack_selected) > 7:
        knapsack_selected = knapsack_selected.nlargest(7, "fantasy_points")
    selected_players.extend(knapsack_selected.to_dict('records'))  # Convert to list of dictionaries
    
    # Step 3: Ensure at least one player from each team
    selected_df = pd.DataFrame(selected_players)
    teams_covered = set(selected_df['Team'])
    missing_teams = set(df['Team']) - teams_covered
    
    # Remove the least performing player from the existing team , recalculate the credits remaining and then add the best player from the missing team with the remaining credits
    for team in missing_teams:
        team_players = df[df['Team'] == team]
        worst_player = selected_df[selected_df['Team'] == team].nsmallest(1, 'fantasy_points')
        remaining_credits = 100 - selected_df['Credits'].sum() + worst_player['Credits'].values[0]
        best_player = team_players[team_players['Credits'] <= remaining_credits].nlargest(1, 'fantasy_points')
        selected_df = selected_df[selected_df['identifier'] != worst_player['identifier'].values[0]]
        selected_df = selected_df.append(best_player)        

    
    return selected_df[['identifier', 'Player Type',"Player Name", 'Team', 'fantasy_points', 'Credits']]



if __name__ == "__main__":
    print(select_team("output.csv"))