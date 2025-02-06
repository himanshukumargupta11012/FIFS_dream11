import google.generativeai as genai
from dotenv import load_dotenv
import os
import torch
from captum.attr import DeepLift, IntegratedGradients, GradientShap
import random
import pandas as pd

# Load API key
load_dotenv(dotenv_path=".env")
key = os.getenv("GENAI_API_KEY")

genai.configure(api_key=key)
llm_model = genai.GenerativeModel("gemini-1.5-flash")

def format_prompt(results, players_id_list):
    """
    Format the LLM prompt for the concatenated explanations for all rows in results.
    """
    separator = "\n---\n"  # Custom separator between player explanations
    combined_prompts = []
    for i, result in enumerate(results):
        # Build the features section for the current player
        features_str = f"{players_id_list[i]}\n"
        features_str += "**Selected**\n" if i < len(results)/2 else "**Not Selected**\n"
        for i, (feature, score) in enumerate(result.items()):
            feature_name = feature if isinstance(feature, str) else ", ".join(feature)  # Handle interaction terms
            features_str += f"{i+1}. **Feature:** {feature_name}\n" \
                            f"   - **Type:** {'Interaction' if isinstance(feature, tuple) else 'Main'}\n" \
                            f"   - **Importance Score:** {score}\n"

        # Add the explanation prompt for the current player
        player_prompt = f"**Prediction Explanation for Fantasy Score**\n\n" \
                        f"**Top contributing features:**\n\n" \
                        f"{features_str}"
        combined_prompts.append(player_prompt)

    # Join all player prompts with the separator
    return f"{separator}".join(combined_prompts) + f"Give priority to the features with lower index since they have a bigger score, and output the individual text explanation for each player, separated by a separator \"{separator}\", in just 2-3 lines, without mentioning the technical details related to the model (for example, don't directly mention the actual feature names, you might give the english understanding of it), focus only on a genuine cricket explanation. Also, there is no need for telling which row's collective response you give, if you output per row one by one, even with repetitions. Note that you have to compare the features with the other players and give positive explanation for first 11 players and negative explanation for the last 11 players."

def generate_explainability_text(player_name, feature1, feature2, feature3, positivity):
    positive_templates = [
        f"{player_name}'s high predicted fantasy score is attributed to their {feature1}, {feature2}, and {feature3}.",
        f"Key factors like {feature1}, {feature2}, and {feature3} significantly contribute to {player_name}'s performance prediction.",
        f"The model identifies {feature1}, {feature2}, and {feature3} as crucial metrics for {player_name}'s projected success.",
        f"{player_name}'s standout performance metrics in {feature1}, {feature2}, and {feature3} drive their fantasy score prediction.",
        f"With strengths in {feature1}, {feature2}, and {feature3}, {player_name} emerges as a top contender for high fantasy points."
    ]
    negative_templates = [
        f"{player_name}'s strong performance in {feature1}, {feature2}, and {feature3} was commendable, but competition was tough, leading to their exclusion.",
        f"With high marks in {feature1}, {feature2}, and {feature3}, {player_name} was close to making the team but was ultimately not selected.",
        f"{player_name} had great metrics in {feature1}, {feature2}, and {feature3}, but others outperformed in critical areas.",
        f"Despite excelling in {feature1}, {feature2}, and {feature3}, {player_name} could not secure a spot on the team due to stiff competition.",
        f"While {player_name} stood out for {feature1}, {feature2}, and {feature3}, their overall profile didn't meet the team's needs this time."
    ] 
    if positivity == 1:
        return random.choice(positive_templates)
    else:
        return random.choice(negative_templates)

def backup_explanations(results, players_id_list, k=15):
    feature_descriptions = {
        'Total_matches_played_sum': f"number of innings played across all matches",
        f'last_{k}_matches_Fours_sum': f"fours hit in the last {k} matches",
        f'last_{k}_matches_Sixes_sum': f"sixes scored in the last {k} matches",
        f'last_{k}_matches_Outs_sum': f"times dismissed in the last {k} matches",
        f'last_{k}_matches_fantasy_points_sum': f"fantasy points accumulated in the last {k} matches",
        f'last_{k}_matches_Dot Balls_sum': f"dot balls faced in the last {k} matches",
        f'last_{k}_matches_Balls Faced_sum': f"total balls faced in the last {k} matches",
        f'last_{k}_matches_Innings Bowled_sum': f"innings bowled in the last {k} matches",
        f'last_{k}_matches_Balls Bowled_sum': f"balls delivered in the last {k} matches",
        f'last_{k}_matches_derived_Dot Ball%': f"percentage of dot balls in the last {k} matches",
        f'last_{k}_matches_derived_Batting Strike Rate': f"strike rate in the last {k} matches",
        f'last_{k}_matches_derived_Batting Avg': f"batting average in the last {k} matches",
        f'last_{k}_matches_derived_Mean Score': f"average score across the last {k} matches",
        f'last_{k}_matches_derived_Boundary%': f"percentage of runs from boundaries in the last {k} matches",
        f'last_{k}_matches_derived_Mean Balls Faced': f"average balls faced per match in the last {k} matches",
        f'last_{k}_matches_derived_Dismissal Rate': f"rate of dismissals in the last {k} matches",
        f'last_{k}_matches_derived_Bowling Dot Ball%': f"percentage of dot balls bowled in the last {k} matches",
        f'last_{k}_matches_derived_Boundary Given%': f"percentage of runs conceded from boundaries in the last {k} matches",
        f'last_{k}_matches_derived_Bowling Avg': f"average runs conceded per wicket in the last {k} matches",
        f'last_{k}_matches_derived_Bowling Strike Rate': f"deliveries per wicket in the last {k} matches",
        'Opponent_total_matches_sum': f"matches played against the opponent",
        'Venue_total_matches_sum': f"matches played at the venue",
        f'last_{k}_matches_Runsgiven_sum': f"runs conceded in the last {k} matches",
        f'last_{k}_matches_Dot Balls Bowled_sum': f"dot balls bowled in the last {k} matches",
        f'last_{k}_matches_Foursgiven_sum': f"fours conceded in the last {k} matches",
        f'last_{k}_matches_Sixesgiven_sum': f"sixes conceded in the last {k} matches",
        f'venue_avg_runs_sum': f"average runs scored at the venue",
        f'venue_avg_wickets_sum': f"average wickets taken at the venue",
        f'last_{k}_matches_Extras_sum': f"extras conceded in the last {k} matches",
        f'last_{k}_matches_centuries_sum': f"centuries scored in the last {k} matches",
        f'last_{k}_matches_half_centuries_sum': f"half-centuries scored in the last {k} matches",
        f'last_{k}_matches_Opposition_Runs_sum': f"runs scored against the opponent in the last {k} matches",
        f'last_{k}_matches_Venue_Runs_sum': f"runs scored at the venue in the last {k} matches",
        f'last_{k}_matches_Wickets_sum': f"wickets taken in the last {k} matches",
        f'last_{k}_matches_LBWs_sum': f"LBWs in the last {k} matches",
        f'last_{k}_matches_Maiden Overs_sum': f"maiden overs bowled in the last {k} matches",
        f'last_{k}_matches_Stumpings_sum': f"stumpings in the last {k} matches",
        f'last_{k}_matches_Catches_sum': f"catches taken in the last {k} matches",
        f'last_{k}_matches_direct run_outs_sum': f"direct run-outs in the last {k} matches",
        f'last_{k}_matches_indirect run_outs_sum': f"indirect run-outs in the last {k} matches",
        f'last_{k}_matches_match_type_Innings Batted_sum': f"innings batted in the specific match type during the last {k} matches",
        f'last_{k}_matches_match_type_Innings Bowled_sum': f"innings bowled in the specific match type during the last {k} matches",
        'match_type_total_matches': f"total matches played in the specific match type",
        'batting_fantasy_points': f"batting fantasy points",
        'bowling_fantasy_points': f"bowling fantasy points",
        'fielding_fantasy_points': f"fielding fantasy points",
        f'last_{k}_matches_venue_Wickets_sum': f"wickets taken at the venue in the last {k} matches",
        f'last_{k}_matches_Opposition_Innings Bowled_sum': f"innings bowled against the opposition in the last {k} matches",
        f'last_{k}_matches_Opposition_Innings Batted_sum': f"innings batted against the opposition in the last {k} matches",
        f'last_{k}_matches_match_type_Wickets_sum': f"wickets taken in the specific match type during the last {k} matches",
        f'last_{k}_matches_Opposition_Wickets_sum': f"wickets taken against the opposition in the last {k} matches",
        f'last_{k}_matches_derived_Economy Rate': f"economy rate in the last {k} matches",
        f'last_{k}_matches_Venue_Innings Bowled_sum': f"innings bowled at the venue in the last {k} matches",
        f'last_{k}_matches_lbw_bowled_sum': f"LBWs and bowled dismissals in the last {k} matches",
        f'last_{k}_matches_Bowleds_sum': f"bowled dismissals in the last {k} matches",
        f'league_avg_runs_sum': f"average runs scored in same league",
        f'last_year_avg_Runs': f"average runs scored in the last year",
        f'last_{k}_matches_match_type_Runs_sum': f"runs scored in the last {k} matches of the specific league",
        f'league_avg_wickets_sum': f"average wickets taken in same league",
        f'match_type_total_matches_sum': f"total matches played in the specific match type",
        f'last_{k}_matches_Runs_sum': f"runs scored in the last {k} matches",
        f'last_year_avg_Wickets': f"average wickets taken in the last year",
    }
    
    explanations = []

    for i, result in enumerate(results):
        player_name = players_id_list[i]
        
        top_features = list(result.keys())[:3] 
        feature1, feature2, feature3 = [f for f in top_features]

        if(feature1 in feature_descriptions and feature2 in feature_descriptions and feature3 in feature_descriptions):
            feature1 = feature_descriptions[feature1]
            feature2 = feature_descriptions[feature2]
            feature3 = feature_descriptions[feature3]
            if i < len(results)/2:
                positivity = 1
            else:
                positivity = 0
            
            explanation_text = generate_explainability_text(
                player_name, 
                feature1, 
                feature2, 
                feature3,
                positivity
            )
            explanations.append(explanation_text)
        else:
            print(f"Feature not found in feature_descriptions! {feature1} {feature2} {feature3}")
    return explanations


def generate_explanations(results, players_id_list):
    """
    Generate explanations for all rows in the result array using a single LLM call.
    """
    system_prompt = """
    You are a cricket fantasy prediction explainability assistant. Your task is to explain the fantasy score prediction based on the most important features. 
    These features may include main terms (individual player statistics) and interaction terms (combinations of player statistics that jointly affect the prediction). 
    For each feature, provide an explanation that is relevant to the cricket match context, and prioritize features based on their importance scores. Note that there are **22** players. 
    Your task includes giving positive explainable texts for the players with tag **Selected**, and negative explainable texts for those with tag **Not Selected**.
    Example of positive and negative explainable texts can be: **Player's high predicted fantasy score is attributed to their feature1, feature2, and feature3.** and **Player's strong performance in feature1, feature2, and feature3 was commendable, but competition was tough, leading to their exclusion.**  
    Just give the result no starting text.
    """
    prompt = format_prompt(results, players_id_list)
    full_prompt = system_prompt + "\n" + prompt
    response = llm_model.generate_content(full_prompt)
    return response
# Function to get the feature names from the indices

def get_feature_names_from_indices(indices, main_features):

    feature_names = []
    for feature in indices:
        if isinstance(feature, tuple):  # Interaction term
            feature_names.append(tuple([main_features[i] for i in feature]))
        else:  # Main term
            feature_names.append(main_features[feature])
    return feature_names

def get_top_features(explainability_scores, columns, k=5):
    topk_values, topk_indices = torch.topk(explainability_scores, k=k, dim=1)


    players_data = []
    for i in range(len(topk_values)):
        topk_columns = [columns[idx] for idx in topk_indices[i].tolist()]
        players_data.append(dict(zip(topk_columns, topk_values[i].tolist())))
    
    return players_data

    # Map indices to features for each row
    # topk_features = [[additive_features[idx] for idx in indices] for indices in topk_indices]

    # # Combine scores and features for each row
    # results = []
    # for row_idx in range(len(topk_values)):
    #     row_results = list(zip(topk_values[row_idx].tolist(), get_feature_names_from_indices(topk_features[row_idx], all_features)))
    #     results.append(row_results)
    
    # return results

def explain_outputs(model, X, columns, players_id_list , method="integrated_gradients"):
    if method=="integrated_gradients" :
        integrated_gradients = IntegratedGradients(model)
        attributions = integrated_gradients.attribute(X, target=0)

    elif method=="deeplift" :
        deeplift = DeepLift(model)
        attributions = deeplift.attribute(X, target=0)

    elif method=="gradient_shap" :
        baseline = torch.zeros_like(X)
        random_baselines = torch.cat([baseline + torch.randn_like(baseline) * 0.01 for _ in range(5)], dim=0)
        gradient_shap = GradientShap(model)
        attributions = gradient_shap.attribute(X,baselines=random_baselines, target=0)

    results = get_top_features(attributions, columns, 5)
    response = generate_explanations(results, players_id_list)
    explainations = response.text
    separator = "\n---\n"
    explaination_list = explainations.split(separator)
    
    return explaination_list

def backup_outputs(model, X, columns, players_id_list, k , method="integrated_gradients"):
    if method=="integrated_gradients" :
        integrated_gradients = IntegratedGradients(model)
        attributions = integrated_gradients.attribute(X, target=0)

    elif method=="deeplift" :
        deeplift = DeepLift(model)
        attributions = deeplift.attribute(X, target=0)

    elif method=="gradient_shap" :
        baseline = torch.zeros_like(X)
        random_baselines = torch.cat([baseline + torch.randn_like(baseline) * 0.01 for _ in range(5)], dim=0)
        gradient_shap = GradientShap(model)
        attributions = gradient_shap.attribute(X,baselines=random_baselines, target=0)
    
    df = pd.DataFrame(attributions, columns=columns)
    columns2remove = ['last_15_matches_match_type_Innings Batted_sum', 'venue_avg_runs_sum', 'last_15_matches_Innings Batted_sum', 'match_type_total_matches_sum', 'league_avg_wickets_sum', 'league_avg_runs_sum', 'venue_avg_wickets_sum', 'Total_matches_played_sum']

    df2 = df.drop(columns=columns2remove)
    new_columns = df2.columns
    results = get_top_features(torch.tensor(df2.values), new_columns, 5)
    explaination_list = backup_explanations(results, players_id_list, k)
    
    return explaination_list