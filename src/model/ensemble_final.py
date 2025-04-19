# python predict_ensemble_v2.py \
#    --match_data_file /path/to/specific_match_data.csv \
#    --mlp_model_path /path/to/output/mlp_k10_final_model.pth \
#    --lgbm_model_path /path/to/output/lgbm_k10_final_model.joblib \
#    --scalers_path /path/to/output/mlp_k10_scalers.pkl \
#    --feature_names_path /path/to/output/lgbm_k10_feature_names.pkl \
#    --output_file /path/to/output/match_predictions_int_sorted.csv \
#    --k 10 \
#    --tie_breaker_feature cumulative_fantasy_points_mean \
#    --tie_breaker_ascending false \
#    --mlp_weight 0.1 \
#    --lgbm_weight 0.9

import os
import argparse
import pickle
import time
import pandas as pd
import torch
import numpy as np
import joblib
import lightgbm as lgb

# Import custom functions and models (ensure these files are accessible)
from model_utils_charan import CustomMLPModel
from feature_utils_charan import process


def load_and_test_model(args, match_df, device):

    # --- Validate Ensemble Weights ---
    if not np.isclose(args.mlp_weight + args.lgbm_weight, 1.0):
        print("Warning: Ensemble weights do not sum to 1.0. Normalizing weights.")
        total_weight = args.mlp_weight + args.lgbm_weight
        if total_weight > 0:
            args.mlp_weight /= total_weight
            args.lgbm_weight /= total_weight
        else: # Handle zero weights case
            args.mlp_weight = 0.5
            args.lgbm_weight = 0.5
            print("Weights were zero, defaulting to 0.5 each.")

    print(f"Using ensemble weights: MLP={args.mlp_weight:.2f}, LGBM={args.lgbm_weight:.2f}")

    

    # --- Data Loading and Processing ---
    # match_df = pd.read_csv(args.match_data_file)


    # Keep track of original identifiers AND the tie-breaker feature
    # identifier_cols = ['match_id', 'player_id', 'player_name', 'team1', 'team2'] # Add other base identifiers if needed
    # cols_to_keep = list(set(identifier_cols + [args.tie_breaker_feature])) # Ensure tie-breaker is included and unique
    # identifiers_df = match_df[cols_to_keep].copy()

    # Apply the SAME feature processing function
    X_match, processed_feature_names = process(match_df, args.k)
    if isinstance(processed_feature_names, pd.Index): processed_feature_names = processed_feature_names.tolist()

    num_input_features = X_match.shape[1]


    # 1. Load Scalers (for MLP)
    with open(args.scalers_path, 'rb') as f: scalers_dict = pickle.load(f)
    scaler_X, scaler_y = scalers_dict.get("x"), scalers_dict.get("y")

    # 2. Load Feature Names (for LGBM)
    with open(args.feature_names_path, 'rb') as f: lgbm_feature_names = pickle.load(f)
    if isinstance(lgbm_feature_names, pd.Index): lgbm_feature_names = lgbm_feature_names.tolist()

    # 3. Load LightGBM Model
    lgbm_model = joblib.load(args.lgbm_model_path)

    # --- Load MLP Model ---
    # Define or load MLP architecture - replace with your actual loading logic if needed
    output_dir = "./mlp_fixed_output"
    learning_rate = 0.0006
    batch_size = 1024    
    hidden_layers_list = [128]
    activation = "gelu"    
    dropout_prob = 0.1975     
    weight_init = "lecun"       
    loss_type = "mse" 
    mlp_model = CustomMLPModel(
        input_size=num_input_features,
        hidden_layers=hidden_layers_list,
        activation=activation,
        dropout_prob=dropout_prob,
        weight_init=weight_init
    ).to(device)

    mlp_model.load_state_dict(torch.load(args.mlp_model_path, map_location=device))
    mlp_model.eval()

    # --- MLP Prediction ---
    try: X_match_scaled = scaler_X.transform(X_match)
    except ValueError as e: raise RuntimeError(f"Error scaling features for MLP: {e}") from e
    X_match_tensor = torch.tensor(X_match_scaled, dtype=torch.float32).to(device)
    with torch.no_grad(): 
        mlp_preds_log_scaled = mlp_model(X_match_tensor).cpu().numpy()
    mlp_preds_log = scaler_y.inverse_transform(mlp_preds_log_scaled).flatten()
    mlp_preds_orig = np.expm1(mlp_preds_log)
    mlp_preds_orig = np.maximum(0, mlp_preds_orig)

    # --- LGBM Prediction ---
    print("\n--- Generating LightGBM Predictions ---")
    # Ensure features are in the correct order if necessary
    X_match_lgbm = pd.DataFrame(X_match, columns=processed_feature_names)[lgbm_feature_names].values
    lgbm_preds_log = lgbm_model.predict(X_match_lgbm) # Predict on unscaled data
    lgbm_preds_orig = np.expm1(lgbm_preds_log) # Assumes log-target training
    lgbm_preds_orig = np.maximum(0, lgbm_preds_orig)
    print("LightGBM predictions generated.")

    # --- Weighted Ensemble Predictions ---
    ensemble_preds = (args.mlp_weight * mlp_preds_orig) + (args.lgbm_weight * lgbm_preds_orig)

    # --- Convert to Integer Predictions ---
    ensemble_preds_int = np.round(ensemble_preds).astype(int)

    print(ensemble_preds_int)
    return ensemble_preds_int

    # --- Sort with Tie-breaking ---
    print(f"Sorting results by integer predictions (desc) then by '{args.tie_breaker_feature}' ({'asc' if args.tie_breaker_ascending else 'desc'})...")
    results_df = results_df.sort_values(
        by=['ensemble_predicted_points_int', args.tie_breaker_feature],
        ascending=[False, args.tie_breaker_ascending] # Sort points descending, tie-breaker as specified
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(args.output_file, index=False)
    print(f"\nPredictions saved to: {args.output_file}")




# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fantasy points using a weighted MLP+LGBM ensemble with tie-breaking.")
    # Input/Output & Model Paths
    parser.add_argument("--match_data_file", type=str, required=True, help="Path to CSV data file for the specific match lineup and context.")
    parser.add_argument("--mlp_model_path", type=str, required=True, help="Path to the trained MLP state dict (.pth).")
    parser.add_argument("--lgbm_model_path", type=str, required=True, help="Path to the trained LightGBM model (.joblib).")
    parser.add_argument("--scalers_path", type=str, required=True, help="Path to the saved MLP scalers (.pkl).")
    parser.add_argument("--feature_names_path", type=str, required=True, help="Path to the saved LGBM feature names (.pkl).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the prediction results CSV.")
    parser.add_argument("--k", type=int, required=True, help="Window size 'k' used during feature processing.")
    # Tie-breaking Configuration
    parser.add_argument("--tie_breaker_feature", type=str, default="cumulative_fantasy_points_mean",
                        help="Column name in '--match_data_file' to use for sorting when integer predictions are tied.")
    parser.add_argument("--tie_breaker_ascending", type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Sort tie-breaker feature ascending (True) or descending (False). E.g., True for credits, False for avg points.")
    # Ensemble Configuration
    parser.add_argument("--mlp_weight", type=float, default=0.5, help="Weight for the MLP model in the ensemble.")
    parser.add_argument("--lgbm_weight", type=float, default=0.5, help="Weight for the LightGBM model in the ensemble.")
    # Optional: Add args for MLP architecture if not inferrable or saved within model
    # parser.add_argument("--mlp_hidden_layers", type=str, default="256,128")
    # parser.add_argument("--mlp_activation", type=str, default="relu")

    args = parser.parse_args()

    load_and_test_model(args)
    