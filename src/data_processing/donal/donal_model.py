import pickle
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from donal_columns import batting_cols, bowling_cols, fielding_cols

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------ Helper Functions ------------------------
training_cutoff_date = "2010-01-01"

def load_data(file_path):
    """Loads and processes data from CSV file."""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] > pd.Timestamp(training_cutoff_date)]  # Filter from 2010 onwards


def select_important_features(df, target_column, threshold=0.01):
    """Uses RandomForest feature importance to select key features."""
    
    logging.info(f"Selecting important features for {target_column}...")

    categorical_cols = ['player', 'team', 'opposition', 'date', 'venue',  'match_id', 'player_id']

    if target_column == "batting_fantasy_points":
        numerical_cols = [col for col in batting_cols if col not in categorical_cols and col != 'batting_fantasy_points']
        numerical_cols_with_target = numerical_cols + [target_column]
    elif target_column == "bowling_fantasy_points":
        numerical_cols = [col for col in bowling_cols if col not in categorical_cols and col != 'bowling_fantasy_points']
        numerical_cols_with_target = numerical_cols + [target_column]
    elif target_column == "fielding_fantasy_points":
        numerical_cols = [col for col in fielding_cols if col not in categorical_cols and col != 'fielding_fantasy_points']
        numerical_cols_with_target = numerical_cols + [target_column]
    else:
        raise ValueError("Invalid category. Choose from 'batting', 'bowling', or 'fielding'.")
    
    # Prepare X and y
    X = df[numerical_cols_with_target].drop(columns=[target_column])
    y = df[target_column]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected_features = feature_importance[feature_importance > threshold].index.tolist()
    
    logging.info(f"Dropped {len(X.columns) - len(selected_features)} low-importance features.")
    
    return df[selected_features + [target_column]], selected_features

def train_and_save_best_model(models, model_inputs, save_path):
    """Trains multiple models, selects the best one, and saves it."""
    logging.info(f"Training models for {save_path}...")

    best_model, best_score = None, float("inf")
    results = {}

    for name, model in tqdm(models.items(), desc="Training Models", leave=True):
        X_train, y_train = model_inputs[name]  # Get correct data for each model
        
        scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1)
        avg_rmse = -np.mean(scores)
        results[name] = {"Avg RMSE (CV)": avg_rmse}

        if avg_rmse < best_score:
            best_score = avg_rmse
            best_model = model

    # Train best model on the full dataset
    X_train, y_train = model_inputs[best_model.__class__.__name__]  # Retrieve correct data
    best_model.fit(X_train, y_train)

    # Save model
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)

    logging.info(f"Best model ({best_model}) saved to {save_path}")

    return pd.DataFrame(results).T, best_model


# ------------------------ Training Functions ------------------------

def train_model(df, target_column, save_path):
    """Trains a model for a given target column and saves it."""
    df_reduced, selected_features = select_important_features(df, target_column)

    X = df_reduced.drop(columns=[target_column])
    y = df_reduced[target_column]

    # Initialize StandardScaler
    # scaler = StandardScaler()
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        # "Lasso Regression": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel="rbf"),  # Needs scaling
        # "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1)  # Needs scaling
    }

    # Prepare model inputs
    model_inputs = {}
    for name, model in models.items():
        # if name in ["SVR", "XGBRegressor"]:
        #     model_inputs[name] = (scaler.fit_transform(X), y)  # Use scaled data
        # else:
        model_inputs[name] = (X, y)  # Use original data

    return train_and_save_best_model(models, model_inputs, save_path)


# ------------------------ Main Execution ------------------------

if __name__ == "__main__":
    logging.info("Starting training process...")

    # Load data
    df = load_data('../data/processed/combined/7_final.csv')

    # Define target variables
    target_columns = {
        "batting_fantasy_points": "best_batting_model.pkl",
        "bowling_fantasy_points": "best_bowling_model.pkl",
        "fielding_fantasy_points": "best_fielding_model.pkl"
    }

    # Train models
    for target, save_path in target_columns.items():
        try:
            results, best_model = train_model(df, target, save_path)
            print(f"\n{target} Model Results:\n", results)
        except Exception as e:
            logging.error(f"Error training {target}: {e}")

    logging.info("Training process completed.")
