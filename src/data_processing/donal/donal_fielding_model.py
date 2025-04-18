import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from donal_columns import fielding_cols
from sklearn.preprocessing import StandardScaler


# Configuration settings
DATA_PATH = "../data/processed/combined/7_final.csv"
CUTOFF_DATE = pd.Timestamp("2025-02-18")
IMPORTANCE_THRESHOLD = 0.01  # Feature selection threshold
MODEL_SAVE_PATH = "donal_saved_models/best_fielding_model.pkl"
TRAIN_START_DATE = pd.Timestamp("2010-01-01")
USE_CROSS_VALIDATION = False # Set to False to train on full train_data without CV

# Load dataset
def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Feature selection using Random Forest
def select_important_features(df, target_column):
    print("Selecting important features using Random Forest...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected_features = feature_importance[feature_importance > IMPORTANCE_THRESHOLD].index.tolist()
    print(f"Selected {len(selected_features)} important features.")
    return selected_features

# Train and evaluate models (with optional Cross-Validation)
def train_and_evaluate(models, X_train, y_train, use_cv=True, scale_features=False):
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    results = {}
    best_model = None
    best_rmse = float("inf")

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()

        if use_cv:
            # Perform 5-Fold Cross Validation
            scores = cross_val_score(model, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5)
            rmse = -scores.mean()  # Convert negative RMSE to positive
        else:
            # Train on the entire train dataset
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)

            mse = mean_squared_error(y_train, y_pred)
            rmse = np.sqrt(mse)
        
        end_time = time.time()
        
        results[name] = {
            "RMSE": rmse,
            "Training Time (s)": round(end_time - start_time, 2)
        }
        
        print(f"{name} completed in {round(end_time - start_time, 2)}s | RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
    
    results_df = pd.DataFrame(results).T
    print("\nModel Performance Summary:")
    print(results_df.to_string())
    return results_df, best_model

# Save the best model
def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nBest model saved to {file_path}")

# Main function
def main():
    df = load_data(DATA_PATH)
    df = df[df["date"] > TRAIN_START_DATE]
    
    # Define fielding-specific columns
    target_column = "fielding_fantasy_points"
    categorical_cols = ['player', 'team', 'opposition', 'date', 'venue',  'match_id', 'player_id']
    numerical_features = [col for col in fielding_cols if col not in categorical_cols and col != target_column]

    # Feature selection
    selected_features = select_important_features(df[numerical_features + [target_column]], target_column)
    df = df[selected_features + [target_column, "date"]]
    
    # Select training data before CUTOFF_DATE
    train_data = df[df["date"] < CUTOFF_DATE]
    X_train, y_train = train_data[selected_features], train_data[target_column]
    
    print("\nTraining Data:", X_train.shape)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf"),
        "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.05)
    }
    
    # Train and evaluate models
    results, best_model = train_and_evaluate(models, X_train, y_train, use_cv=USE_CROSS_VALIDATION)
    
    # Retrain the best model on the full train data (since cross-validation only uses parts)
    best_model.fit(X_train, y_train)

    # Save best model
    save_model(best_model, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
