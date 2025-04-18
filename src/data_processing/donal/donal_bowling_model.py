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
from donal_columns import batting_cols
from sklearn.preprocessing import StandardScaler

# Configuration settings
DATA_PATH = "../data/processed/combined/7_final.csv"
CUTOFF_DATE = pd.Timestamp("2025-02-18")
IMPORTANCE_THRESHOLD = 0.01  # Feature selection threshold
MODEL_SAVE_PATH = "donal_saved_models/best_batting_model.pkl"
TRAIN_START_DATE = pd.Timestamp("2010-01-01")
FEATURES_SAVE_PATH = "features/selected_batting_features.pkl"
USE_CROSS_VALIDATION = True  # Set to False to train on full train_data without CV
EVALUATE_ON_TEST = True


# Ensure the features directory exists
os.makedirs(os.path.dirname(FEATURES_SAVE_PATH), exist_ok=True)


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
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected_features = feature_importance[feature_importance > IMPORTANCE_THRESHOLD].index.tolist()
    print(f"Selected {len(selected_features)} important features.")
    return selected_features

# Train and evaluate models (with optional Cross-Validation)
def train_and_evaluate(models, X_train, y_train, X_test=None, y_test=None, use_cv=True, scale_features=False):
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_test is not None:
            X_test = scaler.transform(X_test)
    else:
        scaler = None

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
            best_model_name = name

    results_df = pd.DataFrame(results).T
    print("\nModel Performance Summary:")
    print(results_df.to_string())

    # Retrain the best model on full training data
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        y_test_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        print(f"\nTest RMSE for {best_model_name}: {test_rmse:.4f}")
    
    return results_df, best_model, scaler


# Save the best model and scaler
def save_model(model, scaler, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"\nBest model saved to {file_path}")



def main(evaluate_on_test=EVALUATE_ON_TEST, scale_features=False):
    df = load_data(DATA_PATH)
    df = df[df["date"] > TRAIN_START_DATE]
    
    target_column = "batting_fantasy_points"
    categorical_cols = ['player', 'team', 'opposition', 'date', 'venue', 'match_id', 'player_id']
    numerical_features = [col for col in batting_cols if col not in categorical_cols and col != target_column]

    selected_features = select_important_features(df[numerical_features + [target_column]], target_column)
    
    # Save selected features
    with open(FEATURES_SAVE_PATH, "wb") as f:
        pickle.dump(selected_features, f)
    print(f"\nSelected features saved to {FEATURES_SAVE_PATH}")


    df = df[selected_features + [target_column, "date"]]

    # Training set before cutoff
    train_data = df[df["date"] < CUTOFF_DATE]
    X_train, y_train = train_data[selected_features], train_data[target_column]

    # Test set after cutoff if needed
    X_test, y_test = None, None
    if evaluate_on_test:
        test_data = df[df["date"] >= CUTOFF_DATE]
        X_test, y_test = test_data[selected_features], test_data[target_column]
    
    print("\nTraining Data:", X_train.shape)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf"),
        "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.05)
    }

    results, best_model, scaler = train_and_evaluate(
        models, X_train, y_train, X_test, y_test, use_cv=USE_CROSS_VALIDATION, scale_features=scale_features
    )

    save_model(best_model, scaler, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main(evaluate_on_test=True, scale_features=True)
