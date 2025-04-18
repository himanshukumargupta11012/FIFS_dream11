from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data, compute_loss2
import pickle
import os
import numpy as np
import pandas as pd

# Paths for saved models
BATTING_MODEL_PATH = "donal_saved_models/best_batting_model.pkl"
BOWLING_MODEL_PATH = "donal_saved_models/best_bowling_model.pkl"
FIELDING_MODEL_PATH = "donal_saved_models/best_fielding_model.pkl"
SELECTED_BATTING_FEATURES_PATH = "features/selected_batting_features.pkl"
SELECTED_BOWLING_FEATURES_PATH = "features/selected_bowling_features.pkl"
SELECTED_FIELDING_FEATURES_PATH = "features/selected_fielding_features.pkl"

# Function to load a saved model and scaler
def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    model = saved_data["model"]
    scaler = saved_data["scaler"]
    return model, scaler


# Load the batting selected features from training
with open(SELECTED_BATTING_FEATURES_PATH, "rb") as f:
    selected_batting_features = pickle.load(f)

# Load the bowling selected features from training
with open(SELECTED_BOWLING_FEATURES_PATH, "rb") as f:
    selected_bowling_features = pickle.load(f)

# Load the fielding selected features from training
with open(SELECTED_FIELDING_FEATURES_PATH, "rb") as f:
    selected_fielding_features = pickle.load(f)


# Load all models
batting_model, batting_scaler = load_model(BATTING_MODEL_PATH)
bowling_model, bowling_scaler = load_model(BOWLING_MODEL_PATH)
fielding_model, fielding_scaler = load_model(FIELDING_MODEL_PATH)


# Load the batting data, bowling data, fielding data
current_dir = os.path.dirname(os.path.realpath(__file__))
data_file_name = "7_final"
combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
combined_df = pd.read_csv(combined_data_path)

combined_df["date"] = pd.to_datetime(combined_df["date"])
start_date = pd.to_datetime("2010-01-01").strftime("%Y-%m-%d")
split_date = pd.to_datetime("2025-02-18").strftime("%Y-%m-%d")
# split_date = pd.Timestamp.today().strftime("%Y-%m-%d")
end_date = pd.to_datetime("2025-03-05").strftime("%Y-%m-%d")


if split_date >= end_date or split_date >= pd.Timestamp.today().strftime("%Y-%m-%d"):
    test = False
else:
    test = True


print(test)


train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
# train_df = train_df[train_df["Total_matches_played_sum"] > 10]
test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]

# Extract the batting data, bowling data, fielding data from the test_df using the selected_batting_features, selected_bowling_features, selected_fielding_features 
batting_data = test_df[selected_batting_features].values
bowling_data = test_df[selected_bowling_features].values
fielding_data = test_df[selected_fielding_features].values

batting_data = pd.DataFrame(batting_data, columns=selected_batting_features)
bowling_data = pd.DataFrame(bowling_data, columns=selected_bowling_features)
fielding_data = pd.DataFrame(fielding_data, columns=selected_fielding_features)


print(batting_data.shape, bowling_data.shape, fielding_data.shape)




# Apply scaling if scaler is available
if batting_scaler:
    batting_data = batting_scaler.transform(batting_data)

if bowling_scaler:
    bowling_data = bowling_scaler.transform(bowling_data)

if fielding_scaler: 
    fielding_data = fielding_scaler.transform(fielding_data)


# Make predictions using the models
predictions = np.zeros((test_df.shape[0],))

predictions += batting_model.predict(batting_data)
predictions += bowling_model.predict(bowling_data)
predictions += fielding_model.predict(fielding_data)

test_df.loc[:, "predicted_points"] = predictions

mae, mape = compute_loss2(test_df)

print(mae)




