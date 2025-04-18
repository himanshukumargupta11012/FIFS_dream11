import pickle
import pandas as pd

# Function to load a saved model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Load the trained models
batting_model = load_model("donal_saved_models/best_batting_model.pkl")
bowling_model = load_model("donal_saved_models/best_bowling_model.pkl")
fielding_model = load_model("donal_saved_models/best_fielding_model.pkl")


# Prepare future match data (must match training features)
future_match_data = pd.DataFrame([
    [0.5, 30, 2, 1],  # Replace with real data
    [0.7, 28, 1, 0],  # Example second player data
], columns=['feature1', 'feature2', 'feature3', 'feature4'])

# Make predictions
future_match_data["Batting Fantasy Points"] = batting_model.predict(future_match_data)
future_match_data["Bowling Fantasy Points"] = bowling_model.predict(future_match_data)
future_match_data["Fielding Fantasy Points"] = fielding_model.predict(future_match_data)

# Calculate Total Fantasy Points
future_match_data["Total Fantasy Points"] = (
    future_match_data["Batting Fantasy Points"] +
    future_match_data["Bowling Fantasy Points"] +
    future_match_data["Fielding Fantasy Points"]
)

# Print results
print(future_match_data)
