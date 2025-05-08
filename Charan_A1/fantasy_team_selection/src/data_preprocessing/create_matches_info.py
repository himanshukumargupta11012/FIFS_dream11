import os
import json
import pandas as pd
from tqdm import tqdm

def create_matches_info(data_path, output_dir):
    # Path to the folder containing all JSON files
    json_dir = os.path.join(data_path, "all_json")
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    # List to store match metadata
    match_data = []

    # Iterate over all JSON files
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(json_dir, json_file)
        try:
            with open(file_path, "r") as f:
                match = json.load(f)
                # Extract metadata
                match_id = match.get("match_id", json_file.split(".")[0])  # Use filename as fallback
                date = match.get("info", {}).get("dates", [None])[0]
                match_type = match.get("info", {}).get("match_type", "unknown")
                gender = match.get("info", {}).get("gender", "unknown")
                teams = " vs ".join(match.get("info", {}).get("teams", []))

                # Append to match_data
                match_data.append({
                    "match_id": match_id,
                    "date": date,
                    "type": match_type,
                    "format": match_type.upper(),
                    "gender": gender,
                    "teams": teams
                })
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    # Convert to DataFrame
    matches_info_df = pd.DataFrame(match_data)

    # Convert date to datetime and sort
    matches_info_df["date"] = pd.to_datetime(matches_info_df["date"], errors="coerce")
    matches_info_df = matches_info_df.sort_values(by="date")

    # Save to CSV
    output_csv_path = os.path.join(output_dir, "matches_info.csv")
    matches_info_df.to_csv(output_csv_path, index=False)
    print(f"matches_info.csv created at: {output_csv_path}")


# Example usage
data_path = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/raw/cricsheet_t20s/"  # Replace with the path to your data folder
output_dir = "/home/ai21btech11012/FIFS_dream11/Charan_A1/fantasy_team_selection/src/data/raw/cricsheet_t20s"   # Replace with the path to your output folder
os.makedirs(output_dir, exist_ok=True)

create_matches_info(data_path, output_dir)