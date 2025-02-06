# Description: This script is used to predict the scores of the players in a match and generate explanations for the same.
import torch
from datetime import datetime
import pandas as pd
from explainability import explain_outputs, backup_outputs
from model_utils import MLPModel
from feature_utils import process
import os
import pickle

# loading the model and test data
test_df = {}
test_df_after_10nov = {}
model = {}
combined_df = {}
k = 15

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for format in ["OD", "T20", "Test"]:
    # loading the data
    test_df[format] = pd.read_csv(f"{current_dir}/../data/processed/test/{k}_{format}.csv")
    test_df[format]['match_id'] = test_df[format]['match_id'].astype(str)
    test_df_after_10nov[format] = pd.read_csv(f"{current_dir}/../data/processed/test_after10nov/{k}_{format}.csv")
    test_df_after_10nov[format]['match_id'] = test_df_after_10nov[format]['match_id'].astype(str)
    combined_df[format] = pd.read_csv(f"{current_dir}/../data/processed/combined/{k}_{format}.csv")
    combined_df[format]['match_id'] = combined_df[format]['match_id'].astype(str)

    # loading the model
    model[format] = MLPModel(35, 128)
    state_dict = torch.load(f"{current_dir}/../model_artifacts/{format}_model.pth", weights_only=True, map_location=device)
    model[format].load_state_dict(state_dict)
    model[format].eval()

with open(f'{current_dir}/../model_artifacts/model_scalers.pkl', 'rb') as file:
    scalers_dict = pickle.load(file)
players_data = pd.read_csv(f"{current_dir}/../data/raw/cricksheet/people.csv")

# function for predicting the scores and generating explanations
def forward(date, format, players_id_list, match_id=None):
    date = datetime.strptime(date, "%Y-%m-%d")
    target_date = datetime.strptime("2024-11-10", "%Y-%m-%d")
    if match_id is not None:
        # Case when the match is before 10th November
        if date <= target_date:
            df = test_df[format]
            filtered_rows = df[df['match_id'] == match_id]
        # case when the match is after 10th November
        else:
            df = test_df_after_10nov[format]
            filtered_rows = df[df['player_id'].isin(players_id_list)]
    # custom match
    else:
        df = combined_df[format]
        filtered_rows = df[df['player_id'].isin(players_id_list)]
        filtered_rows = filtered_rows.sort_values(by=['player_id', 'date'], ascending=[True, False])
        filtered_rows = filtered_rows.drop_duplicates(subset='player_id', keep='first').reset_index(drop=True)
        
    debut_player_points = 10
    debut_explaination = "This player is making his debut in this match, so let's give him/her a chance"
    
    non_debut_ids = list(filtered_rows['player_id'])
    debut_ids = list(set(players_id_list) - set(non_debut_ids))

    test_data, columns = process(filtered_rows, k)


    test_data = torch.tensor(test_data).float()
    test_data = scalers_dict[f"{format}_x"].transform(test_data)
    test_data = torch.from_numpy(test_data).float()
    output = model[format](test_data).detach().numpy()
    output = scalers_dict[f"{format}_y"].inverse_transform(output).squeeze(1).astype(int).tolist()

    output, non_debut_ids = zip(*sorted(zip(output, non_debut_ids), key=lambda x: x[0], reverse=True))
    output = list(output)
    non_debut_ids = list(non_debut_ids)
    non_debut_names = []
    for id in non_debut_ids:
        non_debut_names.append(players_data.loc[players_data['identifier'] == id, 'name'].values[0])
    # explaination_list = explain_outputs(model[format], test_data, columns, non_debut_names)
    backup_explaination = backup_outputs(model[format], test_data, columns, non_debut_names, k, method="integrated_gradients") + [debut_explaination] * len(debut_ids)

    output = output + [debut_player_points] * len(debut_ids)
    # explaination_list += [debut_explaination] * len(debut_ids)

    print(output)
    print(output, explaination_list, non_debut_ids + debut_ids)
    return output, backup_explaination, non_debut_ids + debut_ids


def true_forward(date, format, players_id_list, match_id=None):
    date = datetime.strptime(date, "%Y-%m-%d")
    target_date = datetime.strptime("2024-11-10", "%Y-%m-%d")

    
    if match_id is not None:
        # Case when the match is before 10th November
        if date <= target_date:
            df = test_df[format]
            filtered_rows = df[df['match_id'] == match_id]
        # case when the match is after 10th November
        else:
            df = test_df_after_10nov[format]
            filtered_rows = df[df['player_id'].isin(players_id_list)]
    # custom match
    else:
        df = combined_df[format]
        filtered_rows = df[df['player_id'].isin(players_id_list)]
        filtered_rows = filtered_rows.sort_values(by=['player_id', 'date'], ascending=[True, False])
        filtered_rows = filtered_rows.drop_duplicates(subset='player', keep='first').reset_index(drop=True)
        

    debut_player_points = 10
    debut_explaination = "This player is making his debut in this match, so he is given a default score of 10."
    non_debut_ids = list(filtered_rows['player_id'])
    debut_ids = list(set(players_id_list) - set(non_debut_ids))
    test_data, columns = process(filtered_rows, k)
    test_data = torch.tensor(test_data).float()
    output = model[format](test_data).to(torch.int).squeeze(1).tolist()
    output, non_debut_ids = zip(*sorted(zip(output, non_debut_ids), key=lambda x: x[0], reverse=True))
    output = list(output)
    non_debut_ids = list(non_debut_ids)
    non_debut_names = []
    for id in non_debut_ids:
        non_debut_names.append(players_data.loc[players_data['identifier'] == id, 'name'].values[0])
    explaination_list = explain_outputs(model[format], test_data, columns, non_debut_names)
    # backup_explaination = backup_outputs(model[format], test_data, columns, non_debut_names, k)

    output = output + [debut_player_points] * len(debut_ids)
    explaination_list += [debut_explaination] * len(debut_ids)
    explaination_list = [explaination_list[i].strip('\n') for i in range(len(explaination_list))]
    print(len(output), len(explaination_list), len(non_debut_ids + debut_explaination))
    print(output, explaination_list, non_debut_ids + debut_ids)
    return output, explaination_list, non_debut_ids + debut_ids





format = "OD"
date = "2017-06-07"
ids = [
    "740742ef",
    "0a476045",
    "ba607b88",
    "1c914163",
    "dbe50b21",
    "4a8a2e3b",
    "99d63244",
    "fe93fd9d",
    "2e81a32d",
    "cc1e8c68",
    "462411b3",
    "211de69f",
    "f8fd6e3f",
    "8a75e999",
    "9ab63e7b",
    "64c34cd0",
    "2254ab79",
    "9cb8d7a6",
    "9de62878",
    "e174dadd",
    "2911de16",
    "b3118300"
]

match_id = "1022359"

forward(date, format, ids, match_id)