# python mlp_baseline_final.py -f 15_ODI -k 15 -e 20 -dim 128 -batch_size 1024 -lr 0.005

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset,Dataset
import pandas as pd
import os
from model_utils import MLPModel, train_model
import argparse
from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data, process_batting,process_bowling,process_field
from sklearn.metrics import mean_squared_error
import pickle

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class PlayerOpponentDataset(Dataset):
    def __init__(self, df,dim,bat_model_path,bowl_model_path,field_model_path, player_bat_cols,player_bowl_cols,player_field_cols, fantasy_bat_col,fantasy_bowl_col,fantasy_field_col,save_path = "/home/ai21btech11012/FIFS_dream11/src/data/processed/charan_pairs_real_embedd.pt" ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing match data.
            model (callable): Function or model that generates embeddings from input features.
            player_cols (list): Columns used to create embeddings.
            fantasy_col (str): Column name for fantasy points.
        """
        self.df = df
        self.dim = dim

        self.fantasy_bat_col = fantasy_bat_col
        self.fantasy_bowl_col = fantasy_bowl_col
        self.fantasy_field_col = fantasy_field_col
        
        self.player_bat_cols = player_bat_cols
        self.player_bowl_cols = player_bowl_cols
        self.player_field_cols = player_field_cols

        self.num_bat_features = len(self.player_bat_cols)
        self.num_bowl_features = len(self.player_bowl_cols)
        self.num_field_features = len(self.player_field_cols)
        print("num_bat_features : ",self.num_bat_features)
        state_dict = torch.load(bat_model_path)
        print("********** state_dict saved ***********" )
        for k in state_dict.keys():
            print(k)  # Check which layers exist in the saved model

        self.bat_model = MLPModel(layer_sizes=[self.num_bat_features,self.dim, 4]).to(device)
        print("*************** bat_model created ***************")  
        for k in self.bat_model.state_dict().keys():
            print(k)  # Check which layers exist in the model
        
        self.bat_model = MLPModel(layer_sizes=[self.num_bat_features,self.dim, 4])
        self.bat_model.load_state_dict(torch.load(bat_model_path))
        self.bat_model.to(device)
        self.bat_model.eval()
       
        self.bowl_model = MLPModel(layer_sizes=[self.num_bowl_features,self.dim, 3])
        self.bowl_model.load_state_dict(torch.load(bowl_model_path))
        self.bowl_model.to(device)
        self.bowl_model.eval()

        self.field_model = MLPModel(layer_sizes=[self.num_field_features,self.dim, 4])
        self.field_model.load_state_dict(torch.load(field_model_path))
        self.field_model.to(device)
        self.field_model.eval()

        # Normalize the data 
        self.df = self.normalize_df(self.df)

        self.save_path = save_path

        # Check if dataset exists; if so, load it
        if os.path.exists(save_path):
            print(f"🔄 Loading dataset from {save_path}...")
            self.pairs = torch.load(save_path)
        else:
            print(f"🚀 Creating new dataset and saving to {save_path}...")
            self.pairs = self.create_pairs()
            torch.save(self.pairs, save_path)
    def normalize_df(self,df):
        for col in df.columns:
            if col not in ["player","team","opposition","date","venue","match_type","match_id","player_id"]:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df
    def get_embedding(self,x,kind = "bat"):
        """Generate an embedding from the selected features using the model."""
        activations = None

        with torch.no_grad():
            if kind == "bat":
               model = self.bat_model()
            elif kind == "bowl":
               model = self.bowl_model()
            elif kind == "field":
               model = self.field_model()
           
            def hook_fn(module, input, output):
                nonlocal activations
                activations = input[0].detach()  # Extract input to the last layer

            # Find index of the last Linear layer in the model
            linear_layers = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
            second_last_linear_index = linear_layers[-2]  # Second last Linear layer

            # Register hook on the second last Linear layer
            handle = model.model[second_last_linear_index].register_forward_hook(hook_fn)

            # Perform a forward pass
            _ = model(x)

            # Remove hook
            handle.remove()

        return activations


    def create_pairs(self):
        """Create all player-opponent embedding pairs and fantasy point labels."""
        pairs = []
        for match_id in self.df['match_id'].unique():
            match_players = self.df[self.df['match_id'] == match_id]

            for _,player in match_players.iterrows():

                # Get player's embedding
                player_features = [player[col] for col in self.player_bat_cols]

                player_embedding = self.get_embedding(player_features)
                # Testing
                # player_embedding = torch.randn((128,))

                # Get player's fantasy points (depends on player type)
                player_type = "batsman"
                if player_type == 'batsman':
                    player_fantasy_points = player[self.fantasy_bat_col]
                elif player_type == 'bowler':
                    player_fantasy_points = player[self.fantasy_bowl_col]

                
                # Get opponent embeddings and fantasy points (depends on player type)
                opponent_features = []
                opponent_fantasy_points = []

                for _, opponent in match_players.iterrows():
                    if opponent["team"] != player["team"]:
                        opponent_features.append(list(opponent[self.player_bowl_cols].values))
                        # Testing
                        opponent_fantasy_points.append(opponent[self.fantasy_bowl_col])

                # Compute average opponent embedding
                if opponent_features:

                    opponent_embeddings = torch.stack([self.get_embedding(feat) for feat in opponent_features])
                    # Testing 
                    # opponent_embeddings = torch.randn((len(opponent_features), 128))

                    opponent_embedding = torch.mean(opponent_embeddings, dim=0)
                    avg_opponent_fantasy_points = sum(opponent_fantasy_points) / len(opponent_fantasy_points)
                else:
                    opponent_embedding = torch.zeros_like(player_embedding)
                    avg_opponent_fantasy_points = 0.0

                # Compute label: player's fantasy points - avg opponent fantasy points
                label = player_fantasy_points - avg_opponent_fantasy_points

                    # **Concatenate player and opponent embeddings**
                combined_embedding = torch.cat((player_embedding, opponent_embedding))  # Shape: (256,)

                # **Store normal pair**
                pairs.append((combined_embedding, torch.tensor(label, dtype=torch.float32)))

                # **Store reversed pair with negated label**
                reversed_embedding = torch.cat((opponent_embedding, player_embedding))  # Swap order
                pairs.append((reversed_embedding, torch.tensor(-1 * label, dtype=torch.float32)))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# function for training the model
def MLP_train(args):
    data_file_name = args.f
    batch_size = args.batch_size
    k = args.k
    dim = args.dim

    scalers_dict = {}

    # print(f" -------------------------------- {data_file_name} ---------------------------------------")
    # train_data_path = os.path.join("..", "data", "processed", "train", f"{data_file_name}.csv")
    # test_data_path = os.path.join("..", "data", "processed", "test", f"{data_file_name}.csv")

    # test_df = pd.read_csv(test_data_path)
    # train_df = pd.read_csv(train_data_path)


    # New Code
    # csv_file
    csv_path = "/home/ai21btech11012/FIFS_dream11/src/data/processed/train/15_himanshu.csv"
    data = pd.read_csv(csv_path) 
    # For each match id, we will have 22 datapoints (batsmen vs opponent bowlers) + (bowlers vs opponent batsmen)
 
    fantasy_bat_col = "batting_fantasy_points"
    fantasy_bowl_col = "bowling_fantasy_points"
    fantasy_field_col = "fielding_fantasy_points"

    player_bat_cols = [ 
            'Total_matches_played_sum', 'last_15_matches_Innings Batted_sum',
       'last_15_matches_Runs_sum', 'last_15_matches_Fours_sum',
       'last_15_matches_Sixes_sum', 'last_15_matches_Outs_sum',
       'last_15_matches_Dot Balls_sum', 'last_15_matches_Balls Faced_sum',
       'last_15_matches_centuries_sum', 'last_15_matches_half_centuries_sum',
       'last_15_matches_duck_outs_sum', 'last_year_avg_Runs',
       'last_year_avg_Wickets', 'last_15_matches_derived_Batting Strike Rate',
       'last_15_matches_derived_Batting Avg',
       'last_15_matches_derived_Mean Score',
       'last_15_matches_derived_Boundary%',
       'last_15_matches_derived_Mean Balls Faced',
       'last_15_matches_derived_Dismissal Rate', 'Venue_total_matches_sum',
       'last_15_matches_Venue_Runs_sum',
       'last_15_matches_Venue_Innings Batted_sum',
       'Opposition_total_matches_sum', 'last_15_matches_Opposition_Runs_sum',
       'last_15_matches_Opposition_Innings Batted_sum',
       'match_type_total_matches_sum', 'last_15_matches_match_type_Runs_sum',
       'last_15_matches_match_type_Innings Batted_sum', 'venue_avg_runs_sum',
       'venue_avg_wickets_sum', 'league_avg_runs_sum',
       'league_avg_wickets_sum'
    ] 
    player_bowl_cols = [
        'Total_matches_played_sum', 'last_15_matches_Innings Batted_sum',
       'last_15_matches_Runs_sum', 'last_15_matches_Innings Bowled_sum',
       'last_15_matches_Balls Bowled_sum', 'last_15_matches_Extras_sum',
       'last_15_matches_Maiden Overs_sum', 'last_15_matches_Runsgiven_sum',
       'last_15_matches_Dot Balls Bowled_sum',
       'last_15_matches_Foursgiven_sum', 'last_15_matches_Sixesgiven_sum',
       'last_year_avg_Runs', 'last_year_avg_Wickets',
       'last_15_matches_derived_Dot Ball%',
       'last_15_matches_derived_Economy Rate',
       'last_15_matches_derived_Bowling Dot Ball%',
       'last_15_matches_derived_Boundary Given%',
       'last_15_matches_derived_Bowling Avg',
       'last_15_matches_derived_Bowling Strike Rate',
       'Venue_total_matches_sum', 'last_15_matches_Venue_Runs_sum',
       'last_15_matches_Venue_Innings Batted_sum',
       'last_15_matches_Venue_Innings Bowled_sum',
       'Opposition_total_matches_sum', 'last_15_matches_Opposition_Runs_sum',
       'last_15_matches_Opposition_Innings Batted_sum',
       'last_15_matches_Opposition_Innings Bowled_sum',
       'match_type_total_matches_sum', 'last_15_matches_match_type_Runs_sum',
       'last_15_matches_match_type_Innings Bowled_sum', 'venue_avg_runs_sum',
       'venue_avg_wickets_sum', 'league_avg_runs_sum',
       'league_avg_wickets_sum'
    ]
    player_field_cols = [
           'Total_matches_played_sum', 'last_15_matches_Innings Batted_sum',
       'last_15_matches_Runs_sum', 'last_15_matches_Stumpings_sum',
       'last_15_matches_Catches_sum', 'last_15_matches_direct run_outs_sum',
       'last_15_matches_indirect run_outs_sum', 'last_year_avg_Runs',
       'last_year_avg_Wickets', 'Venue_total_matches_sum',
       'last_15_matches_Venue_Runs_sum', 'last_15_matches_Venue_Wickets_sum',
       'last_15_matches_Venue_Innings Batted_sum',
       'Opposition_total_matches_sum', 'last_15_matches_Opposition_Runs_sum',
       'last_15_matches_Opposition_Wickets_sum',
       'last_15_matches_Opposition_Innings Batted_sum',
       'match_type_total_matches_sum', 'last_15_matches_match_type_Runs_sum',
       'last_15_matches_match_type_Wickets_sum', 'venue_avg_runs_sum',
       'venue_avg_wickets_sum', 'league_avg_runs_sum',
       'league_avg_wickets_sum'
    ]

    bat_model_path = "/home/ai21btech11012/FIFS_dream11/src/model_artifacts/15_ODI_bat_model.pth"
    bowl_model_path = "/home/ai21btech11012/FIFS_dream11/src/model_artifacts/15_ODI_bowl_model.pth"
    field_model_path = "/home/ai21btech11012/FIFS_dream11/src/model_artifacts/15_ODI_field_model.pth"

    Dataset = PlayerOpponentDataset(data,dim,bat_model_path,bowl_model_path,field_model_path,player_bat_cols,player_bowl_cols,player_field_cols,fantasy_bat_col,fantasy_bowl_col,fantasy_field_col)

    print("Dataset creation Done")

    train_size = int(0.8 * len(Dataset))
    test_size = len(Dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(Dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_input_size = train_loader.dataset[0][0].size(0)
    print("model_input_size : ",model_input_size)

    # Model : model_input.size() , hidden_1, hidden_2, fantasy_points 
    full_model = MLPModel(layer_sizes=[model_input_size, 128, 64, 1]).to(device)


    train_model(full_model, train_loader, test_loader, args, should_save_best_model=False, device=device)    

    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # test_model(full_model, test_loader, device=device)
    # model = train_model(full_model, train_loader, test_loader, args, should_save_best_model=False, device=device)
    # torch.save(model.state_dict(), f"../model_artifacts/{data_file_name}_model.pth")

    # test_predictions = test_model(model, test_loader, device=device)
    # true_values = y_test

    # test_predictions_scaled = test_predictions.detach().numpy()

    # true_values_original = test_df["fantasy_points"].values
    # test_predictions_original = scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten().astype(int)
    # # test_predictions_original = test_predictions_scaled
    # mse_loss = mean_squared_error(true_values_original, test_predictions_original)

    # print("mse_loss_original_scale : ", mse_loss)

    # results = compute_overlap_true_test(true_values, test_predictions, y_test_id)
    # print(f"Average Overlap: {results}")
    # test_df["predicted_points"] = test_predictions_original
    # MAE, MAPE = compute_loss(test_df[["match_id", "predicted_points", "fantasy_points"]])

    # print(f"\n****\nMAE : {MAE}, MAPE : {MAPE}\n****\n")
    
    # with open(f'../model_artifacts/{data_file_name}_scalers.pkl', 'wb') as file:
    #     pickle.dump(scalers_dict, file)



def main():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")
    parser.add_argument("-f", type=str, required=True, help="File name")
    parser.add_argument("-k", type=int, required=True, help="The value of k")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-dim", type=int, default=128, help="MLP layer")
    parser.add_argument("-batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("-lr", type=float, default=0.005, help="batch size")

    args = parser.parse_args()


    MLP_train(args)

if __name__ == "__main__":
    main()

