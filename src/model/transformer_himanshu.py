# python transformer_himanshu.py -f 7_final -e 20 -dim 128 -batch_size 1024 -lr 0.005 -model_name test

from sklearn.discriminant_analysis import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from model_utils import MLPModel, train_classifier_model, WeightedMSELoss, PlayerSelectorTransformer, evaluate_classifier_model, MLPClassifier
import argparse
from feature_utils import process, compute_overlap_true_test, compute_loss, normalise_data, classification_process
import pickle
from torch import optim

current_dir = os.path.dirname(os.path.abspath(__file__))

def set_seeds(seed=0):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at the start
set_seeds()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# function for training the model
def MLP_train(args):
    data_file_name = args.f
    batch_size = args.batch_size
    k = args.k
    dim = args.dim

    scalers_dict = {}

    print(f" -------------------------------- {data_file_name} ---------------------------------------")

    combined_data_path = os.path.join(current_dir, "..", "data", "processed", "combined", f"{data_file_name}.csv")
 
    combined_df = pd.read_csv(combined_data_path)
    # print(combined_df.columns)
    # combined_df = combined_df[combined_df["Total_matches_played_sum"] > 10]
    # # print(f"after shape : {combined_df.shape}")

    combined_df["date"] = pd.to_datetime(combined_df["date"])

    # ------------- Data split -------------
    start_date = pd.to_datetime("2010-01-01")
    split_date = pd.to_datetime("2025-10-19")
    # split_date = pd.to_datetime(pd.Timestamp.today().strftime("%Y-%m-%d"))
    end_date = pd.to_datetime("2025-10-05")


    train_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= split_date)]
    test_df = combined_df[(combined_df["date"] > split_date) & (combined_df["date"] <= end_date)]
    print(train_df.shape, test_df.shape)
    # --------------------------------------

    is_mlp = False
    X_train, y_train, _ = classification_process(train_df, k, is_mlp=is_mlp)
    X_test, y_test, _ = classification_process(test_df, k, is_mlp=is_mlp)

    scaler_X = StandardScaler()
    if len(X_train.shape) == 3:
        X_train_2d = X_train.reshape(-1, X_train.shape[-1])
        X_train_2d = scaler_X.fit_transform(X_train_2d)
        X_train = X_train_2d.reshape(X_train.shape)
        X_train = torch.from_numpy(X_train).to(torch.float32)

    else:
        X_train = scaler_X.fit_transform(X_train)
        X_train = torch.from_numpy(X_train).to(torch.float32)

    
    save_path = f'{current_dir}/../model_artifacts/{args.model_name}_d-{data_file_name}_sd-{start_date.strftime("%Y-%m-%d")}_ed-{split_date.strftime("%Y-%m-%d")}'

    scalers_dict[f"x"] = scaler_X
    with open(f'{save_path}_scalers.pkl', 'wb') as file:
        pickle.dump(scalers_dict, file)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if test_df.shape[0] > 0:
        if len(X_test.shape) == 3:
            X_test_2d = X_test.reshape(-1, X_test.shape[-1])
            X_test_2d = scaler_X.transform(X_test_2d)
            X_test = X_test_2d.reshape(X_test.shape)
            X_test = torch.from_numpy(X_test).to(torch.float32)
        else:
            X_test = scaler_X.transform(X_test)
            X_test = torch.from_numpy(X_test).to(torch.float32)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        test_loader = train_loader
        test_df = train_df

    num_input_features = X_train.shape[-1]
    print(num_input_features)

    # Define and train the model
    if is_mlp:
        full_model = MLPClassifier(layer_sizes=[num_input_features, 768, dim, 22]).to(device)
    else:
        full_model = PlayerSelectorTransformer(embed_dim=num_input_features, transformer_dim=num_input_features, num_heads=1, num_layers=2).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(full_model.parameters(), lr=args.lr, weight_decay=1e-1)
    model = train_classifier_model(full_model, train_loader, test_loader, optimizer, criterion, args, should_save_best_model=False, device=device)
    
    # Save the model
    torch.save(model.state_dict(), f"{save_path}_model.pth")

    # Evaluate the model
    evaluate_classifier_model(model, train_loader, device=device, print_results=True)
    evaluate_classifier_model(model, test_loader, device=device, print_results=True)
    

    # test_predictions = test_predictions.numpy()

    # test_df.loc[:, "predicted_points"] = test_predictions



def main():
    parser = argparse.ArgumentParser(description="Parse hyperparameters")
    parser.add_argument("-f", type=str, required=True, help="File name")
    parser.add_argument("-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("-dim", type=int, default=128, help="MLP layer")
    parser.add_argument("-batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("-lr", type=float, default=0.005, help="batch size")
    parser.add_argument("-model_name", type=str, default="MLP", help="Model name")

    args = parser.parse_args()

    k = args.f.split("_")[0]
    args.k = int(k)

    MLP_train(args)

if __name__ == "__main__":
    main()

