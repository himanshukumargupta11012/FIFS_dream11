import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from collections import defaultdict

class LSTMEnhancedEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_prob=0.3):
        super(LSTMEnhancedEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM Layer with Dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout_prob)

        # Dense layers for enhanced embedding
        self.fc1 = nn.Linear(hidden_dim + embedding_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, player_embedding):
        # x: Sequential data (batch_size, seq_len, embedding_dim)
        # player_embedding: Generalized player embedding (batch_size, embedding_dim)

        lstm_out, _ = self.lstm(x)
        # Taking the last hidden state as context vector
        context_vector = lstm_out[:, -1, :]

        # Combine context vector with generalized player embedding
        combined = torch.cat((context_vector, player_embedding), dim=1)

        # Pass through dense layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        enhanced_embedding = self.fc2(x)

        return enhanced_embedding

# Monte Carlo Dropout during Inference
class MCDropoutLSTM(nn.Module):
    def __init__(self, base_model, num_samples=10):
        super(MCDropoutLSTM, self).__init__()
        self.base_model = base_model
        self.num_samples = num_samples

    def forward(self, x, player_embedding):
        predictions = torch.stack([self.base_model(x, player_embedding) for _ in range(self.num_samples)])
        return predictions.mean(0), predictions.std(0)

class BattingLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super(BattingLoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        loss = F.mse_loss(predictions, targets)
        penalty = self.penalty_weight * torch.mean(torch.abs(predictions - targets))
        return loss + penalty

class BowlingLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super(BowlingLoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        loss = F.mse_loss(predictions, targets)
        penalty = self.penalty_weight * torch.mean(torch.abs(predictions - targets))
        return loss + penalty

class FieldingLoss(nn.Module):
    def __init__(self, penalty_weight=0.5):
        super(FieldingLoss, self).__init__()
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        loss = F.mse_loss(predictions, targets)
        penalty = self.penalty_weight * torch.mean(torch.abs(predictions - targets))
        return loss + penalty

def normalise_data(X, y, MinMax=True):
    if MinMax:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
    else:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # Reshape y to 2D before scaling

    return X_scaled, y_scaled, scaler_X, scaler_y

def process(df, k, return_tensor=True):
    data_start = df.columns.get_loc('Total_matches_played_sum')

    if df.isnull().values.any() or np.isinf(df.iloc[:, data_start:].values).any():
        print("There are NaN or infinite values in the DataFrame.")
        
    new_df = df.iloc[:, data_start:]
    new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'last_{k}_matches_derived')]
    new_df = new_df.loc[:, ~new_df.columns.str.startswith(f'cumulative_derived_')]
    new_df = new_df.drop(
        [   
            'year',
            'cumulative_Innings Batted_sum',
            'cumulative_Outs_sum',
            'cumulative_Dot Balls_sum',
            f'last_{k}_matches_Outs_sum',
            f'last_{k}_matches_Dot Balls_sum',  
            f'last_{k}_matches_Balls Faced_sum',
            f'last_{k}_matches_Innings Bowled_sum',
            f'last_{k}_matches_Balls Bowled_sum',
            'Opponent_total_matches',
            'Venue_total_matches',
            f'last_{k}_matches_Foursgiven_sum',
            f'last_{k}_matches_Sixesgiven_sum',
            f'last_{k}_matches_Extras_sum',
            f'last_{k}_matches_centuries_sum',
            f'last_{k}_matches_half_centuries_sum',
            f'last_{k}_matches_Wickets_sum', 
            f'last_{k}_matches_LBWs_sum',
            f'last_{k}_matches_derived_Economy Rate',
            f'last_{k}_matches_lbw_bowled_sum',
            f'last_{k}_matches_Bowleds_sum',
            f'last_{k}_matches_duck_outs_sum',
            f'last_{k}_matches_3wickets_sum',
            f'last_{k}_matches_4wickets_sum',
            f'last_{k}_matches_5wickets_sum',
        ],
        axis=1,  # Specify dropping columns
        errors='ignore'  # Avoid errors if columns are missing
    )
    print(new_df.columns)
    if not return_tensor:
        return new_df
        
    X_train_tensor = torch.tensor(new_df.values, dtype=torch.float32)
    return X_train_tensor, new_df.columns

def compute_overlap_true_test(true_tensor, pred_tensor, pred_match_id):
    grouped_data = defaultdict(lambda: {'test': [], 'true': []})

    for key, val2, val3 in zip(pred_match_id, pred_tensor, true_tensor):
        grouped_data[key]['test'].append(val2)
        grouped_data[key]['true'].append(val3)

    results = {}
    for key, value in grouped_data.items():
        top_11_indices_list2 = sorted(range(len(value['test'])), 
                                      key=lambda i: value['test'][i], reverse=True)[:11]
        top_11_indices_list3 = sorted(range(len(value['true'])), 
                                      key=lambda i: value['true'][i], reverse=True)[:11]
        
        matching_indices = set(top_11_indices_list2).intersection(top_11_indices_list3)
        results[key] = len(matching_indices) 

    average_matching_indices = sum(results.values()) / len(results)

    print("average_matching_indices : ", average_matching_indices)
    
    return average_matching_indices

# Function to compute MAE and MAPE loss
def compute_loss(df):
    prediction_list = []
    actual_list = []
    for match_id, match_data in df.groupby('match_id'):
        top_predicted = match_data.sort_values(by='predicted_points', ascending=False).iloc[:11]['predicted_points']
        top_actual = match_data.sort_values(by='fantasy_points', ascending=False).iloc[:11]['fantasy_points']

        top_predicted += 4
        top_predicted.iloc[0] *= 2
        top_predicted.iloc[1] = int(top_predicted.iloc[1] * 1.5)

        top_actual += 4
        top_actual.iloc[0] *= 2
        top_actual.iloc[1] = int(top_actual.iloc[1] * 1.5)

        predicted_sum = top_predicted.sum()
        actual_sum = top_actual.sum()

        prediction_list.append(predicted_sum)
        actual_list.append(actual_sum)

    prediction_list = np.array(prediction_list)
    actual_list = np.array(actual_list)

    MAE = np.mean(np.abs(prediction_list - actual_list))
    MAPE = np.mean(np.abs((prediction_list - actual_list) / actual_list))

    return MAE, MAPE

# Example usage
if __name__ == "__main__":
    embedding_dim = 64
    hidden_dim = 32
    output_dim = 64
    dropout_prob = 0.3
    k = 15

    base_model = LSTMEnhancedEmbedding(embedding_dim, hidden_dim, output_dim, dropout_prob)
    model = MCDropoutLSTM(base_model, num_samples=10)
    
    player_embedding = torch.randn(1, embedding_dim)
    last_k_stats = torch.randn(1, k, embedding_dim)
    
    enhanced_embedding_mean, enhanced_embedding_std = model(last_k_stats, player_embedding)
    
    print("Enhanced Embedding Mean:", enhanced_embedding_mean)
    print("Enhanced Embedding Std:", enhanced_embedding_std)
