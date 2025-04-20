from torch import nn
from torch import optim
import torch
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# class BattingLoss(nn.Module):
#     def __init__(self, base_loss=nn.MSELoss(), penalty_weight=1.0):
#         super(BattingLoss, self).__init__()
#         self.base_loss = base_loss
#         self.penalty_weight = penalty_weight

#     def forward(self, predictions, targets):
#         # Base MSE loss
#         mse_loss = self.base_loss(predictions, targets)
        
#         # Constraint violation penalty
#         runs_pred = predictions[:, 0]
#         fours_pred = predictions[:, 1]
#         sixes_pred = predictions[:, 2]
#         balls_pred = predictions[:, 3]
        
#         constraint_violation = torch.relu(fours_pred * 4 + sixes_pred * 6 - runs_pred)
#         penalty = self.penalty_weight * torch.mean(constraint_violation)
        
#         ball_constraint_violation = torch.relu(sixes_pred + fours_pred - balls_pred)
#         ball_penalty = self.penalty_weight * torch.mean(ball_constraint_violation)
        
#          # Constraint: if is_playing is zero, all other predictions must be zero
#         zero_constraint_violation = torch.relu(-1 * balls_pred * (runs_pred + fours_pred + sixes_pred))
#         zero_penalty = self.penalty_weight * torch.mean(zero_constraint_violation)
        
#         # Constraint: all predictions must be greater than or equal to zero
#         non_negative_violation = torch.relu(-predictions)
#         non_negative_penalty = self.penalty_weight * torch.mean(non_negative_violation)
        
#         # Total loss
#         total_loss = mse_loss*2 + penalty*1 + ball_penalty*0.5 + non_negative_penalty*0.5 +zero_penalty*0.25
#         return total_loss

class BattingLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), penalty_weight=1.0):
        super(BattingLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        mse_loss = self.base_loss(predictions, targets)

        runs_pred = predictions[:, 0]
        fours_pred = predictions[:, 1]
        sixes_pred = predictions[:, 2]
        balls_pred = predictions[:, 3]

        # Constraint 1: Runs consistency
        constraint_violation = torch.relu(fours_pred * 4 + sixes_pred * 6 - runs_pred)
        penalty = self.penalty_weight * torch.mean(constraint_violation)

        # Constraint 2: Balls faced should be at least fours + sixes
        ball_constraint_violation = torch.relu(sixes_pred + fours_pred - balls_pred)
        ball_penalty = self.penalty_weight * torch.mean(ball_constraint_violation)

        # Constraint 3: Extra runs should be within a reasonable range (soft constraint)
        extra_runs = runs_pred - (fours_pred * 4 + sixes_pred * 6)
        soft_constraint_violation = torch.relu(extra_runs - balls_pred * 3)
        soft_penalty = self.penalty_weight * torch.mean(soft_constraint_violation)

        # Constraint 4: If balls == 0, then all other values should also be 0
        zero_constraint_violation = torch.relu(-1 * balls_pred * (runs_pred + fours_pred + sixes_pred))
        zero_penalty = self.penalty_weight * torch.mean(zero_constraint_violation)

        # Constraint 5: All predictions must be non-negative
        non_negative_violation = torch.relu(-predictions)
        non_negative_penalty = self.penalty_weight * torch.mean(non_negative_violation)

        # Constraint 6: Ensure strike rate is within reasonable limits
        strike_rate = (runs_pred / (balls_pred + 1e-6)) * 100  # Avoid division by zero
        low_strike_rate_violation = torch.relu(50 - strike_rate)
        high_strike_rate_violation = torch.relu(strike_rate - 250)
        strike_rate_penalty = self.penalty_weight * torch.mean(low_strike_rate_violation + high_strike_rate_violation)

        # Total loss with adjusted weights
        total_loss = (
            mse_loss * 2
            + penalty * 1
            + ball_penalty * 1.5
            + zero_penalty * 0.5
            + non_negative_penalty * 0.5
            + soft_penalty * 0.75
            + strike_rate_penalty * 0.25
        )

        return total_loss

    
# class BowlingLoss(nn.Module):
#     def __init__(self, base_loss=nn.MSELoss(), penalty_weight=1.0):
#         super(BowlingLoss, self).__init__()
#         self.base_loss = base_loss
#         self.penalty_weight = penalty_weight

#     def forward(self, predictions, targets):
#         # Base MSE loss
#         mse_loss = self.base_loss(predictions, targets)
        
#         # Constraint violation penalty
#         balls_pred = predictions[:, 0]
#         maiden_pred = predictions[:, 1]
#         runs_pred = predictions[:, 2]
        
#         constraint_violation = torch.relu(maiden_pred * 6 - balls_pred)
#         penalty = self.penalty_weight * torch.mean(constraint_violation)
        
        
#          # Constraint: if is_playing is zero, all other predictions must be zero
#         zero_constraint_violation = torch.relu(-1 * balls_pred * (runs_pred))
#         zero_penalty = self.penalty_weight * torch.mean(zero_constraint_violation)
        
#         # Constraint: all predictions must be greater than or equal to zero
#         non_negative_violation = torch.relu(-predictions)
#         non_negative_penalty = self.penalty_weight * torch.mean(non_negative_violation)
        
#         # Total loss
#         total_loss = mse_loss*2 + penalty*1 + non_negative_penalty*1 + zero_penalty*0.25
#         return total_loss


class BowlingLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), penalty_weight=1.0):
        super(BowlingLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        mse_loss = self.base_loss(predictions, targets)

        balls_pred = predictions[:, 0]  # Balls bowled
        maiden_pred = predictions[:, 1]  # Maiden overs
        runs_pred = predictions[:, 3]  # Runs conceded

        # 1. Constraint: Maidens cannot exceed available balls
        maiden_constraint_violation = torch.relu(maiden_pred * 6 - balls_pred)  
        
        # Use torch.where to prevent issues when balls_pred is zero
        valid_maidens = torch.where(balls_pred > 0, torch.floor(balls_pred / 6), torch.zeros_like(balls_pred))
        invalid_maiden_violation = torch.relu(maiden_pred - valid_maidens)

        # 2. Constraint: Runs per ball should be within a reasonable limit (adjusted from 1 to 1.5)
        runs_per_ball_violation = torch.relu(runs_pred - (balls_pred * 1.5))  

        # 3. Zero Balls Constraint: If no balls bowled, runs and maidens must be zero
        zero_balls_mask = (balls_pred == 0).float()
        zero_balls_violation = torch.relu(zero_balls_mask * (runs_pred + maiden_pred))

        # 4. Constraint: All predictions must be non-negative
        non_negative_violation = torch.relu(-predictions)

        # Compute penalties with appropriate weights
        total_loss = (
            mse_loss * 2
            + maiden_constraint_violation.mean() * 1.5
            + invalid_maiden_violation.mean() * 1.0
            + runs_per_ball_violation.mean() * 2.0 
            + zero_balls_violation.mean() * 1.0  # Increased weight for clear constraint
            + non_negative_violation.mean() * 0.25
        )

        return total_loss




class WicketLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), penalty_weight=1.0):
        super(WicketLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        # Base MSE loss
        mse_loss = self.base_loss(predictions, targets)
        
        # Constraint violation penalty
        wickets_pred = predictions[:, 0]
        bonus_pred = predictions[:, 1]
        
        constraint_violation = torch.relu(bonus_pred - wickets_pred)
        penalty = self.penalty_weight * torch.mean(constraint_violation)
        
        
         # Constraint: if is_playing is zero, all other predictions must be zero
        zero_constraint_violation = torch.relu(-1 * wickets_pred * (bonus_pred))
        zero_penalty = self.penalty_weight * torch.mean(zero_constraint_violation)
        
        # Constraint: all predictions must be greater than or equal to zero
        non_negative_violation = torch.relu(-predictions)
        non_negative_penalty = self.penalty_weight * torch.mean(non_negative_violation)
        
        # Total loss
        total_loss = mse_loss*2 + penalty*1 + non_negative_penalty*1 + zero_penalty*0.25
        return total_loss
    
class FieldingLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), penalty_weight=1.0):
        super(FieldingLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty_weight = penalty_weight

    def forward(self, predictions, targets):
        # Base MSE loss
        mse_loss = self.base_loss(predictions, targets)
        
        # Constraint violation penalty
        catch_pred = predictions[:, 0]
        stumping_pred = predictions[:, 1]
        drun_out_pred = predictions[:, 2]
        idrun_out_pred = predictions[:, 3]
        
        
        # Constraint: all predictions must be greater than or equal to zero
        non_negative_violation = torch.relu(-predictions)
        non_negative_penalty = self.penalty_weight * torch.mean(non_negative_violation)
        
        # Total loss
        total_loss = mse_loss + non_negative_penalty *0.5
        return total_loss
    
# model architecture
class MLPModel(nn.Module):
    def __init__(self, layer_sizes):
        super(MLPModel, self).__init__()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        dropout_prob = 0.25
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # if i < len(layer_sizes) - 2:
            layers.append(self.relu)
            layers.append(nn.Dropout(p=dropout_prob))
            # layers.append(self.leaky_relu)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training function for regressor models
def train_regressor_model(model, train_loader, test_loader, args, should_save_best_model=False, device="cpu", save_dir="../model_artifacts", criterion=nn.MSELoss()):
    # Extract hyperparameters from args
    if isinstance(args, dict):
        k = args.get("k", None)
        num_epochs = args.get("e", 25)
        batch_size = args.get("batch_size", 32)
        lr = args.get("lr", 0.005)
        data_file_name = args.get("f", None)
    else:
        k = args.k
        num_epochs = args.e
        batch_size = args.batch_size
        lr = args.lr
        data_file_name = args.f

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model_path = f"{save_dir}/{data_file_name}_k={k}_lr={lr}_e={num_epochs}_b={batch_size}_bestmodel.pth"

    best_model_state_dict = None

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Save the best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            if should_save_best_model:
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with Test Loss: {best_loss:.4f}")

    model.load_state_dict(best_model_state_dict)

    return model


# Training function for classification models
def train_classifier_model(model, train_loader, test_loader, optimizer, criterion, args, should_save_best_model=False, device="cpu", save_dir="../model_artifacts"):
    # Extract hyperparameters from args
    if isinstance(args, dict):
        k = args.get("k", None)
        num_epochs = args.get("e", 25)
        batch_size = args.get("batch_size", 32)
        lr = args.get("lr", 0.005)
        data_file_name = args.get("f", None)
    else:
        k = args.k
        num_epochs = args.e
        batch_size = args.batch_size
        lr = args.lr
        data_file_name = args.f

    
    best_accuracy = float(0)
    best_model_path = f"{save_dir}/{data_file_name}_k={k}_lr={lr}_e={num_epochs}_b={batch_size}_bestmodel.pth"

    best_model_state_dict = None

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate_model(model, test_loader, criterion, device)
        test_accuracy = evaluate_classifier_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state_dict = copy.deepcopy(model.state_dict())
            if should_save_best_model:
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with Test Loss: {best_accuracy:.4f}")

    model.load_state_dict(best_model_state_dict)

    return model

# Evaluation function
def evaluate_model(model, test_loader, criterion=nn.MSELoss(), device="cpu", return_predictions=False):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    if return_predictions:
        predictions_list = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            if return_predictions:
                predictions_list.extend(list(predictions.cpu().numpy()))

    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss if not return_predictions else (avg_test_loss, torch.tensor(predictions_list).flatten())


# Evaluation function
def evaluate_classifier_model(model, test_loader, device="cpu", print_results=False):
    model.eval()  # Set the model to evaluation mode

    pred_labels = []
    true_labels = []
    all_overlaps = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            topk_indices = torch.topk(predictions, k=11, dim=1).indices  # shape: (batch, 11)

            # Create binary predictions: 1 for selected players, 0 otherwise
            preds = torch.zeros_like(predictions)
            preds.scatter_(1, topk_indices, 1)

            # preds = torch.round(predictions)  # shape: (batch, 22)
            # preds = torch.where(predictions > 0.5, 1, 0)  # shape: (batch, 22)

            pred_labels.extend(list(preds.cpu().numpy().flatten()))
            true_labels.extend(list(batch_y.cpu().numpy().flatten()))

            overlap = torch.sum(preds * batch_y, dim=1)
            all_overlaps.extend(list(overlap.cpu().numpy()))

    # all_overlaps = torch.mean(preds * batch_y, dim=1)
    confusion_matri = confusion_matrix(true_labels, pred_labels)
    pred_labels = torch.tensor(pred_labels)
    true_labels = torch.tensor(true_labels)
    accuracy = (pred_labels == true_labels).sum().item() / len(pred_labels)
    avg_overlap = sum(all_overlaps) / len(all_overlaps)
    if print_results:
        print(f"Confusion Matrix:\n{confusion_matri}")
        print(f"Average Overlap: {avg_overlap}")

    return accuracy
    # return avg_test_loss if not return_predictions else (avg_test_loss, torch.tensor(predictions_list).flatten())




class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=50, high_weight=2.0, low_weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight

    def forward(self, y_pred, y_true):
        # weights = torch.where(y_true > self.threshold, self.high_weight, self.low_weight)
        weights = 2.4 + y_true
        mse = (y_pred - y_true) ** 2
        weighted_loss = weights * mse
        return weighted_loss.mean()


class PlayerSelectorTransformer(nn.Module):
    def __init__(self, embed_dim=128, transformer_dim=128, num_heads=4, num_layers=2):
        super(PlayerSelectorTransformer, self).__init__()   
        self.is_mlp = embed_dim != transformer_dim
        if self.is_mlp:
            # Optional linear projection if your player embeddings are not already of embed_dim
            self.input_proj = nn.Linear(embed_dim, transformer_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=num_heads, 
            batch_first=True  # important: allows input as (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head to output selection probability for each player
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, 22, embed_dim)
        if self.is_mlp:
            x = self.input_proj(x)  # shape: (batch_size, 22, transformer_dim)

        # Transformer encoder processes all player embeddings with self-attention
        x = self.transformer_encoder(x)  # shape: (batch_size, 22, transformer_dim)

        # Apply classifier to each player embedding to get selection probability
        probs = self.classifier(x).squeeze(-1)  # shape: (batch_size, 22)
        return probs


class MLPClassifier(nn.Module):
    def __init__(self, layer_sizes):
        super(MLPClassifier, self).__init__()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.relu = nn.ReLU()
        dropout_prob = 0.25
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(self.relu)
                layers.append(nn.Dropout(p=dropout_prob))
            else:
                layers.append(nn.Sigmoid())
            # layers.append(self.leaky_relu)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)