from torch import nn
from torch import optim
import torch
import copy
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

# model architecture
class MLPModel(nn.Module):
    def __init__(self, input_features, hidden_units):
        super(MLPModel, self).__init__()
        self.dropout = nn.Dropout(p=0.5)  # Increased dropout
        self.model = nn.Sequential(
            nn.BatchNorm1d(input_features),  # Added batch norm
            nn.Linear(input_features, hidden_units),
            nn.LeakyReLU(negative_slope=0.1),
            self.dropout,
            nn.BatchNorm1d(hidden_units),
            nn.Linear(hidden_units, hidden_units//2),
            nn.LeakyReLU(negative_slope=0.1),
            self.dropout,
            nn.BatchNorm1d(hidden_units//2),
            nn.Linear(hidden_units//2, 1)
        )
        
    def forward(self, x, mc_dropout=False):
        if mc_dropout:
            self.train()  # Enable dropout during inference
        return self.model(x)

class EnsembleModel(nn.Module):
    def __init__(self, input_features, hidden_units, num_models=3):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([
            MLPModel(input_features, hidden_units) for _ in range(num_models)
        ])
    
    def forward(self, x, mc_dropout=False):
        predictions = [model(x, mc_dropout) for model in self.models]
        return torch.mean(torch.stack(predictions), dim=0)

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.1, label_smoothing=0.1):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, historical_std=None):
        # Label smoothing
        target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Base MSE loss
        mse_loss = self.mse(pred, target)
        
        if historical_std is not None:
            # Add variance-aware term
            confidence_penalty = torch.mean(torch.abs(pred - target) / (historical_std + 1e-6))
            return mse_loss + self.alpha * confidence_penalty
        return mse_loss

class OverlapLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(OverlapLoss, self).__init__()
        self.alpha = alpha  # weight for MSE
        self.beta = beta    # weight for overlap term
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, batch_ids=None):
        # MSE component
        mse_loss = self.mse(pred, target)
        
        # Overlap component
        batch_size = pred.shape[0]
        if batch_ids is not None and batch_size > 1:
            unique_ids = torch.unique(batch_ids)
            overlap_loss = 0
            
            for match_id in unique_ids:
                mask = (batch_ids == match_id)
                match_preds = pred[mask]
                match_targets = target[mask]
                
                # Get indices of top 11 players
                _, pred_top_indices = torch.topk(match_preds.flatten(), k=min(11, len(match_preds)))
                _, true_top_indices = torch.topk(match_targets.flatten(), k=min(11, len(match_targets)))
                
                # Calculate overlap penalty
                pred_set = set(pred_top_indices.cpu().numpy())
                true_set = set(true_top_indices.cpu().numpy())
                overlap = len(pred_set.intersection(true_set))
                overlap_penalty = 1.0 - (overlap / min(11, len(match_preds)))
                overlap_loss += overlap_penalty
            
            overlap_loss = overlap_loss / len(unique_ids)
            return self.alpha * mse_loss + self.beta * overlap_loss
        
        return mse_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training function
def train_model(model, train_loader, test_loader, args, should_save_best_model=False, device="cpu", save_dir="../model_artifacts"):
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

    # Get overlap weight from args
    overlap_weight = args.overlap_weight if hasattr(args, 'overlap_weight') else 0.4
    mse_weight = 1 - overlap_weight

    criterion = OverlapLoss(alpha=mse_weight, beta=overlap_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # Added L2 regularization
    
    # Warmup scheduler
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/num_epochs,
        div_factor=10
    )
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)  # Increased patience
    
    best_loss = float('inf')
    best_model_path = f"{save_dir}/{data_file_name}_k={k}_lr={lr}_e={num_epochs}_b={batch_size}_bestmodel.pth"

    best_model_state_dict = None

    model.to(device)
    model.train()

    curriculum_epochs = num_epochs // 3
    current_noise = 0.0
    max_noise = args.noise_factor

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Gradually increase noise (curriculum learning)
        if epoch < curriculum_epochs:
            current_noise = (epoch / curriculum_epochs) * max_noise

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Add noise to both inputs and labels
            input_noise = torch.normal(mean=0., std=args.noise_factor, size=batch_X.shape).to(device)
            label_noise = torch.normal(mean=0., std=args.noise_factor * 0.1, size=batch_y.shape).to(device)
            
            noisy_input = batch_X + input_noise
            noisy_labels = batch_y + label_noise

            optimizer.zero_grad()
            predictions = model(noisy_input)
            loss = criterion(predictions, noisy_labels)
            
            # Add L1 regularization
            l1_lambda = 0.001
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced clip value
            
            optimizer.step()
            scheduler.step()  # Step per batch
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping check
        early_stopping(test_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            if should_save_best_model:
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with Test Loss: {best_loss:.4f}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model_state_dict)

    return model

# Evaluation function
def evaluate_model(model, test_loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    total_overlap = 0
    num_batches = 0

    with torch.no_grad():
        for batch_X, batch_y, batch_ids in test_loader:  # Assuming batch_ids is now included
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y, batch_ids)
            total_loss += loss.item()
            
            # Calculate overlap for monitoring
            for match_id in torch.unique(batch_ids):
                mask = (batch_ids == match_id)
                match_preds = predictions[mask]
                match_targets = batch_y[mask]
                
                _, pred_top_indices = torch.topk(match_preds.flatten(), k=min(11, len(match_preds)))
                _, true_top_indices = torch.topk(match_targets.flatten(), k=min(11, len(match_targets)))
                
                pred_set = set(pred_top_indices.cpu().numpy())
                true_set = set(true_top_indices.cpu().numpy())
                overlap = len(pred_set.intersection(true_set))
                total_overlap += overlap / min(11, len(match_preds))
                num_batches += 1

    avg_overlap = total_overlap / num_batches
    print(f"Current overlap: {avg_overlap:.4f}")
    return total_loss / len(test_loader)

# Testing loop
def test_model(model, test_loader, device="cpu", num_mc_samples=100, noise_factor=0.1, batch_mc_size=10):
    predictions_list = []
    uncertainties = []
    model.train()
    
    if isinstance(model, EnsembleModel):
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                ensemble_preds = []
                
                # Process MC samples in smaller batches
                for mc_batch in range(0, num_mc_samples, batch_mc_size):
                    mc_batch_size = min(batch_mc_size, num_mc_samples - mc_batch)
                    # Create multiple noisy versions of input at once
                    batch_X_repeated = batch_X.repeat(mc_batch_size, 1)
                    noise = torch.normal(mean=0., std=noise_factor, size=batch_X_repeated.shape).to(device)
                    noisy_inputs = batch_X_repeated + noise
                    
                    # Get predictions for all noisy inputs at once
                    preds = [m(noisy_inputs, mc_dropout=True) for m in model.models]
                    stacked_preds = torch.stack(preds)  # [num_models, mc_batch_size*batch_size, 1]
                    ensemble_preds.append(stacked_preds)
                
                # Combine all MC batches
                all_preds = torch.cat(ensemble_preds, dim=1)
                all_preds = all_preds.view(len(model.models), num_mc_samples, -1, 1)
                
                # Calculate statistics
                mean_pred = torch.mean(all_preds, dim=(0,1))
                std_pred = torch.std(all_preds, dim=(0,1))
                
                predictions_list.extend(mean_pred.cpu().numpy().flatten())
                uncertainties.extend(std_pred.cpu().numpy().flatten())
    else:
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_predictions = []
                
                for _ in range(num_mc_samples):
                    noise = torch.normal(mean=0., std=noise_factor, size=batch_X.shape).to(device)
                    noisy_input = batch_X + noise
                    pred = model(noisy_input, mc_dropout=True)
                    batch_predictions.append(pred)
                
                stacked_preds = torch.stack(batch_predictions)
                mean_pred = torch.mean(stacked_preds, dim=0)
                std_pred = torch.std(stacked_preds, dim=0)
                
                predictions_list.extend(mean_pred.cpu().numpy().flatten())
                uncertainties.extend(std_pred.cpu().numpy().flatten())
    
    predictions = torch.tensor(predictions_list)
    uncertainties = torch.tensor(uncertainties)
    
    print(f"Average prediction uncertainty: {torch.mean(uncertainties):.4f}")
    return predictions, uncertainties