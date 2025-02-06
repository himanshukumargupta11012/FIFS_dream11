from torch import nn
from torch import optim
import torch
import copy

# model architecture
class MLPModel(nn.Module):
    def __init__(self, input_features, hidden_units):
        super(MLPModel, self).__init__()
        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        dropout_prob = 0.5
        self.model = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_units, 1),
            nn.LeakyReLU(negative_slope=0.1)
            # self.selu
        )
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, test_loader, args, game_format, should_save_best_model=False, device="cpu", save_dir="../model_artifacts"):
    # Extract hyperparameters from args
    if isinstance(args, dict):
        k = args.get("k", None)
        num_epochs = args.get("e", 25)
        batch_size = args.get("batch_size", 32)
        lr = args.get("lr", 0.005)
    else:
        k = args.k
        num_epochs = args.e
        batch_size = args.batch_size
        lr = args.lr

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model_path = f"{save_dir}/{game_format}_k={k}_lr={lr}_e={num_epochs}_b={batch_size}_bestmodel.pth"

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

# Evaluation function
def evaluate_model(model, test_loader, criterion, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    # Calculate and return average test loss
    avg_test_loss = total_loss / len(test_loader)
    return avg_test_loss

# Testing loop
def test_model(model, test_loader, device="cpu"):
    model.eval()  # Set the model to evaluation model
    criterion = nn.MSELoss()
    total_loss = 0
    predictions_list = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
            predictions_list.extend(predictions.cpu().numpy().flatten())
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

    return torch.tensor(predictions_list)