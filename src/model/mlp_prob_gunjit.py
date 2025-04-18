import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

# Probabilistic MLP model: outputs mean and log variance for each target
class MLPProbModel(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=0.25):
        """
        layer_sizes: list of integers.
            For example, [input_dim, hidden_dim, output_dim]
            Note: final layer will output 2*output_dim (mean and log variance)
        """
        super(MLPProbModel, self).__init__()
        self.leaky_relu = nn.ReLU()
        layers = []
        # Build hidden layers (all hidden layers use standard size)
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(self.leaky_relu)
            layers.append(nn.Dropout(p=dropout_prob))
        # Final layer: outputs 2 * output_dim for mean and log variance
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1] * 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        # Split the outputs into mean and log variance parts
        mean, log_var = torch.chunk(out, 2, dim=1)
        return mean, log_var

# Negative Log Likelihood Loss for probabilistic regression
class ProbabilisticLoss(nn.Module):
    def __init__(self):
        super(ProbabilisticLoss, self).__init__()

    def forward(self, predictions, targets):
        mean, log_var = predictions
        # Ensure variance is positive
        var = torch.exp(log_var) + 1e-6
        # Compute per-sample negative log likelihood of Gaussian distribution
        nll = 0.5 * torch.log(2 * 3.1415926 * var) + 0.5 * ((targets - mean) ** 2) / var
        return torch.mean(nll)

# Training function for probabilistic model
def train_model_prob(model, train_loader, test_loader, args, 
                     should_save_best_model=False, device="cpu", 
                     save_dir="../model_artifacts", loss_fn=ProbabilisticLoss()):
    # Extract hyperparameters from args
    if isinstance(args, dict):
        k = args.get("k", None)
        num_epochs = args.get("e", 25)
        batch_size = args.get("batch_size", 32)
        lr = args.get("lr", 0.005)
        data_file_name = args.get("f", "default")
    else:
        k = args.k
        num_epochs = args.e
        batch_size = args.batch_size
        lr = args.lr
        data_file_name = args.f

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_model_path = f"{save_dir}/{data_file_name}_prob_k={k}_lr={lr}_e={num_epochs}_b={batch_size}_bestmodel.pth"
    best_model_state_dict = None

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        test_loss = evaluate_model_prob(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            if should_save_best_model:
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with Test Loss: {best_loss:.4f}")

    model.load_state_dict(best_model_state_dict)
    return model

# Evaluation function for probabilistic model
def evaluate_model_prob(model, test_loader, loss_fn=ProbabilisticLoss(), device="cpu", return_predictions=False):
    model.eval()
    total_loss = 0
    predictions_list = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            loss = loss_fn(preds, batch_y)
            total_loss += loss.item()
            if return_predictions:
                mean, _ = preds
                predictions_list.extend(mean.cpu().numpy())
    avg_test_loss = total_loss / len(test_loader)
    if return_predictions:
        return avg_test_loss, torch.tensor(predictions_list)
    else:
        return avg_test_loss

# Example usage (to be run in your training script/notebook)
if __name__ == "__main__":
    # Dummy example values:
    input_dim = 10
    hidden_dim = 64
    output_dim = 4  # number of targets
    layer_sizes = [input_dim, hidden_dim, output_dim]
    # For probabilistic model, final layer outputs 2*output_dim
    model = MLPProbModel(layer_sizes).to("cpu")

    # Here, use your own DataLoader instances for train_loader and test_loader
    # For example:
    # train_loader = ...
    # test_loader = ...
    # args = {"e": 25, "batch_size": 32, "lr": 0.005, "f": "sample"}

    # model = train_model_prob(model, train_loader, test_loader, args, device="cpu")
    # test_loss, predictions = evaluate_model_prob(model, test_loader, device="cpu", return_predictions=True)
    pass