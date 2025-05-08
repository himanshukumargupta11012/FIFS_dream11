import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.nn as nn
import torch.optim as optim

# Autoencoder with supervised head for player embeddings
class AutoencoderSupervised(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(AutoencoderSupervised, self).__init__()
        # Encoder: compress stats into latent embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder: reconstruct original stats from the embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Supervised head: predict fantasy points from latent embedding
        self.supervised_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(),
            nn.Linear(latent_dim//2, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        fantasy_points = self.supervised_head(z)
        return x_reconstructed, fantasy_points, z

# Example training loop for the model
def train_model(model, dataloader, num_epochs, device, lambda_weight=0.5):
    model.to(device)
    # Define loss functions: reconstruction and supervised (fantasy points prediction)
    recon_loss_fn = nn.MSELoss()
    supervised_loss_fn = nn.MSELoss()  # Assuming fantasy points are continuous values
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            # Expecting each batch as a dictionary with keys 'stats' and 'fantasy_points'
            stats = batch['stats'].to(device)
            fantasy_targets = batch['fantasy_points'].to(device)
            optimizer.zero_grad()

            recon_stats, fantasy_preds, _ = model(stats)
            loss_recon = recon_loss_fn(recon_stats, stats)
            loss_supervised = supervised_loss_fn(fantasy_preds, fantasy_targets)
            # Combine losses with lambda weighting
            loss = lambda_weight * loss_recon + (1 - lambda_weight) * loss_supervised

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * stats.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    # Hyperparameters (update these based on your dataset)
    input_dim = 100    # Dimension of player historical match statistics
    latent_dim = 64    # Dimension of latent embedding
    hidden_dim = 100   # Hidden size for encoder/decoder layers
    num_epochs = 50
    batch_size = 32

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a sample dataset for demonstration.
    # Replace this with your actual data loader.

    class PlayerStatsDataset(Dataset):
        def __init__(self, num_samples=1000):
            # Randomly generated data for demonstration purposes.
            self.stats = torch.tensor(np.random.rand(num_samples, input_dim), dtype=torch.float32)
            # Random fantasy points; adjust as appropriate for your data.
            self.fantasy_points = torch.tensor(np.random.rand(num_samples, 1), dtype=torch.float32)

        def __len__(self):
            return len(self.stats)

        def __getitem__(self, idx):
            return {
                'stats': self.stats[idx],
                'fantasy_points': self.fantasy_points[idx]
            }

    dataset = PlayerStatsDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = AutoencoderSupervised(input_dim, latent_dim, hidden_dim)

    # Train the model
    train_model(model, dataloader, num_epochs, device)