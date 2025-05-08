import torch
import torch.nn as nn
import torch.optim as optim

# Venue Embedding Model: Predict avg fantasy points from venue features and learn an embedding.
class VenueEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(VenueEmbedding, self).__init__()
        # Encoder for venue features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        # Supervised head: predict average fantasy points
        self.supervised_head = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x: venue features
        embedding = self.encoder(x)
        avg_fantasy_points = self.supervised_head(embedding)
        return embedding, avg_fantasy_points

if __name__ == '__main__':
    # ...existing code...
    input_dim = 20       # Example: number of venue-specific features
    embedding_dim = 16   # Embedding dimension for venue
    hidden_dim = 32
    num_epochs = 30
    batch_size = 16

    # Dummy dataset for demonstration
    dummy_venues = torch.randn(100, input_dim)
    dummy_targets = torch.randn(100, 1)

    model = VenueEmbedding(input_dim, embedding_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        _, preds = model(dummy_venues)
        loss = loss_fn(preds, dummy_targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
