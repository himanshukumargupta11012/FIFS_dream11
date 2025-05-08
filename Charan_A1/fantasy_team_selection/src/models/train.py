import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import embedding and model modules (assumed available)
from src.embeddings.player_embedding import AutoencoderSupervised
from src.embeddings.opponent_embedding import OpponentEmbedding
from src.embeddings.venue_embedding import VenueEmbedding
from src.models.fantasy_point_model import FantasyPointPredictor

# Dummy Dataset for demonstration
class FantasyDataset(Dataset):
    def __init__(self, num_samples=500):
        super(FantasyDataset, self).__init__()
        # Dummy input dimensions (same as in each embedding)
        self.player_stats = torch.randn(num_samples, 100)  # raw stats for player autoencoder input
        # For opponent, assume we have a set of player embeddings per opponent team
        # We'll simulate a set of embeddings (e.g., 11 opponent players, embedding size 64)
        self.opponent_embeddings = torch.randn(num_samples, 11, 64)
        # Dummy venue features for VenueEmbedding module (input dim = 20)
        self.venue_features = torch.randn(num_samples, 20)
        # Dummy context features (e.g., 10-dimensional match-specific features)
        self.context_features = torch.randn(num_samples, 10)
        # Target fantasy points
        self.targets = torch.randn(num_samples, 1)

    def __len__(self):
        return self.player_stats.size(0)

    def __getitem__(self, idx):
        return {
            'player_stats': self.player_stats[idx],
            'opponent_embeddings': self.opponent_embeddings[idx],
            'venue_features': self.venue_features[idx],
            'context_features': self.context_features[idx],
            'target': self.targets[idx]
        }

def train_fantasy_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters and dimensions
    player_input_dim = 100
    player_latent_dim = 64
    player_hidden_dim = 100

    venue_input_dim = 20
    venue_embedding_dim = 16
    venue_hidden_dim = 32

    # Context dimension and opponent embedding dimension (assumed same as player_latent_dim)
    context_dim = 10
    opponent_emb_dim = player_latent_dim  # 64

    # Instantiate pretrained embedding models (in practice, load pretrained weights)
    # For demo, we instantiate and assume they are pretrained and frozen.
    player_model = AutoencoderSupervised(player_input_dim, player_latent_dim, player_hidden_dim).to(device)
    # Assume pretrained -> freeze
    for param in player_model.parameters():
        param.requires_grad = False

    opponent_model = OpponentEmbedding(opponent_emb_dim).to(device)
    # Opponent embedding is learned dynamically, so keep training.
    
    venue_model = VenueEmbedding(venue_input_dim, venue_embedding_dim, venue_hidden_dim).to(device)
    # Assume pretrained -> freeze
    for param in venue_model.parameters():
        param.requires_grad = False

    # Fantasy predictor model (trainable)
    predictor = FantasyPointPredictor(player_latent_dim, opponent_emb_dim, venue_embedding_dim, context_dim).to(device)

    optimizer = optim.Adam(list(opponent_model.parameters()) + list(predictor.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    # DataLoader
    dataset = FantasyDataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 20
    for epoch in range(num_epochs):
        predictor.train()
        opponent_model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            # Obtain player embedding E_P using autoencoder encoder
            stats = batch['player_stats'].to(device)
            with torch.no_grad():
                _, _, player_emb = player_model(stats)
            
            # Compute opponent embedding E_{Opp}
            opp_embs = batch['opponent_embeddings'].to(device)  # shape: (batch, 11, 64)
            # Process each sample's opponent players independently
            opponent_emb = []
            for opp in opp_embs:
                opp_emb = opponent_model(opp)  # shape: (64,)
                opponent_emb.append(opp_emb)
            opponent_emb = torch.stack(opponent_emb, dim=0)

            # Get venue embedding E_{Venue}
            venue_feats = batch['venue_features'].to(device)
            with torch.no_grad():
                venue_emb, _ = venue_model(venue_feats)

            # Get context features
            context = batch['context_features'].to(device)

            # Forward pass through fantasy point predictor
            predictions = predictor(player_emb, opponent_emb, venue_emb, context)
            target = batch['target'].to(device)
            loss = loss_fn(predictions, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * stats.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    train_fantasy_model()
#     dataset = FantasyDataset(num_samples=num_samples)