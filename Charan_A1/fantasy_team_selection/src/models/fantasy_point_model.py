import torch
import torch.nn as nn

class FantasyPointPredictor(nn.Module):
    def __init__(self, player_dim, opponent_dim, venue_dim, context_dim, hidden_dims=[128, 64]):
        super(FantasyPointPredictor, self).__init__()
        # Combined input dimension
        input_dim = player_dim + opponent_dim + venue_dim + context_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)  # output fantasy points (scalar)
        )

    def forward(self, player_emb, opponent_emb, venue_emb, context):
        # Concatenate all features
        x = torch.cat([player_emb, opponent_emb, venue_emb, context], dim=-1)
        prediction = self.model(x)
        return prediction

if __name__ == '__main__':
    # Dimensions
    player_dim = 64
    opponent_dim = 64
    venue_dim = 16
    context_dim = 10  # e.g., match context features

    # Instantiate the FantasyPointPredictor
    predictor = FantasyPointPredictor(player_dim, opponent_dim, venue_dim, context_dim)
  
    # Import embedding modules (assumed available)
    from src.embeddings.player_embedding import AutoencoderSupervised
    from src.embeddings.opponent_embedding import OpponentEmbedding
    from src.embeddings.venue_embedding import VenueEmbedding

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor.to(device)
    
    # Instantiate embedding modules (dummy/pretrained instances)
    player_model = AutoencoderSupervised(input_dim=100, latent_dim=player_dim, hidden_dim=100).to(device)
    opponent_model = OpponentEmbedding(embedding_dim=player_dim).to(device)
    venue_model = VenueEmbedding(input_dim=20, embedding_dim=venue_dim, hidden_dim=32).to(device)
    
    # Create dummy inputs for embeddings
    dummy_player_stats = torch.randn(1, 100).to(device)         # single player's raw stats
    dummy_opponent_stats = torch.randn(11, 100).to(device)        # simulate raw stats for 11 opponent players
    dummy_venue_features = torch.randn(1, 20).to(device)          # single venue features
    dummy_context = torch.randn(1, context_dim).to(device)        # dummy context features
    
    # Retrieve individual embeddings:
    with torch.no_grad():
        # Player embedding E_P from autoencoder bottleneck
        _, _, player_embedding = player_model(dummy_player_stats)
        # For opponent, first get embeddings for each opponent player 
        # (simulate with same player_model for demonstration)
        _, _, opp_embeddings = player_model(dummy_opponent_stats)
        opponent_embedding = opponent_model(opp_embeddings)
        # Venue embedding from venue module
        venue_embedding, _ = venue_model(dummy_venue_features)
    
    # Feed concatenated features into the predictor
    prediction = predictor(player_embedding, opponent_embedding, venue_embedding, dummy_context)
    
    print("Predicted Fantasy Points:", prediction.item())
