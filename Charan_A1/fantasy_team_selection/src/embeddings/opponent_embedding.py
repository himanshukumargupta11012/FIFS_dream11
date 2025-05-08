import torch
import torch.nn as nn

# Opponent Embedding Module using attention over player embeddings.
class OpponentEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(OpponentEmbedding, self).__init__()
        # Attention: projects each player embedding to a scalar score.
        self.attention = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, player_embeddings):
        # player_embeddings: tensor of shape (num_players, embedding_dim)
        # Compute attention scores and apply softmax to obtain weights.
        attn_scores = self.attention(player_embeddings)
        attn_weights = torch.softmax(attn_scores, dim=0)
        # Compute weighted sum -> opponent embedding.
        opponent_embedding = torch.sum(attn_weights * player_embeddings, dim=0)
        return opponent_embedding

# Example usage:
if __name__ == '__main__':
    embedding_dim = 64
    num_players = 11  # Example number of players in opponent team

    # Dummy player embeddings
    dummy_player_embeddings = torch.randn(num_players, embedding_dim)

    opponent_embedder = OpponentEmbedding(embedding_dim)
    opp_embedding = opponent_embedder(dummy_player_embeddings)
    print("Opponent Embedding:", opp_embedding)
