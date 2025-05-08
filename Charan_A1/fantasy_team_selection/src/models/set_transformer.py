# set_transformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Building Blocks ---
class MultiheadAttention(nn.Module):
    """ Standard Multihead Attention """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False) # Bias often False in Transformers
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x shape: (B, N, C) - Batch, Sequence/Set Length, Embedding Dim
        # attn_mask shape: (B, 1, N, N) - expects 1/True where attention should be masked
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Shape: (B, num_heads, N, head_dim)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale # Shape: (B, num_heads, N, N)

        if attn_mask is not None:
             # Ensure mask dtype matches weights if needed, add large negative number
             # Mask expects True where values should be ignored.
             attn_weights = attn_weights.masked_fill(attn_mask == 1, float('-inf'))

        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights) # Dropout on attention weights

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C) # Combine heads
        output = self.out_proj(attn_output)
        return output

class FeedForward(nn.Module):
    """ Standard Feed Forward Network """
    def __init__(self, embed_dim, hidden_dim, dropout=0.1, activation=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SetAttentionBlock(nn.Module):
    """ Pre-Norm Transformer Block """
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout) # Dropout after attention/FFN modules

    def forward(self, x, attn_mask=None):
        # x shape: (B, N, C)
        # attn_mask shape: (B, 1, N, N) for MHA
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out) # Apply dropout after adding residual

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout(ffn_out) # Apply dropout after adding residual
        return x

# --- Main Multi-Task Model ---
class SetTransformerMultiTask(nn.Module):
    def __init__(self, input_feature_dim, embed_dim, num_heads, num_transformer_blocks, hidden_dim_transformer_factor,
                 point_head_layers, prob_head_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Initial Embedding Layer (MLP phi)
        self.input_embed = nn.Sequential(
            nn.Linear(input_feature_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout), # Dropout after activation
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim) # LayerNorm after final linear projection
        )

        # Optional: Positional Encoding (if order matters, usually not for sets but sometimes added)
        # self.pos_embed = nn.Parameter(torch.randn(1, max_set_size, embed_dim) * 0.02)

        # 2. Set Transformer Blocks
        transformer_hidden_dim = embed_dim * hidden_dim_transformer_factor
        self.transformer_blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads, transformer_hidden_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        # Final LayerNorm after transformer blocks
        self.norm_out = nn.LayerNorm(embed_dim)

        # 3. Point Prediction Head
        self.point_head = self._create_mlp_head(embed_dim, point_head_layers, 1, dropout)

        # 4. Probability Prediction Head
        self.prob_head = self._create_mlp_head(embed_dim, prob_head_layers, 1, dropout)


    def _create_mlp_head(self, input_dim, layer_dims, output_dim, dropout):
        layers = []
        current_dim = input_dim
        for h_dim in layer_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x_set, attn_mask=None):
        """
        Args:
            x_set (Tensor): Input features (batch, set_size, num_features)
            attn_mask (Tensor, optional): Padding mask (batch, set_size), True where padded.
                                         Will be converted to MHA mask internally.
        Returns:
            Tuple[Tensor, Tensor]: predicted_points_scaled, predicted_probabilities
        """
        B, N, _ = x_set.shape

        # 1. Get initial embeddings
        embeddings = self.input_embed(x_set) # Shape: (B, N, C)
        # Optional: Add positional encodings here if used
        # embeddings = embeddings + self.pos_embed[:, :N, :]

        # Prepare attention mask for MHA: (B, 1, N, N)
        mha_mask = None
        if attn_mask is not None:
            # Input attn_mask shape: (B, N), True if padded
            # Expand to (B, 1, N, N): Compare each element with every other
            mha_mask = attn_mask.unsqueeze(1).unsqueeze(2) # -> (B, 1, 1, N)
            mha_mask = mha_mask.expand(-1, -1, N, -1) # -> (B, 1, N, N)
            # Or simply: mha_mask = attn_mask[:, None, None, :] | attn_mask[:, None, :, None] # Combine row/col masks

        # 2. Pass through Transformer blocks
        transformer_output = embeddings
        for block in self.transformer_blocks:
             transformer_output = block(transformer_output, attn_mask=mha_mask)

        # Apply final LayerNorm
        transformer_output = self.norm_out(transformer_output) # (B, N, C)

        # 3. Predict Points
        predicted_points_scaled = self.point_head(transformer_output) # Shape: (B, N, 1)

        # 4. Predict Probabilities
        prob_logits = self.prob_head(transformer_output) # Shape: (B, N, 1)
        predicted_probabilities = torch.sigmoid(prob_logits) # Apply sigmoid

        return predicted_points_scaled, predicted_probabilities