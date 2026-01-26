import torch
import torch.nn as nn

class MutationEffectTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.out_head(x).squeeze(-1)

