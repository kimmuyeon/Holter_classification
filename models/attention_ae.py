import torch.nn as nn
import torch

class AttentionAE(nn.Module):
    def __init__(self, input_dim=140, latent_dim=8):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, latent_dim), nn.ReLU()
        )
        # Attention scoring
        self.attn_layer = nn.Sequential(
            nn.Linear(latent_dim, 1), nn.Sigmoid()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, input_dim),  nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, input_dim]
        z     = self.encoder(x)               # [B, latent_dim]
        scores= self.attn_layer(z)            # [B, 1]
        ctx   = scores * z                    # [B, latent_dim]
        recon = self.decoder(ctx)             # [B, input_dim]
        return recon, scores                  # scores: importance

