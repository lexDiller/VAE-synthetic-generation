# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, img_channels=3, feature_dim=32, latent_dim=128):
        super().__init__()
        # --- Encoder ---
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim*2, 4, 2, 1), # 32→16
            nn.ReLU(),
            nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1),# 16→8
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(feature_dim*4*8*8, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim*4*8*8, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim, feature_dim*4*8*8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1), # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim*2, feature_dim,   4, 2, 1), # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim, img_channels,    4, 2, 1), # 32→64
            nn.Tanh(),  # выход в диапазоне [-1,1]
        )

    def reparameterize(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h_dec = self.fc_dec(z).view(-1, 128, 8, 8)
        x_rec = self.dec(h_dec)
        return x_rec, mu, logvar
