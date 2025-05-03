# src/model.py
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self,
                 img_channels: int = 3,
                 feature_dim: int = 32,
                 latent_dim:   int = 128,
                 img_size:    tuple[int,int] = (64,64)):  # теперь принимаем img_size
        super().__init__()
        H, W = img_size

        # --- Encoder: 3× downsample (÷2³=÷8) ---
        self.enc = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim,     4, 2, 1),  # H,W → H/2, W/2
            nn.ReLU(),
            nn.Conv2d(feature_dim,   feature_dim*2, 4, 2, 1),  # → H/4, W/4
            nn.ReLU(),
            nn.Conv2d(feature_dim*2, feature_dim*4, 4, 2, 1),  # → H/8, W/8
            nn.ReLU(),
        )

        # Пробегаем через энкодер «фиктивный» тензор, чтобы узнать, какой у нас C,H_enc,W_enc
        with torch.no_grad():
            dummy = torch.zeros(1, img_channels, H, W)
            h_enc = self.enc(dummy)
        _, C, H_enc, W_enc = h_enc.shape

        self.C, self.H_enc, self.W_enc = C, H_enc, W_enc
        self.flatten_dim = C * H_enc * W_enc

        # Линейные слои для латента
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, self.flatten_dim)

        # --- Decoder: 3× upsample обратно до (H,W) ---
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, 4, 2, 1),  # H/8→H/4
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim*2, feature_dim,   4, 2, 1),  # H/4→H/2
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim,   img_channels, 4, 2, 1),  # H/2→H
            nn.Tanh(),  # выход в [-1,1]
        )

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1) Encoder
        h = self.enc(x)                                   # (B, C, H_enc, W_enc)
        h = h.view(x.size(0), self.flatten_dim)           # (B, flatten_dim)

        # 2) Латент
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        # 3) Decoder
        h_dec = self.fc_dec(z).view(                       # (B, C, H_enc, W_enc)
            x.size(0), self.C, self.H_enc, self.W_enc
        )
        x_rec = self.dec(h_dec)                           # (B, img_channels, H, W)

        return x_rec, mu, logvar
