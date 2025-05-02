# generate_vae.py
import torch
import torchvision.utils as vutils
from model import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 128

vae = VAE(img_channels=3, feature_dim=32, latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load("checkpoints/vae_epoch50.pt", map_location=device))
vae.eval()

n_samples = 64
z = torch.randn(n_samples, latent_dim).to(device)
with torch.no_grad():
    h_dec = vae.fc_dec(z).view(-1, 128, 8, 8)
    samples = vae.dec(h_dec)


samples = (samples + 1) / 2
vutils.save_image(samples, "generated_vae.png", nrow=8)
print("Синтетические изображения сохранены в generated_vae.png")
