# generate_vae.py
import torch
import torchvision.utils as vutils
from model import VAE
import os

os.makedirs("samples", exist_ok=True)

device     = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 128
img_size   = (448, 544)    # ← ОБЯЗАТЕЛЬНО тот же, что и в train.py
feature_dim = 32           # тоже должен совпадать
checkpoint = "checkpoints/vae_epoch200.pt"

# 1) Инициализируем модель с тем же img_size и feature_dim
vae = VAE(
    img_channels=3,
    feature_dim=feature_dim,
    latent_dim=latent_dim,
    img_size=img_size
).to(device)

vae.load_state_dict(torch.load(checkpoint, map_location=device))
vae.eval()

# 2) Сэмплинг латентных векторов
n_samples = 64
z = torch.randn(n_samples, latent_dim).to(device)

# 3) Генерация через динамический reshape
with torch.no_grad():
    h_dec = vae.fc_dec(z).view(
        -1,
        vae.C,        # число каналов после энкодера
        vae.H_enc,    # высота after downsampling
        vae.W_enc     # ширина  after downsampling
    )
    samples = vae.dec(h_dec)

# 4) Нормализация и сохранение
samples = (samples + 1) / 2  # из [-1,1] → [0,1]
for i, img in enumerate(samples):
    vutils.save_image(img, f"samples/sample_{i:02d}.png")

print(f"Сохранено {n_samples} сэмплов в папку samples/")
