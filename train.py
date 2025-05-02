# train_vae.py
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import VAE

data_path   = "data/images"
img_size    = 64
batch_size  = 128
lr          = 1e-3
epochs      = 50
latent_dim  = 128
device      = "cuda" if torch.cuda.is_available() else "cpu"

loader = get_dataloader(data_path, img_size, batch_size)
vae    = VAE(img_channels=3, feature_dim=32, latent_dim=latent_dim).to(device)
opt    = optim.Adam(vae.parameters(), lr=lr)
tb     = SummaryWriter("runs/vae_experiment")


def loss_fn(x_rec, x, mu, logvar):
    rec_loss = F.mse_loss(x_rec, x, reduction="sum")
    kl_loss  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec_loss + kl_loss, rec_loss, kl_loss

global_step = 0
for epoch in range(1, epochs+1):
    vae.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
    epoch_loss = 0
    for x, _ in pbar:
        x = x.to(device)
        x_rec, mu, logvar = vae(x)
        loss, rec_l, kl_l = loss_fn(x_rec, x, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()

        epoch_loss += loss.item()
        tb.add_scalar("Loss/total", loss.item(), global_step)
        tb.add_scalar("Loss/reconstruction", rec_l.item(), global_step)
        tb.add_scalar("Loss/KL", kl_l.item(), global_step)
        global_step += 1
    pbar.set_postfix(avg_loss=epoch_loss/len(loader.dataset))

    torch.save(vae.state_dict(), f"checkpoints/vae_epoch{epoch}.pt")

tb.close()
