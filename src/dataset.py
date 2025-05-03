# src/dataset.py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, img_size=(448, 544)):
        self.paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg","jpeg","png"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0

def get_dataloader(path, img_size=(448,544), batch_size=64, shuffle=True, num_workers=4):
    ds = UnlabeledImageDataset(path, img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
