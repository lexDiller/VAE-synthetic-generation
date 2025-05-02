from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(path, img_size=64, batch_size=128, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.ImageFolder(root=path, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
