import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import numpy as np

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=8), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(128, 128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
 
    ]
)
val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

train_set_full = torchvision.datasets.CIFAR10(root='/home', train=True, 
                                              download=True, transform=train_transform)

val_ratio = 0.2
n_train_examples = int(len(train_set_full) * (1 - val_ratio))
n_val_examples = len(train_set_full) - n_train_examples

indices = np.arange(len(train_set_full))
np.random.shuffle(indices)
train_indices = indices[:n_train_examples]
val_indices = indices[n_train_examples:]

train_set = torch.utils.data.Subset(train_set_full, train_indices)

val_set_full = torchvision.datasets.CIFAR10(root='/home/', train=True, 
                                            download=True, transform=val_transform)
val_set = torch.utils.data.Subset(val_set_full, val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=50, shuffle=False, num_workers=4)
