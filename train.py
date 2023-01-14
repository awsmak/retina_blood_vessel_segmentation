import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from UNET.model import BuildUnet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from data import DriveDataset

if __name__ == "__main__":
    # Seeding
    seeding(42)

    # Directories
    create_dir("files")

    # Load dataset
    train_x = glob("augmented_data/train/images/*")
    train_y = glob("augmented_data/train/masks*")

    val_x = glob("augmented_data/test/images/*")
    val_y = glob("augmented_data/test/masks*")

    data_str = f"Dataset Size:\nTrain:{len(train_x)}, valid:{len(val_x)}"
    print(data_str)

    # Hyper parameters
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    # Dataset and Dataloader
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(val_x, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2

    )

    # Model
    device = torch.device('cuda')
    model = BuildUnet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """Training the model"""
    

