import torch
from GAN.dataset import PreiswerkDataset, CustomDataset
from GAN.utils import denormalize_tensor, scale_generator_output, create_video
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from models.cProGAN import Generator, Discriminator, ConditionalProGAN
from tqdm import tqdm
from datetime import datetime
import time
import math
import wandb
from torchvision.utils import save_image
import os
from GAN.dataset_splitter import DatasetSplitter
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict(
        batch_size=16,
        n_epochs=40,
        desired_resolution=192,
        noise_vector_length=128,
        G_learning_rate=0.0001,
        D_learning_rate=0.001,
        GP_lambda=10,
        n_critic=1,
        patient="A",
        surrogates="US"
    )

    wandb.init(project="CustomData-cProGAN-mr_surrogate", config=config)

    surrogate_freq, mri_freq = 50, 2.9
    dataset = CustomDataset("C:\data", config["patient"], (500, 1000), int(surrogate_freq//mri_freq))
    splitter = DatasetSplitter(dataset, .8, .1, .1)

    train_dataloader = DataLoader(splitter.get_train_dataset(), batch_size=config["batch_size"], shuffle=True)

    val_patterns = ["Regular Breathing", "Shallow Breathing", "Deep Breathing"]
    val_loaders = []
    for pattern in val_patterns:
        loader = DataLoader(splitter.val_subsets[pattern], batch_size=10, shuffle=False)
        val_loaders.append(loader)

    cProGAN = ConditionalProGAN(
        noise_vector_length=config["noise_vector_length"],
        device=device,
        desired_resolution=config["desired_resolution"],
        G_lr=config["G_learning_rate"],
        D_lr=config["D_learning_rate"],
        n_critic=config["n_critic"],
    )

    for i in range(config["n_epochs"]):
        start_time = time.time()
        D_loss, G_loss = cProGAN.train_single_epoch(train_dataloader, config["n_epochs"], i, gp_lambda=config["GP_lambda"])
        end_time = time.time()

        for pattern, val_loader in zip(val_patterns, val_loaders):
            fake_imgs, real_imgs, _ = cProGAN.evaluate(val_loader)
            video = create_video(fake_imgs, real_imgs)
            wandb.log({
                f"{pattern} Video": wandb.Video(video, fps=5)
            }, step=i)

        wandb.log({
            "D_loss": D_loss,
            "G_loss": G_loss,
            "Epoch": i,
            "Training_epoch_time": end_time - start_time,
        }, step=i)


if __name__ == '__main__':
    main()
