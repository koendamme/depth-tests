import torch
from dataset import CustomDataset
from utils import denormalize_tensor, scale_generator_output, create_video
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
from dataset_splitter import DatasetSplitter
import numpy as np
import gc


def get_mean_std(train_dataset):
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    data = next(iter(loader))
    
    heat = data["heat"].mean(), data["heat"].std()
    coil = data["coil"].mean(), data["coil"].std()
    us = data["us_wave"].mean(), data["us_wave"].std()

    return heat, coil, us



def eval(cProGAN, val_patterns, val_loaders, step, alpha):
    vids = []
    ssim_res, nmse_res = {}, {}
    n = 0
    mean_ssim, mean_nmse = 0, 0
    for pattern, val_loader in zip(val_patterns, val_loaders):
        fake_imgs, real_imgs, nmse, ssim = cProGAN.evaluate(val_loader, step, alpha)
        ssim_res[pattern] = ssim.mean()
        nmse_res[pattern] = nmse.mean()
        n += len(ssim)
        mean_ssim += ssim.mean() * len(ssim)
        mean_nmse += nmse.mean() * len(nmse)
        vid = np.concatenate([fake_imgs, real_imgs[:, None, :, :]], axis=3)
        vid = np.uint8((vid + 1) / 2 * 255).repeat(3, axis=1)
        vids.append(wandb.Video(vid, fps=5, caption=pattern))

    ssim_res["All"] = mean_ssim / n
    nmse_res["All"] = mean_nmse / n

    return vids, ssim_res, nmse_res


def train(subject):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict(
        batch_size=6,
        n_epochs=60,
        desired_resolution=128,
        G_learning_rate=0.001,
        D_learning_rate=0.001,
        GP_lambda=10,
        n_critic=1,
        patient=subject,
        surrogates="All",
        D_layers=[8, 16, 32, 64, 128, 256],
        G_layers=[256, 128, 64, 32, 16, 8]
    )

    run = wandb.init(project=f"CustomData-cProGAN-All_Surrogates", config=config, tags=[subject, "madore_us", "improved_sync"])

    data_root = os.path.join("C:", os.sep, "data", "Formatted_datasets")
    dataset = CustomDataset(data_root, config["patient"])
    splitter = DatasetSplitter(dataset, .8, .1, .1)
    train_dataset = splitter.get_train_dataset()
    heat_normalizer, coil_normalizer, us_normalizer = get_mean_std(train_dataset)

    dataset = CustomDataset(data_root, config["patient"], coil_normalizer, heat_normalizer, us_normalizer)
    splitter = DatasetSplitter(dataset, .8, .1, .1)
    train_dataset = splitter.get_train_dataset()

    val_patterns = ["Regular Breathing", "Shallow Breathing", "Deep Breathing", "Deep BH", "Half Exhale BH", "Full Exhale BH"]
    val_loaders = []
    for pattern in val_patterns:
        loader = DataLoader(splitter.val_subsets[pattern], batch_size=10, shuffle=False, pin_memory=True)
        val_loaders.append(loader)

    cProGAN = ConditionalProGAN(
        device=device,
        desired_resolution=config["desired_resolution"],
        G_lr=config["G_learning_rate"],
        D_lr=config["D_learning_rate"],
        n_critic=config["n_critic"],
        n_epochs=config["n_epochs"],
        D_layers=config["D_layers"],
        G_layers=config["G_layers"],
        heat_length=dataset[0]["heat"].shape[0],
        coil_length=dataset[0]["coil"].shape[0],
        us_length=dataset[0]["us_wave"].shape[0]
    )
    
    prog_epochs = [0, 0, 0, 10, 20, 30]
    batch_sizes = [0, 0, 0, 8, 8, 4]
    top_ssim = 0
    best_epoch = 0
    for step, n_epochs in enumerate(prog_epochs):
        alpha = 0
        train_dataloader = DataLoader(train_dataset, batch_size=batch_sizes[step], shuffle=True, pin_memory=True) if n_epochs != 0 else None

        for i in range(n_epochs):
            start_time = time.time()
            D_loss, G_loss, alpha = cProGAN.train_single_epoch(train_dataloader, sum(prog_epochs[:step])+i, config["GP_lambda"], step, alpha, n_epochs, len(train_dataset))
            end_time = time.time()
            vids, ssim, nmse = eval(cProGAN, val_patterns, val_loaders, step, alpha)

            if step == cProGAN.total_steps - 1 and alpha == 1 and ssim["All"] > top_ssim:
                top_ssim = ssim["All"]
                torch.save(cProGAN.G.state_dict(), f"C:\\dev\\depth-tests\\GAN\\best_models\\{run.name}.pth")

            wandb.log({
                "D_loss": D_loss,
                "G_loss": G_loss,
                "Epoch": i,
                "Training_epoch_time": end_time - start_time,
                "Videos": vids,
                "SSIM": ssim,
                "NMSE": nmse
            })

            del vids, ssim, nmse
            gc.collect()

    artifact = wandb.Artifact(f"{run.name}_epoch{best_epoch}", type='best model')
    artifact.add_file(f"C:\\dev\\depth-tests\\GAN\\best_models\\{run.name}.pth")
    run.log_artifact(artifact)
    run.finish()


def main():
    subjects = ["A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    for s in subjects:
        train(s)


if __name__ == '__main__':
    main()
