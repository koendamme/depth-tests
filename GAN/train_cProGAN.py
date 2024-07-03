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


def eval(cProGAN, val_patterns, val_loaders):
    vids = []
    ssim_res, nmse_res = {}, {}
    for pattern, val_loader in zip(val_patterns, val_loaders):
        fake_imgs, real_imgs, nmse, ssim = cProGAN.evaluate(val_loader)
        ssim_res[pattern] = ssim.mean()
        nmse_res[pattern] = nmse.mean()
        vid = np.concatenate([fake_imgs, real_imgs[:, None, :, :]], axis=3)
        vid = np.uint8((vid + 1) / 2 * 255).repeat(3, axis=1)
        vids.append(wandb.Video(vid, fps=5, caption=pattern))

    return vids, ssim_res, nmse_res


def train(subject):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict(
        batch_size=12,
        n_epochs=120,
        desired_resolution=128,
        noise_vector_length=32,
        G_learning_rate=0.001,
        D_learning_rate=0.001,
        GP_lambda=40,
        n_critic=3,
        patient=subject,
        surrogates="All",
        D_layers= [8, 32, 64, 128, 256, 512],
        G_layers = [512, 256, 128, 64, 32, 8]
    )

    run = wandb.init(project=f"CustomData-cProGAN-All_Surrogates", config=config, tags=[subject, "no-surr-processor"])

    data_root = os.path.join("C:", os.sep, "data", "Formatted_datasets")
    dataset = CustomDataset(data_root, config["patient"])
    splitter = DatasetSplitter(dataset, .8, .1, .1)

    train_dataloader = DataLoader(splitter.get_train_dataset(), batch_size=config["batch_size"], shuffle=True, pin_memory=True)

    val_patterns = ["Regular Breathing", "Shallow Breathing", "Deep Breathing"]
    val_loaders = []
    for pattern in val_patterns:
        loader = DataLoader(splitter.val_subsets[pattern], batch_size=10, shuffle=False, pin_memory=True)
        val_loaders.append(loader)

    cProGAN = ConditionalProGAN(
        noise_vector_length=config["noise_vector_length"],
        device=device,
        desired_resolution=config["desired_resolution"],
        G_lr=config["G_learning_rate"],
        D_lr=config["D_learning_rate"],
        n_critic=config["n_critic"],
        n_epochs=config["n_epochs"],
        total_surrogate_length=dataset.signals_between_mrs*3,
        D_layers=config["D_layers"],
        G_layers=config["G_layers"]
    )
    artifact = wandb.Artifact('model', type='model')
    for i in range(config["n_epochs"]):
        start_time = time.time()
        D_loss, G_loss = cProGAN.train_single_epoch(train_dataloader, i, gp_lambda=config["GP_lambda"])
        end_time = time.time()
        vids, ssim, nmse = eval(cProGAN, val_patterns, val_loaders)

        model_save_path = os.path.join("model_checkpoints", run.name)

        if cProGAN.curr_step == cProGAN.total_steps - 1:
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
            torch.save(cProGAN.G.state_dict(), os.path.join(model_save_path, f"Epoch{i}.pth"))
            artifact.add_file(os.path.join(model_save_path, f"Epoch{i}.pth"))
            run.log_artifact(artifact)

        wandb.log({
            "D_loss": D_loss,
            "G_loss": G_loss,
            "Epoch": i,
            "Training_epoch_time": end_time - start_time,
            "Videos": vids,
            "SSIM": ssim,
            "NMSE": nmse
        })

        del vids
        gc.collect()
    run.finish()


def main():
    subjects = ["A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
    for s in subjects:
        train(s)


if __name__ == '__main__':
    main()
