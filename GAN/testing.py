import numpy as np
import wandb
import pandas as pd
import os
from models.cProGAN import Generator
import torch
import json
from dataset import CustomDataset
from dataset_splitter import DatasetSplitter
from torch.utils.data import DataLoader
import cv2


def generate_fake_images(data, G, device, noise_vector_length):
    dataloader = DataLoader(data, batch_size=8, shuffle=False)

    fake, real = None, None
    for data in dataloader:
        mr_batch = data["mr"]
        wave_batch = None
        us_batch = data["us_wave"]
        coil_batch = data["coil"]
        heat_batch = data["heat"]
        mr_batch, heat_batch = mr_batch.to(device), heat_batch.to(device)
        us_batch, coil_batch = us_batch.to(device), coil_batch.to(device)

        noise_batch = torch.randn(mr_batch.shape[0], noise_vector_length, 1, 1, device=device)
        fake_batch = G(noise_batch, wave_batch, us_batch, coil_batch, heat_batch, 5, 1)

        if fake is None and real is None:
            real = mr_batch
            fake = fake_batch.squeeze()
        else:
            fake = torch.concatenate([fake, fake_batch.squeeze()], axis=0)
            real = torch.concatenate([real, mr_batch], axis=0)

    return real, fake


    return [], []


def get_optimal_model(wandb_run, splitter):
    history = wandb_run.history()
    ssim_values = history.loc[:, ["SSIM.Deep Breathing", "SSIM.Shallow Breathing", "SSIM.Regular Breathing"]].to_numpy()
    db, sb, rb = len(splitter.val_subsets["Deep Breathing"]), len(splitter.val_subsets["Shallow Breathing"]), len(splitter.val_subsets["Regular Breathing"])
    ssim_averages = np.average(ssim_values, axis=1, weights=[db, sb, rb])
    idx = np.argmax(ssim_averages[110:]) + 110
    model_path = os.path.join("model_checkpoints", wandb_run.name, f"Epoch{idx}.pth")
    return model_path


def track_border(real, fake):
    for i in range(real.shape[0]):
        img = torch.concatenate([real[i], fake[i]], dim=1).cpu().detach().numpy()

        cv2.imshow("Frame", (img + 1)/2)
        cv2.waitKey(100)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    api = wandb.Api()
    run = api.run("thesis-koen/CustomData-cProGAN-All_Surrogates/94yonpz0")
    config = json.loads(run.json_config)

    dataset = CustomDataset(os.path.join("C:", os.sep, "data", "Formatted_datasets"), config["patient"]["value"])
    splitter = DatasetSplitter(dataset, .8, .1, .1)

    model_path = get_optimal_model(run, splitter)
    G = Generator(config["noise_vector_length"]["value"], dataset.signals_between_mrs*3, config["G_layers"]["value"]).to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()

    data = splitter.test_subsets["Regular Breathing"]
    real, fake = generate_fake_images(data, G, device, config["noise_vector_length"]["value"])
    track_border(real, fake)





if __name__ == '__main__':
    main()