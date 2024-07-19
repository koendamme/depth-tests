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
import matplotlib.pyplot as plt


def extract_border(img, threshold, x, show=False):
    _, binary_mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    line = cleaned_mask[:, x]

    y = np.where(line != 0)[0][0]

    color_image = None
    if show:
        color_image = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(color_image, (x, y), 3, [255, 0, 0], 2)
    return y*1.9, color_image


def get_mean_std(train_dataset):
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    data = next(iter(loader))
    
    heat = data["heat"].mean(), data["heat"].std()
    coil = data["coil"].mean(), data["coil"].std()
    us = data["us_wave"].mean(), data["us_wave"].std()

    return heat, coil, us


def generate_fake_images(data, G, device):
    dataloader = DataLoader(data, batch_size=10, shuffle=False, pin_memory=True)

    fake_to_return = []
    real_to_return = []
    us = []
    coil = []
    heat = []
    for data in dataloader:
        mr_batch = data["mr"].to(device)
        wave_batch = None
        us_wave_batch = data["us_wave"].to(device)
        coil_batch = data["coil"].to(device)
        heat_batch = data["heat"].to(device)
        us_raw_batch = None

        noise_batch = torch.randn(mr_batch.shape[0], 256-32, 1, 1, device=device)
        fake_batch = G(noise_batch, wave_batch, us_wave_batch, coil_batch, heat_batch, us_raw_batch, 5, 1)

        fake_to_return.extend(np.uint8((fake_batch[:, 0, :, :].detach().cpu().numpy()+1)/2*255))
        real_to_return.extend(np.uint8((mr_batch.detach().cpu().numpy()+1)/2*255))
        us.extend(us_wave_batch.detach().cpu().numpy().flatten())
        coil.extend(coil_batch.detach().cpu().numpy().flatten())
        heat.extend(heat_batch.detach().cpu().numpy().flatten())

    return real_to_return, fake_to_return, us, coil, heat


def track_border(real, fake, threshold, x, show=False):    
    real_waveform, fake_waveform = [], []
    for r, f in zip(real, fake):
        y_real, image_real = extract_border(r, threshold, x)
        y_fake, image_fake = extract_border(f, threshold, x)

        real_waveform.append(y_real)
        fake_waveform.append(y_fake)

        if show:
            img = np.concatenate([image_fake, image_real], axis=1)
            cv2.imshow("Frame", img)
            cv2.waitKey(200)
    
    return real_waveform, fake_waveform


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    api = wandb.Api()
    run = api.run("thesis-koen/CustomData-cProGAN-All_Surrogates/wo3eotzf")
    config = json.loads(run.json_config)

    data_root = os.path.join("C:", os.sep, "data", "Formatted_datasets")
    dataset = CustomDataset(data_root, config["patient"]["value"])
    splitter = DatasetSplitter(dataset, .8, .1, .1)
    train_dataset = splitter.get_train_dataset()
    heat_normalizer, coil_normalizer, us_normalizer = get_mean_std(train_dataset)

    dataset = CustomDataset(data_root, config["patient"]["value"], coil_normalizer, heat_normalizer, us_normalizer)
    splitter = DatasetSplitter(dataset, .8, .1, .1)

    model_path = f"C:\\dev\\depth-tests\\GAN\\best_models\\{run.name}.pth"
    G = Generator(
        heat_length=dataset[0]["heat"].shape[0],
        coil_length=dataset[0]["coil"].shape[0],
        us_length=dataset[0]["us_wave"].shape[0],
        layers=config["G_layers"]["value"],
    ).to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()

    for pattern in ["Regular Breathing", "Shallow Breathing", "Deep Breathing", "Deep BH", "Half Exhale BH", "Full Exhale BH"]:
        data = splitter.test_subsets[pattern]
        real, fake, us, coil, heat = generate_fake_images(data, G, device)
        if config["patient"]["value"] == "A2":
            threshold = dataset.settings["MRI"]["Waveform parameters"]["Threshold"]
            x = dataset.settings["MRI"]["Waveform parameters"]["x"]
        else:
            threshold = dataset.settings["MRI"]["Updated_Waveform_parameters"]["Threshold"]
            x = dataset.settings["MRI"]["Updated_Waveform_parameters"]["x"]

        real_waveform, fake_waveform = track_border(real, fake, threshold, x-32)
        # x = np.linspace(0, len(real_waveform), len(real_waveform))
        # x_surrogates = np.linspace(0, len(real_waveform), len(us))

        # plt.figure()
        # fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        # axs[0].plot(x, real_waveform, label="Real")
        # axs[0].plot(x, fake_waveform, label="Fake")
        # axs[0].set_ylim([100, 200])
        # axs[0].set_title(f"Liver movement for {pattern}")
        # axs[0].set_ylabel("Displacement (mm)")
        # axs[0].set_xlabel("Frame number")
        # axs[0].legend()

        # axs[1].set_title("US")
        # axs[1].plot(x_surrogates, us)

        # axs[2].set_title("Heat")
        # axs[2].plot(x_surrogates, heat)

        # axs[3].set_title("Coil")
        # axs[3].plot(x_surrogates, coil)
        # plt.show()

        plt.figure()
        plt.plot(real_waveform, label="Real")
        plt.plot(fake_waveform, label="Fake")
        # plt.ylim([50, 150])
        plt.title(f"Liver movement for {pattern}")
        plt.ylabel("Displacement (mm)")
        plt.xlabel("Frame number")
        plt.legend()
        plt.savefig(f"{pattern}.png")
        print(f"{pattern} Done!")


if __name__ == '__main__':
    main()