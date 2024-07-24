import torch
import wandb
import json
import os
from dataset import CustomDataset
from dataset_splitter import DatasetSplitter
from models.cProGAN import Generator
from torch.utils.data import DataLoader
import numpy as np
from utils import get_mean_std
import cv2


def generate_fake_images(data, G, device):
    dataloader = DataLoader(data, batch_size=10, shuffle=False, pin_memory=True)

    fake_to_return = []
    real_to_return = []

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

    return real_to_return, fake_to_return


def main():
    runs = {
        "A2": "warm-frog-211",
        "A3": "toasty-smoke-212",
        "B1": "cool-rain-213",
        "B2": "atomic-firebrand-214",
        "B3": "devoted-oath-215",
        "C1": "volcanic-darkness-216",
        "C2": "major-sky-217",
        "C3": "helpful-flower-218",
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = os.path.join("C:", os.sep, "data", "Formatted_datasets")
    save_root = os.path.join("C:", os.sep, "data", "results")

    for subject in runs.keys():
        os.mkdir(os.path.join(save_root, subject))
        dataset = CustomDataset(data_root, subject)
        splitter = DatasetSplitter(dataset, .8, .1, .1)
        train_dataset = splitter.get_train_dataset()
        heat_normalizer, coil_normalizer, us_normalizer = get_mean_std(train_dataset)

        dataset = CustomDataset(data_root, subject, coil_normalizer, heat_normalizer, us_normalizer)
        splitter = DatasetSplitter(dataset, .8, .1, .1)

        model_path = f"C:\\dev\\depth-tests\\GAN\\best_models\\{runs[subject]}.pth"
        G = Generator(
            heat_length=dataset[0]["heat"].shape[0],
            coil_length=dataset[0]["coil"].shape[0],
            us_length=dataset[0]["us_wave"].shape[0],
            layers=[256, 128, 64, 32, 16, 8],
        ).to(device)
        G.load_state_dict(torch.load(model_path))
        G.eval()

        for pattern in ["Regular Breathing", "Shallow Breathing", "Deep Breathing", "Deep BH", "Half Exhale BH", "Full Exhale BH"]:
            os.mkdir(os.path.join(save_root, subject, pattern))
            data = splitter.test_subsets[pattern]
            dataloader = DataLoader(data, batch_size=10, shuffle=False, pin_memory=True)
            
            i = 0
            for data in dataloader:
                mr_batch = data["mr"].to(device)
                us_wave_batch = data["us_wave"].to(device)
                coil_batch = data["coil"].to(device)
                heat_batch = data["heat"].to(device)

                noise_batch = torch.randn(mr_batch.shape[0], 256-32, 1, 1, device=device)
                fake_batch = G(noise_batch, None, us_wave_batch, coil_batch, heat_batch, 5, 1)

                fake_batch_processed = (fake_batch[:, 0, :, :].detach().cpu().numpy()+1)/2*255
                real_batch_processed = np.uint8((mr_batch.detach().cpu().numpy()+1)/2*255)

                for fake, real in zip(fake_batch_processed, real_batch_processed):
                    concat = np.concatenate([fake, real], axis=1)
                    cv2.imwrite(os.path.join(save_root, subject, pattern, f"{i}.png"), concat)
                    i+=1


if __name__ == '__main__':
    main()