import os
from GAN.dataset import CustomDataset
from GAN.models.cProGAN import Generator
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from run_names import combined_runs, heat_runs, coil_runs, us_runs
from tqdm import tqdm


def main():
    # combined_runs = {
    #     "A1": "major-sunset-255",
    #     "A2": "warm-frog-211",
    #     "A3": "toasty-smoke-212",
    #     "B1": "cool-rain-213",
    #     "B2": "atomic-firebrand-214",
    #     "B3": "devoted-oath-215",
    #     "C1": "volcanic-darkness-216",
    #     "C2": "major-sky-217",
    #     "C3": "helpful-flower-218",
    # }
    

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dataset = CustomDataset("F:\Formatted_datasets", "B1")

    model_path = f"C:\\dev\\depth-tests\\GAN\\best_models\\{combined_runs['B1']}.pth"

    G = Generator(
        heat_length=dataset[0]["heat"].shape[0],
        coil_length=dataset[0]["coil"].shape[0],
        us_length=dataset[0]["us_wave"].shape[0],
        layers=[256, 128, 64, 32, 16, 8],
    ).to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()

    times = []
    for _ in tqdm(range(10000)):
        us_wave_batch = torch.randn(1, dataset[0]["us_wave"].shape[0], device=device)
        coil_batch = torch.randn(1, dataset[0]["coil"].shape[0], device=device)
        heat_batch = torch.randn(1, dataset[0]["heat"].shape[0], device=device)
        noise_batch = torch.randn(1, 224, 1, 1, device=device)
        
        start = time.time()
        _ = G(noise_batch, us_wave_batch, coil_batch, heat_batch, 5, 1)
        end = time.time()

        elapsed = end - start
        times.append(elapsed)

    times = np.array(times)[1:]
    print(f"Mean: {times.mean()}")
    print(f"Std:  {times.std()}")


if __name__ == "__main__":
    main()