import torch
import os
from GAN.dataset import CustomDataset
from GAN.dataset_splitter import DatasetSplitter
from GAN.models.cProGAN import Generator
from torch.utils.data import DataLoader
import numpy as np
from GAN.utils import get_mean_std
import cv2
from run_names import combined_runs, heat_runs, coil_runs, us_runs


# def generate_fake_images(data, G, device):
#     dataloader = DataLoader(data, batch_size=10, shuffle=False, pin_memory=True)

#     fake_to_return = []
#     real_to_return = []

#     for data in dataloader:
#         mr_batch = data["mr"].to(device)
#         wave_batch = None
#         us_wave_batch = data["us_wave"].to(device)
#         coil_batch = data["coil"].to(device)
#         heat_batch = data["heat"].to(device)
#         us_raw_batch = None

#         noise_batch = torch.randn(mr_batch.shape[0], 256-32, 1, 1, device=device)
#         fake_batch = G(noise_batch, wave_batch, us_wave_batch, coil_batch, heat_batch, us_raw_batch, 5, 1)

#         fake_to_return.extend(np.uint8((fake_batch[:, 0, :, :].detach().cpu().numpy()+1)/2*255))
#         real_to_return.extend(np.uint8((mr_batch.detach().cpu().numpy()+1)/2*255))

#     return real_to_return, fake_to_return


def main():
    runs = us_runs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = os.path.join("F:", os.sep, "Formatted_datasets")
    save_root = os.path.join("F:", os.sep, "results", "us_model")
    subjects = ["D1", "D2", "D3", "E1", "E2", "E3", "F1", "F3", "F4", "G2", "G3", "G4"]

    for subject in subjects:
        os.mkdir(os.path.join(save_root, subject))
        dataset = CustomDataset(data_root, subject)
        splitter = DatasetSplitter(dataset, .8, .1, .1)
        train_dataset = splitter.get_train_dataset()
        heat_normalizer, coil_normalizer, us_normalizer = get_mean_std(train_dataset)

        dataset = CustomDataset(data_root, subject, coil_normalizer, heat_normalizer, us_normalizer)
        splitter = DatasetSplitter(dataset, .8, .1, .1)

        model_path = f"C:\\dev\\depth-tests\\GAN\\best_models\\{runs[subject]}.pth"
        G = Generator(
            heat_length=0, #dataset[0]["heat"].shape[0],
            coil_length=0, #dataset[0]["coil"].shape[0],
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
                coil_batch = None #data["coil"].to(device)
                heat_batch = None #data["heat"].to(device)

                noise_batch = torch.randn(mr_batch.shape[0], 256-32, 1, 1, device=device)
                fake_batch = G(noise_batch, us_wave_batch, coil_batch, heat_batch, 5, 1)

                fake_batch_processed = (fake_batch[:, 0, :, :].detach().cpu().numpy()+1)/2*255
                real_batch_processed = (mr_batch.detach().cpu().numpy()+1)/2*255

                for fake, real in zip(fake_batch_processed, real_batch_processed):
                    concat = np.concatenate([fake, real], axis=1)
                    cv2.imwrite(os.path.join(save_root, subject, pattern, f"{i}.png"), concat)
                    i+=1


if __name__ == '__main__':
    main()