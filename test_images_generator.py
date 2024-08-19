import torch
import os
from GAN.dataset import CustomDataset
from GAN.dataset_splitter import DatasetSplitter
from GAN.models.cProGAN import Generator
from torch.utils.data import DataLoader
import numpy as np
from GAN.utils import get_mean_std
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
    # Combined model runs
    # runs = {
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

    # Only heat model runs
    runs = {
        "A1": "peachy-moon-258",
        "A2": "peach-cosmos-225",
        "A3": "dandy-tree-226",
        "B1": "daily-sea-227",
        "B2": "winter-forest-228",
        "B3": "proud-puddle-229",
        "C1": "fresh-snow-230",
        "C2": "driven-durian-231",
        "C3": "blooming-frost-232"  
    }

    # Only coil model runs
    # runs = {
    #     "A1": "effortless-oath-256",
    #     "A2": "volcanic-planet-233",
    #     "A3": "ruby-haze-234",
    #     "B1": "breezy-darkness-235",
    #     "B2": "twilight-totem-236",
    #     "B3": "fiery-tree-237",
    #     "C1": "firm-dawn-238",
    #     "C2": "peachy-shadow-239",
    #     "C3": "vague-paper-240"
    # }

    # Only us model runs
    # runs = {
    #     "A1": "lucky-plant-257",
    #     "A2": "cool-aardvark-242",
    #     "A3": "classic-blaze-243",
    #     "B1": "restful-voice-244",
    #     "B2": "treasured-bush-245",
    #     "B3": "jolly-salad-246",
    #     "C1": "toasty-bird-247",
    #     "C2": "sleek-microwave-248",
    #     "C3": "dutiful-dawn-249"
    # }


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = os.path.join("F:", os.sep, "Formatted_datasets")
    save_root = os.path.join("F:", os.sep, "results", "heat_model")

    for subject in runs.keys():
        if subject != "A1":
            continue
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
            coil_length=0, #dataset[0]["coil"].shape[0],
            us_length=0, #dataset[0]["us_wave"].shape[0],
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
                us_wave_batch = None #data["us_wave"].to(device)
                coil_batch = None #data["coil"].to(device)
                heat_batch = data["heat"].to(device)

                noise_batch = torch.randn(mr_batch.shape[0], 256-32, 1, 1, device=device)
                fake_batch = G(noise_batch, us_wave_batch, coil_batch, heat_batch, 5, 1)

                fake_batch_processed = (fake_batch[:, 0, :, :].detach().cpu().numpy()+1)/2*255
                real_batch_processed = (mr_batch.detach().cpu().numpy()+1)/2*255

                for fake, real in zip(fake_batch_processed, real_batch_processed):
                    concat = np.concatenate([fake, real], axis=1)
                    cv2.imwrite(os.path.join(save_root, subject, pattern, f"{i}.png"), concat)
                    i+=1


if __name__ == '__main__':
    G = Generator(
        heat_length=14,
        coil_length=14,  # dataset[0]["coil"].shape[0],
        us_length=14,  # dataset[0]["us_wave"].shape[0],
        layers=[256, 128, 64, 32, 16, 8],
    )

    model_parameters = filter(lambda p: p.requires_grad, G.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


    # main()