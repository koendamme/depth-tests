import torch
from GAN.dataset import PreiswerkDataset
from GAN.utils import denormalize_tensor, scale_generator_output
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from ProGAN import Generator, Discriminator, ConditionalProGAN
from tqdm import tqdm
from datetime import datetime
import time
import math
import wandb
from torchvision.utils import save_image
import os


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict(
        batch_size=8,
        n_epochs=1000,
        desired_resolution=256,
        noise_vector_length=256,
        G_learning_rate=0.001,
        D_learning_rate=0.001,
        GP_lambda=10,
        patient="H",
        surrogates="US"
    )

    wandb.init(project="Preiswerk-cProGAN", config=config)

    dataset = PreiswerkDataset(config["patient"], device=device)
    train_length = int(len(dataset) * .9)
    train, test = random_split(dataset, [train_length, len(dataset) - train_length])
    train_dataloader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    cProGAN = ConditionalProGAN(
        noise_vector_length=config["noise_vector_length"],
        device=device,
        desired_resolution=config["desired_resolution"],
        G_lr=config["G_learning_rate"],
        D_lr=config["D_learning_rate"]
    )

    datestring = datetime.now().strftime("%d-%m-%H%M")
    os.mkdir(os.path.join("train_video", datestring))

    for i in range(config["n_epochs"]):
        start_time = time.time()
        D_loss, G_loss = cProGAN.train_single_epoch(train_dataloader, config["n_epochs"], i, gp_lambda=config["GP_lambda"])
        end_time = time.time()

        test_imgs, real_imgs = cProGAN.evaluate(test_dataloader, i)

        wandb.log({
            "D_loss": D_loss,
            "G_loss": G_loss,
            "Test_images": [wandb.Image(scale_generator_output(test_imgs[i]), caption=f"Test Image {i}") for i in range(test_imgs.shape[0])],
            "Real_images": [wandb.Image(scale_generator_output(real_imgs[i]), caption=f"Real Image {i}") for i in range(real_imgs.shape[0])],
            "Epoch": i,
            "Training_epoch_time": end_time - start_time
        })

        save_image(scale_generator_output(test_imgs[0]), os.path.join("train_video", datestring, f"Epoch{i}_0_fake.png"))
        save_image(scale_generator_output(real_imgs[0]), os.path.join("train_video", datestring, f"Epoch{i}_0_real.png"))
        save_image(scale_generator_output(test_imgs[1]), os.path.join("train_video", datestring, f"Epoch{i}_1_fake.png"))
        save_image(scale_generator_output(real_imgs[1]), os.path.join("train_video", datestring, f"Epoch{i}_1_real.png"))


if __name__ == '__main__':
    main()
