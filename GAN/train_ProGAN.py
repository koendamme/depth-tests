import torch
from GAN.dataset import PreiswerkDataset
from GAN.utils import denormalize_tensor, scale_generator_output
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from ProGAN import Generator, Discriminator, ConditionalProGAN
from tqdm import tqdm
import math
import wandb


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict(
        batch_size = 8,
        n_epochs = 250,
        desired_resolution = 256,
        noise_vector_length = 256,
        G_learning_rate = 0.001,
        D_learning_rate = 0.001,
        GP_lambda = 10
    )

    wandb.init(project="Preiswerk-cProGAN", config=config)

    dataset = PreiswerkDataset("B", device=device)
    train_length = int(len(dataset) * .9)
    train, test = random_split(dataset, [train_length, len(dataset) - train_length])
    train_dataloader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    cProGAN = ConditionalProGAN(
        noise_vector_length=config["noise_vector_length"],
        device=device,
        depth_feature_length=dataset[0][1].shape[0],
        desired_resolution=config["desired_resolution"],
        G_lr=config["G_learning_rate"],
        D_lr=config["D_learning_rate"]
    )

    for i in range(config["n_epochs"]):
        D_loss, G_loss = cProGAN.train_single_epoch(train_dataloader, config["n_epochs"], i, gp_lambda=config["GP_lambda"])
        test_imgs, real_imgs = cProGAN.evaluate(test_dataloader, i)
        # real_imgs = denormalize_tensor(real_imgs, dataset.mri_mu, dataset.mri_sigma)

        wandb.log({
            "D_loss": D_loss,
            "G_loss": G_loss,
            "Test_images": [wandb.Image(scale_generator_output(test_imgs[i]), caption=f"Test Image {i}") for i in range(test_imgs.shape[0])],
            "Real_images": [wandb.Image(scale_generator_output(real_imgs[i]), caption=f"Real Image {i}") for i in range(real_imgs.shape[0])],
            "Epoch": i
        })


if __name__ == '__main__':
    main()
