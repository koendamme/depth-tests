import torch
from GAN.dataset import PreiswerkDataset
from torch.utils.data import DataLoader, random_split
from ProGAN import Generator, Discriminator, ConditionalProGAN
from tqdm import tqdm
import math


def get_alpha(curr_epoch, epochs_per_step, quickness):
    alpha = quickness*(curr_epoch % epochs_per_step)/epochs_per_step

    return alpha if alpha <= 1 else 1


def get_step(n_epochs, total_steps, curr_epoch):
    return int((total_steps-1)/(n_epochs-1) * curr_epoch)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = PreiswerkDataset("B", device=device)
    train_length = int(len(dataset) * .9)
    train, test = random_split(dataset, [train_length, len(dataset) - train_length])
    batch_size = 8
    n_epochs = 100
    desired_resolution = 256
    noise_vector_length = 256

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    cProGAN = ConditionalProGAN(noise_vector_length, device, dataset[0][1].shape[0], desired_resolution)
    for i in range(n_epochs):
        D_loss, G_loss = cProGAN.train_single_epoch(train_dataloader, n_epochs, i)
        # print(D_loss, G_loss)


if __name__ == '__main__':
    main()
