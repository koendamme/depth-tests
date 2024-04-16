import torch
from GAN.dataset import PreiswerkDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from ProGAN import Generator, Discriminator, ConditionalProGAN
from tqdm import tqdm
import math


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)
    dataset = PreiswerkDataset("B", device=device)
    train_length = int(len(dataset) * .9)
    train, test = random_split(dataset, [train_length, len(dataset) - train_length])
    batch_size = 8
    n_epochs = 50
    desired_resolution = 256
    noise_vector_length = 256

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    cProGAN = ConditionalProGAN(noise_vector_length, device, dataset[0][1].shape[0], desired_resolution)
    d_loss_line, g_loss_line = [], []
    for i in range(n_epochs):
        D_loss, G_loss = cProGAN.train_single_epoch(train_dataloader, n_epochs, i, gp_lambda=10)
        # print(D_loss, G_loss)
        d_loss_line.append(D_loss)
        g_loss_line.append(G_loss)
        cProGAN.evaluate(test_dataloader, i)

    plt.figure()
    plt.plot(d_loss_line)
    plt.plot(g_loss_line)
    plt.savefig("learning curve.png")


if __name__ == '__main__':
    main()
