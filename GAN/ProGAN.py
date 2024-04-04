import torch
from torch.utils.data import DataLoader
from GAN.dataset import PreiswerkDataset


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2),
        ])

    def forward(self, input):
        x = self.block(input)
        return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.initial_block = torch.nn.Sequential(*[
            torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, padding=0),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2)
        ])

        self.layers = torch.nn.ModuleList([])

    def add_layer(self):
        self.layers.append(ConvBlock(512, 512))

    def forward(self, input):
        x = self.initial_block(input)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


class ProGAN(torch.nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data = PreiswerkDataset("B", device)
    # dataloader = DataLoader(data, batch_size=8, shuffle=True)
    # batch = next(iter(dataloader))

    G = Generator(512)
    block1 = ConvBlock(512, 512)

    latent_vector = torch.randn((4, 512, 1, 1))
    print(latent_vector.shape)
    o = G(latent_vector)
    print(o.shape)

    o = block1(o)

    print(o.shape)


if __name__ == '__main__':
    main()
