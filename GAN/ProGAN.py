import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GAN.dataset import PreiswerkDataset


class MiniBatchStd(torch.nn.Module):
    def __init__(self):
        super(MiniBatchStd, self).__init__()

    def forward(self, x):
        minibatch_std = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        return torch.cat([x, minibatch_std], dim=1)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
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

        self.togray_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1,),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1),
            torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        ])

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, padding=0),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.2)
            ]),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 16),
            ConvBlock(16, 8)
        ])

    def forward(self, x, step):
        for i in range(step + 1):
            # Dont upsample the first layer
            x = F.interpolate(x, scale_factor=2, mode='nearest') if i != 0 else x
            x = self.layers[i](x)

        x = self.togray_layers[step](x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fromgray_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1)
        ])

        self.layers = torch.nn.ModuleList([
            ConvBlock(16, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            torch.nn.Sequential(*[
                MiniBatchStd(),
                torch.nn.Conv2d(in_channels=513, out_channels=512, kernel_size=3, padding=1),
                torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(in_features=512, out_features=1)
            ])
        ])

    def forward(self, x, step):
        x = self.fromgray_layers[len(self.fromgray_layers) - step - 1](x)
        for i in range(len(self.layers)-step - 1, len(self.layers)):
            x = self.layers[i](x)

            # Dont pool for the last layer
            x = F.avg_pool2d(x, kernel_size=2) if i != len(self.layers) - 1 else x

        return x


class ProGAN(torch.nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data = PreiswerkDataset("B", device)
    # dataloader = DataLoader(data, batch_size=8, shuffle=True)
    # batch = next(iter(dataloader))

    D = Discriminator()
    G = Generator(512)
    print(D)

    input = torch.randn((4, 1, 4, 4))
    print(input.shape)

    for step in range(7):
        print("-----")
        print(f"step {step}")
        # res = 2**(step+3)
        input = torch.randn(4, 512, 1, 1)

        o = G(input, step)
        print("Generator output: ", o.shape)

        d_o = D(o, step)
        print("Discriminator output", d_o.shape)


if __name__ == '__main__':
    main()
