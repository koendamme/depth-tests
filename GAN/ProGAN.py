import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from GAN.dataset import PreiswerkDataset


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
    def __init__(self, in_channels, n_channels):
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

        for block in self.layers:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = block(x)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        self.layers = torch.nn.ModuleList([])
        self.fromgray_layers = torch.nn.ModuleList([])

        # self.layers.append(ConvBlock(1, 1))

        for i in range(0, len(n_channels) - 1):
            self.layers.append(ConvBlock(n_channels[i], n_channels[i+1]))
            self.fromgray_layers.append(torch.nn.Conv2d(in_channels=1, out_channels=n_channels[i], kernel_size=1))

        self.fromgray_layers.append(torch.nn.Conv2d(in_channels=1, out_channels=n_channels[-1], kernel_size=1))


        print(self.layers)
        print(self.fromgray_layers)

    def forward(self, x, step):
        print(len(self.fromgray_layers) - step - 1)
        x = self.fromgray_layers[len(self.fromgray_layers) - step - 1](x)

        for i in range(len(self.layers)-step, len(self.layers)):
            x = self.layers[i](x)
            x = F.avg_pool2d(x, kernel_size=2)

        return x




class ProGAN(torch.nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data = PreiswerkDataset("B", device)
    # dataloader = DataLoader(data, batch_size=8, shuffle=True)
    # batch = next(iter(dataloader))

    n_channels = [512, 512, 256, 128, 64, 16]
    n_channels.reverse()
    D = Discriminator(n_channels)
    # print(D)

    # input = torch.randn((4, 1, 4, 4))
    # print(input.shape)

    for step in range(7):
        print("-----")
        print(f"step {step}")
        res = 2**(step+2)
        print(res)
        input = torch.randn(4, 1, res, res)
        o = D(input, step)
        print(o.shape)


if __name__ == '__main__':
    main()
