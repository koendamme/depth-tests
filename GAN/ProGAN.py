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
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fromgray_layers = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1),
            # torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1),
            torch.nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1)
        ])

        self.layers = torch.nn.ModuleList([
            ConvBlock(16, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            # ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        ])

        self.final_block = torch.nn.Sequential(*[
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4)
        ])

    def forward(self, x, step):
        x = self.fromgray_layers[len(self.fromgray_layers) - step - 1](x)

        for i in range(len(self.layers)-step - 1, len(self.layers)):
            x = self.layers[i](x)
            x = F.avg_pool2d(x, kernel_size=2)

        # TODO batch std
        x = self.final_block(x)

        return x



        # x = self.fromgray_layers[len(self.fromgray_layers) - step - 1](x)
        #
        # for i in range(len(self.layers)-step, len(self.layers)):
        #     x = self.layers[i](x)
        #     x = F.avg_pool2d(x, kernel_size=2)
        #
        # return x




class ProGAN(torch.nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data = PreiswerkDataset("B", device)
    # dataloader = DataLoader(data, batch_size=8, shuffle=True)
    # batch = next(iter(dataloader))

    D = Discriminator()
    # print(D)

    # input = torch.randn((4, 1, 4, 4))
    # print(input.shape)

    for step in range(6):
        print("-----")
        print(f"step {step}")
        res = 2**(step+3)
        print(res)
        input = torch.randn(4, 1, res, res)
        o = D(input, step)
        print(o.shape)


if __name__ == '__main__':
    main()
