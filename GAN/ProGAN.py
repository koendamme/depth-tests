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


class PixelWiseNormalization(torch.nn.Module):
    def __init__(self):
        super(PixelWiseNormalization, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class WeightedConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(WeightedConv2d, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.scale = (2 / kernel_size**2 * in_channels)**.5
        self.bias = self.conv.bias
        self.conv.bias = None

        torch.nn.init.normal_(self.conv.weight, mean=0, std=1)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, apply_pixelnorm = False):
        super(ConvBlock, self).__init__()
        self.apply_pixelnorm = apply_pixelnorm
        self.conv1 = WeightedConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.conv2 = WeightedConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pixelnorm = PixelWiseNormalization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixelnorm(x) if self.apply_pixelnorm else x
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.pixelnorm(x) if self.apply_pixelnorm else x
        x = self.leaky_relu(x)

        return x


class Generator(torch.nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.togray_layers = torch.nn.ModuleList([
            WeightedConv2d(in_channels=512, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=512, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=512, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=256, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=128, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=64, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=16, out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=8, out_channels=1, kernel_size=1)
        ])

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=1, padding=0),
                PixelWiseNormalization(),
                torch.nn.LeakyReLU(0.2),
                WeightedConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.2),
                PixelWiseNormalization(),
            ]),
            ConvBlock(512, 512, apply_pixelnorm=True),
            ConvBlock(512, 512, apply_pixelnorm=True),
            ConvBlock(512, 256, apply_pixelnorm=True),
            ConvBlock(256, 128, apply_pixelnorm=True),
            ConvBlock(128, 64, apply_pixelnorm=True),
            ConvBlock(64, 16, apply_pixelnorm=True),
            ConvBlock(16, 8, apply_pixelnorm=True)
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
            WeightedConv2d(in_channels=1, out_channels=16, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=64, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=128, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=256, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=512, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=512, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=512, kernel_size=1)
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
                WeightedConv2d(in_channels=513, out_channels=512, kernel_size=3, padding=1),
                WeightedConv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(in_features=512, out_features=1)
            ])
        ])

    def forward(self, input, step, alpha):
        x = self.fromgray_layers[len(self.fromgray_layers) - step - 1](input)
        x_hat = F.avg_pool2d(input, kernel_size=2)
        for i in range(len(self.layers) - step - 1, len(self.layers)):
            x = self.layers[i](x)

            # Don't pool for the last layer
            x = F.avg_pool2d(x, kernel_size=2) if i != len(self.layers) - 1 else x

            # Fade-in in first layer except for step 0
            x = x_hat * (1-alpha) + x * alpha if i == len(self.layers) - step - 1 and step != 0 else x

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

    input = torch.randn((4, 1, 4, 4))
    print(input.shape)

    for step in range(7):
        print("-----")
        print(f"step {step}")
        # res = 2**(step+3)
        input = torch.randn(4, 512, 1, 1)

        o = G(input, step)
        print("Generator output: ", o.shape)

        d_o = D(o, step, .5)
        print("Discriminator output", d_o.shape)


if __name__ == '__main__':
    main()
