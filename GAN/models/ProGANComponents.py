import torch


class MiniBatchStd(torch.nn.Module):
    def __init__(self):
        super(MiniBatchStd, self).__init__()

    def forward(self, x):
        std = torch.std(x, dim=1)
        mu = std.mean()
        rep = mu.repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        # minibatch_std = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        return torch.cat([x, rep], dim=1)


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
        self.scale = (2 / (kernel_size**2 * in_channels))**.5
        self.bias = self.conv.bias
        self.conv.bias = None

        torch.nn.init.normal_(self.conv.weight, mean=0, std=1)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, apply_pixelnorm=False):
        super(ConvBlock, self).__init__()
        self.apply_pixelnorm = apply_pixelnorm
        self.conv1 = WeightedConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.conv2 = WeightedConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pixelnorm = PixelWiseNormalization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.pixelnorm(x) if self.apply_pixelnorm else x

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.pixelnorm(x) if self.apply_pixelnorm else x

        return x