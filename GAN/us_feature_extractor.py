import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=2)
        self.pool = torch.nn.AvgPool1d(kernel_size=2)
        self.dropout = torch.nn.Dropout(p=.2)

    def forward(self, input):
        x = self.conv(input)[:, :, :-2]    # Causal convolution
        x = self.pool(x)
        x = self.dropout(x)
        return x


class UsFeatureExtractor(torch.nn.Module):
    def __init__(self, input_length):
        super(UsFeatureExtractor, self).__init__()

        self.output_length = input_length//2//2//2//2
        self.model = torch.nn.Sequential(*[
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16),
            ConvBlock(16, 1)
        ])

    def forward(self, input):
        x = self.model(input).squeeze()
        if x.dim() == 1:
            x = x[None, :]
        return x
