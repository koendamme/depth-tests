import torch
from GAN.dataset import PreiswerkDataset


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
        return self.model(input).squeeze()


class Generator(torch.nn.Module):
    def __init__(self, input_length, depth_length, output_image_size, p_dropout):
        super(Generator, self).__init__()
        self.us_feature_extractor = UsFeatureExtractor(1000)
        first_out_features = 256
        self.model = torch.nn.Sequential(*[
            torch.nn.Linear(input_length + self.us_feature_extractor.output_length + depth_length, first_out_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features, first_out_features*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features*2, first_out_features*4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features*4, output_image_size[0] * output_image_size[1]),
            torch.nn.Tanh()
        ])

    def forward(self, noise, us, depth):
        us_features = self.us_feature_extractor(us)

        x = torch.hstack((noise, us_features, depth))
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self, input_image_size, depth_length, p_dropout):
        super(Discriminator, self).__init__()
        self.us_feature_extractor = UsFeatureExtractor(1000)

        first_out_features = 1024
        self.model = torch.nn.Sequential(*[
            torch.nn.Linear(input_image_size[0] * input_image_size[1] + self.us_feature_extractor.output_length + depth_length, first_out_features),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features, first_out_features // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features // 2, first_out_features // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features // 4, 1),
            torch.nn.Sigmoid()
        ])

    def forward(self, mr, us, depth):
        us_features = self.us_feature_extractor(us)
        x = torch.hstack((mr, us_features, depth))

        return self.model(x)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = PreiswerkDataset("B", device)
    extractor = UsFeatureExtractor(1000).to(device)

    o = extractor(data.us[:8])

    print(o.shape)
