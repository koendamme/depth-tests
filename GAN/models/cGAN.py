import torch
from GAN.dataset import PreiswerkDataset
from GAN.us_feature_extractor import UsFeatureExtractor


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

            torch.nn.Linear(first_out_features * 4, first_out_features * 8),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features*8, output_image_size[0] * output_image_size[1]),
            torch.nn.Tanh()
        ])

    def forward(self, noise, us, depth):
        us_features = self.us_feature_extractor(us)
        if us_features.dim() == 1:
            us_features = us_features[None, :]

        x = torch.hstack((noise, us_features, depth))
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self, input_image_size, depth_length, p_dropout):
        super(Discriminator, self).__init__()
        self.us_feature_extractor = UsFeatureExtractor(1000)

        first_out_features = 2048
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

            torch.nn.Linear(first_out_features // 4, first_out_features // 8),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=p_dropout),

            torch.nn.Linear(first_out_features // 8, 1),
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
