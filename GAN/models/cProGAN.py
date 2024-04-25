import torch
import torch.nn.functional as F
from GAN.us_feature_extractor import UsFeatureExtractor
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from GAN.metrics import normalized_mean_squared_error


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


class Generator(torch.nn.Module):
    def __init__(self, noise_length, us_signal_length, us_channels):
        super(Generator, self).__init__()
        self.us_feature_extractor = UsFeatureExtractor(us_signal_length, us_channels)
        self.initial_layer = torch.nn.Sequential(*[
            WeightedConv2d(in_channels=self.us_feature_extractor.output_length + noise_length, out_channels=512, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            PixelWiseNormalization()
        ])

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
                torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0),
                torch.nn.LeakyReLU(0.2),
                PixelWiseNormalization(),
                WeightedConv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.2),
                PixelWiseNormalization(),
            ]),
            ConvBlock(512, 512, apply_pixelnorm=True),
            ConvBlock(512, 512, apply_pixelnorm=True),
            ConvBlock(512, 256, apply_pixelnorm=True),
            ConvBlock(256, 128, apply_pixelnorm=True),
            ConvBlock(128, 64, apply_pixelnorm=True),
            ConvBlock(64, 16, apply_pixelnorm=True)
        ])

    def forward(self, x, us, step, alpha):
        # Extract features from surrogates and concat with noise
        us_features = self.us_feature_extractor(us)[:, :, None, None]
        x = torch.concatenate([us_features, x], dim=1)

        x = self.initial_layer(x)

        for i in range(step + 1):
            # Don't upsample the first layer
            x_upscaled = F.interpolate(x, scale_factor=2, mode='nearest') if i != 0 else x

            x = self.layers[i](x_upscaled)

        final_out = self.togray_layers[step](x)

        # Fade-in except on step 0
        if step != 0:
            final_upscaled = self.togray_layers[step - 1](x_upscaled)
            o = final_out * alpha + final_upscaled * (1 - alpha)
            return torch.tanh(o)

        return torch.tanh(final_out)


class Discriminator(torch.nn.Module):
    def __init__(self, us_signal_length, us_channels):
        super(Discriminator, self).__init__()
        self.us_feature_extractor = UsFeatureExtractor(us_signal_length, us_channels)

        self.fromgray_layers = torch.nn.ModuleList([
            WeightedConv2d(in_channels=1, out_channels=16, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=64, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=128, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=256, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=512, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=512, kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=512, kernel_size=1)
        ])
        self.act = torch.nn.LeakyReLU(0.2)

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
                torch.nn.LeakyReLU(0.2),
                WeightedConv2d(in_channels=512, out_channels=512, kernel_size=4, padding=0),
                torch.nn.LeakyReLU(0.2)
            ])
        ])

        self.final_combination_block = torch.nn.Sequential(*[
            torch.nn.Linear(in_features=self.us_feature_extractor.output_length + 512, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=1)
        ])

    def forward(self, input, us, step, alpha):
        us_features = self.us_feature_extractor(us)

        x = self.fromgray_layers[len(self.fromgray_layers) - step - 1](input)
        x = self.act(x)

        for i in range(len(self.layers) - step - 1, len(self.layers)):
            x = self.layers[i](x)

            # Don't pool for the last layer
            x = F.avg_pool2d(x, kernel_size=2) if i != len(self.layers) - 1 else x

            # Fade-in in first layer except for step 0
            if i == len(self.layers) - step - 1 and step != 0:
                x_hat = F.avg_pool2d(input, kernel_size=2)
                x_hat = self.fromgray_layers[len(self.fromgray_layers) - step](x_hat)
                x = x_hat * (1 - alpha) + x * alpha

        x = torch.concatenate([x[:, :, 0, 0], us_features], dim=1)
        x = self.final_combination_block(x)

        return x


class ConditionalProGAN(torch.nn.Module):
    def __init__(self, noise_vector_length, device, desired_resolution, G_lr, D_lr, us_signal_length, us_channels, n_critic):
        super(ConditionalProGAN, self).__init__()

        self.noise_vector_length = noise_vector_length
        self.device = device
        self.desired_resolution = desired_resolution
        self.total_steps = 1 + math.log2(desired_resolution / 4)
        self.D = Discriminator(us_signal_length, us_channels).to(device)
        self.G = Generator(noise_vector_length, us_signal_length, us_channels).to(device)
        self.n_critic = n_critic
        self.curr_step = 0
        self.curr_alpha = 0

        self.G_optimizer = torch.optim.Adam(self.G.parameters(), betas=(0, 0.99), lr=G_lr, eps=1e-8)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), betas=(0, 0.99), lr=D_lr, eps=1e-8)

    def train_single_epoch(self, dataloader, total_epochs, current_epoch, gp_lambda):
        self.D.train()
        self.G.train()
        self.curr_step = self._get_step(total_epochs, self.total_steps, current_epoch)
        self.curr_alpha = self._get_alpha(current_epoch, total_epochs // self.total_steps, quickness=2)

        running_D_loss, running_G_loss = 0, 0
        for i_batch, (us_batch, mri_batch) in tqdm(enumerate(dataloader),
                                                                desc=f"Epoch {current_epoch + 1}, step {self.curr_step}, alpha {round(self.curr_alpha, 2)}: ",
                                                                total=len(dataloader)):

            us_batch, mri_batch = us_batch.to(self.device), mri_batch.to(self.device)

            for t in range(self.n_critic):
                noise_batch = torch.randn(us_batch.shape[0], self.noise_vector_length, 1, 1, device=self.device)
                fake = self.G(noise_batch, us_batch, self.curr_step, self.curr_alpha)
                real_input = torch.nn.functional.adaptive_avg_pool2d(mri_batch, (4 * 2 ** self.curr_step, 4 * 2 ** self.curr_step))

                d_fake = self.D(fake.detach(), us_batch, self.curr_step, self.curr_alpha)
                d_real = self.D(real_input[:, None], us_batch, self.curr_step, self.curr_alpha)

                gp = self.compute_gradient_penalty(real_input[:, None], fake, us_batch)
                d_loss = (
                        -(torch.mean(d_real) - torch.mean(d_fake))
                        + gp_lambda * gp
                        + (0.001 * torch.mean(d_real ** 2))
                )

                self.D_optimizer.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()

            noise_batch = torch.randn(us_batch.shape[0], self.noise_vector_length, 1, 1, device=self.device)
            fake_mr = self.G(noise_batch, us_batch, self.curr_step, self.curr_alpha)
            g_fake = self.D(fake_mr, us_batch, self.curr_step, self.curr_alpha)
            g_loss = -torch.mean(g_fake)

            self.G_optimizer.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()

            running_D_loss += d_loss.cpu().item()
            running_G_loss += g_loss.cpu().item()

        return running_D_loss, running_G_loss

    def evaluate(self, dataloader):
        self.D.eval()
        self.G.eval()

        us_batch, mr_batch = next(iter(dataloader))
        us_batch, mr_batch = us_batch.to(self.device), mr_batch.to(self.device)
        noise_batch = torch.randn(us_batch.shape[0], self.noise_vector_length, 1, 1, device=self.device)
        fake = self.G(noise_batch, us_batch, self.curr_step, self.curr_alpha)
        # mr_downscaled = torch.nn.functional.adaptive_avg_pool2d(mr_batch, (4 * 2 ** self.curr_step, 4 * 2 ** self.curr_step))
        # nmse = normalized_mean_squared_error(fake, mr_downscaled)

        fake_upscaled = F.interpolate(fake, scale_factor=2**(self.total_steps - self.curr_step - 1), mode='nearest')
        nmse = normalized_mean_squared_error(fake_upscaled, mr_batch)

        assert fake_upscaled.shape[-1] == self.desired_resolution

        return fake_upscaled, mr_batch, nmse

    def compute_gradient_penalty(self, real, fake, us):
        epsilon = torch.rand((real.shape[0], 1, 1, 1), device=self.device)
        x_hat = (epsilon * real + (1-epsilon)*fake.detach()).requires_grad_(True)

        score = self.D(x_hat, us, self.curr_step, self.curr_alpha)
        gradient = torch.autograd.grad(
            inputs=x_hat,
            outputs=score,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm-1)**2)
        return gradient_penalty

    @staticmethod
    def _get_alpha(curr_epoch, epochs_per_step, quickness):
        alpha = quickness * (curr_epoch % epochs_per_step) / epochs_per_step

        return alpha if alpha <= 1 else 1

    @staticmethod
    def _get_step(n_epochs, total_steps, curr_epoch):
        epochs_per_step = n_epochs//total_steps
        step = int((total_steps-1)/(n_epochs - epochs_per_step - 1) * curr_epoch)

        return min(step, total_steps-1)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data = PreiswerkDataset("B", device)
    # dataloader = DataLoader(data, batch_size=8, shuffle=True)
    # batch = next(iter(dataloader))

    # model = ConditionalProGAN(512, device, 256, 0.001, 0.001, 1000, 64)
    #
    # model.load_state_dict(torch.load("model.pt"))
    # torch.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    main()
