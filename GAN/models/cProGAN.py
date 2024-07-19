import torch
import torch.nn.functional as F
from GAN.models.us_feature_extractor import UsFeatureExtractor
from tqdm import tqdm
import math
import numpy as np
from metrics import normalized_mean_squared_error, ssim
from models.ProGANComponents import WeightedConv2d, PixelWiseNormalization, ConvBlock, MiniBatchStd


class Generator(torch.nn.Module):
    def __init__(self, noise_length, concatenated_surrogate_length, layers):
        super(Generator, self).__init__()
        self.surrogate_processor = torch.nn.Sequential(*[
            torch.nn.Linear(in_features=concatenated_surrogate_length, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.ReLU()
            # torch.nn.BatchNorm1d(32)
        ])

        self.initial_layer = torch.nn.Sequential(*[
            WeightedConv2d(in_channels=noise_length + 32, out_channels=512, kernel_size=1),
            torch.nn.LeakyReLU(0.2),
            PixelWiseNormalization()
        ])

        self.togray_layers = torch.nn.ModuleList([
            WeightedConv2d(in_channels=layers[0], out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=layers[1], out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=layers[2], out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=layers[3], out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=layers[4], out_channels=1, kernel_size=1),
            WeightedConv2d(in_channels=layers[5], out_channels=1, kernel_size=1),
        ])

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.ConvTranspose2d(in_channels=512, out_channels=layers[0], kernel_size=6, stride=1, padding=0),
                torch.nn.LeakyReLU(0.2),
                PixelWiseNormalization(),
                WeightedConv2d(in_channels=layers[0], out_channels=layers[0], kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.2),
                PixelWiseNormalization(),
            ]),
            ConvBlock(layers[0], layers[1], apply_pixelnorm=True),
            ConvBlock(layers[1], layers[2], apply_pixelnorm=True),
            ConvBlock(layers[2], layers[3], apply_pixelnorm=True),
            ConvBlock(layers[3], layers[4], apply_pixelnorm=True),
            ConvBlock(layers[4], layers[5], apply_pixelnorm=True)
        ])

    def forward(self, x, mr_wave, us_wave, coil_wave, heat_wave, step, alpha):
        # Extract features from surrogates and concat with noise
        to_concat = [s for s in [mr_wave, us_wave, coil_wave, heat_wave] if s is not None]
        concatenated_surrogates = torch.concatenate(to_concat, dim=1)

        surr_features = self.surrogate_processor(concatenated_surrogates)
        latent_vector = torch.concatenate([surr_features[:, :, None, None], x], dim=1)

        x = self.initial_layer(latent_vector)

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
    def __init__(self, concatenated_surrogate_length, layers):
        super(Discriminator, self).__init__()

        self.fromgray_layers = torch.nn.ModuleList([
            WeightedConv2d(in_channels=1, out_channels=layers[0], kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=layers[1], kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=layers[2], kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=layers[3], kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=layers[4], kernel_size=1),
            WeightedConv2d(in_channels=1, out_channels=layers[5], kernel_size=1)
        ])
        self.act = torch.nn.LeakyReLU(0.2)

        self.layers = torch.nn.ModuleList([
            ConvBlock(layers[0], layers[1]),
            ConvBlock(layers[1], layers[2]),
            ConvBlock(layers[2], layers[3]),
            ConvBlock(layers[3], layers[4]),
            ConvBlock(layers[4], layers[5]),
            torch.nn.Sequential(*[
                MiniBatchStd(),
                WeightedConv2d(in_channels=layers[5]+1, out_channels=layers[5], kernel_size=3, padding=1),
                torch.nn.LeakyReLU(0.2),
                WeightedConv2d(in_channels=layers[5], out_channels=layers[5], kernel_size=6, padding=0),
                torch.nn.LeakyReLU(0.2)
            ])
        ])
        self.final_combination_block = torch.nn.Sequential(*[
            torch.nn.Linear(in_features=layers[5] + concatenated_surrogate_length, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=1)
        ])

    def forward(self, input, mr_wave, us_wave, coil_wave, heat_wave, step, alpha):
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

        to_concat = [s for s in [mr_wave, us_wave, coil_wave, heat_wave, x.squeeze()] if s is not None]
        x = torch.concatenate(to_concat, dim=1)
        x = self.final_combination_block(x)

        return x


class ConditionalProGAN(torch.nn.Module):
    def __init__(self,
                 noise_vector_length,
                 device,
                 desired_resolution,
                 G_lr,
                 D_lr,
                 n_critic,
                 n_epochs,
                 total_surrogate_length,
                 D_layers,
                 G_layers):
        super(ConditionalProGAN, self).__init__()
        self.desired_resolution = desired_resolution
        self.total_steps = 1 + math.log2(desired_resolution / 6)
        if n_epochs % self.total_steps != 0:
            raise Exception("Total number of epochs should be divisible by the total number of steps")
        self.n_epochs = n_epochs
        self.noise_vector_length = noise_vector_length
        self.device = device
        self.D = Discriminator(total_surrogate_length, D_layers).to(device)
        self.G = Generator(noise_vector_length, total_surrogate_length, G_layers).to(device)
        self.n_critic = n_critic
        self.curr_step = 0
        self.curr_alpha = 0
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), betas=(0, 0.99), lr=G_lr, eps=1e-8)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), betas=(0, 0.99), lr=D_lr, eps=1e-8)

        self.G_scheduler = lr_scheduler.LinearLR(self.G_optimizer, start_factor=1, end_factor=.01, total_iters=n_epochs)
        self.D_scheduler = lr_scheduler.LinearLR(self.D_optimizer, start_factor=1, end_factor=.01, total_iters=n_epochs)

    def train_single_epoch(self, dataloader, current_epoch, gp_lambda, save_checkoint_path=None):
        self.D.train()
        self.G.train()
        self.curr_step = self._get_step_linear(self.n_epochs, self.total_steps, current_epoch)
        self.curr_alpha = self._get_alpha_linear(current_epoch, self.n_epochs // self.total_steps, quickness=2)

        # self.curr_step = self._get_step_root(self.n_epochs, self.total_steps, current_epoch)
        # self.curr_alpha = self._get_alpha_root(self.n_epochs, self.total_steps, current_epoch, self.curr_step, quickness=2)

        running_D_loss, running_G_loss = 0, 0
        for i_batch, data in tqdm(enumerate(dataloader), desc=f"Epoch {current_epoch + 1}, step {self.curr_step}, alpha {round(self.curr_alpha, 2)}: ", total=len(dataloader)):
            mri_batch = data["mr"]
            wave_batch = None
            us_batch = data["us_wave"]
            coil_batch = data["coil"]
            heat_batch = data["heat"]
            mri_batch, heat_batch = mri_batch.to(self.device), heat_batch.to(self.device)
            us_batch, coil_batch = us_batch.to(self.device), coil_batch.to(self.device)

            for t in range(self.n_critic):
                noise_batch = torch.randn(mri_batch.shape[0], self.noise_vector_length, 1, 1, device=self.device)
                fake = self.G(noise_batch, wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)
                real_input = torch.nn.functional.adaptive_avg_pool2d(mri_batch, (6 * 2 ** self.curr_step, 6 * 2 ** self.curr_step))
                d_fake = self.D(fake.detach(), wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)
                d_real = self.D(real_input[:, None], wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)

                gp = self.compute_gradient_penalty(real_input[:, None], fake, wave_batch, us_batch, coil_batch, heat_batch)
                d_loss = (
                        -(torch.mean(d_real) - torch.mean(d_fake))
                        + gp_lambda * gp
                        + (0.001 * torch.mean(d_real ** 2))
                )

                self.D_optimizer.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()

            noise_batch = torch.randn(mri_batch.shape[0], self.noise_vector_length, 1, 1, device=self.device)
            fake_mr = self.G(noise_batch, wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)
            g_fake = self.D(fake_mr, wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)
            g_loss = -torch.mean(g_fake)

            self.G_optimizer.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()

            running_D_loss += d_loss.cpu().item()
            running_G_loss += g_loss.cpu().item()

        self.G_scheduler.step()
        self.D_scheduler.step()

        return running_D_loss, running_G_loss

    def evaluate(self, dataloader):
        self.D.eval()
        self.G.eval()

        fake_to_return = []
        real_to_return = []
        all_nmse = []
        all_ssim = []
        for data in dataloader:
            mr_batch = data["mr"]
            wave_batch = None
            us_batch = data["us_wave"]
            coil_batch = data["coil"]
            heat_batch = data["heat"]
            mr_batch, heat_batch = mr_batch.to(self.device), heat_batch.to(self.device)
            us_batch, coil_batch = us_batch.to(self.device), coil_batch.to(self.device)

            noise_batch = torch.randn(mr_batch.shape[0], self.noise_vector_length, 1, 1, device=self.device)
            fake = self.G(noise_batch, wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)
            fake_upscaled = F.interpolate(fake, scale_factor=2**(self.total_steps - self.curr_step - 1), mode='nearest')
            nmse = normalized_mean_squared_error(fake_upscaled, mr_batch).detach().cpu().numpy().flatten()
            curr_ssim = ssim(fake_upscaled, mr_batch[:, None, :, :], self.device).detach().cpu().numpy().flatten()
            all_nmse.extend(nmse)
            all_ssim.extend(curr_ssim)
            fake_to_return.extend(fake_upscaled.detach().cpu().numpy())
            real_to_return.extend(mr_batch.detach().cpu().numpy())

        return np.array(fake_to_return), np.array(real_to_return), np.array(all_nmse), np.array(all_ssim)

    def compute_gradient_penalty(self, real, fake, wave_batch, us_batch, coil_batch, heat_batch):
        epsilon = torch.rand((real.shape[0], 1, 1, 1), device=self.device)
        x_hat = (epsilon * real + (1-epsilon)*fake.detach()).requires_grad_(True)

        score = self.D(x_hat, wave_batch, us_batch, coil_batch, heat_batch, self.curr_step, self.curr_alpha)
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
    def _get_alpha_linear(curr_epoch, epochs_per_step, quickness):
        alpha = quickness * (curr_epoch % epochs_per_step) / epochs_per_step

        return min(alpha, 1)

    @staticmethod
    def _get_step_linear(n_epochs, total_steps, curr_epoch):
        epochs_per_step = n_epochs // total_steps
        step = int(curr_epoch / (epochs_per_step))

        return min(step, int(total_steps - 1))

    @staticmethod
    def _get_step_root(n_epochs, n_steps, curr_epoch):
        return math.floor(n_steps / math.sqrt(n_epochs) * math.sqrt(curr_epoch))

    def _get_milestones(self, n_epochs, n_steps):
        milestones = []
        for i in range(1, int(n_steps)):
            milestone = math.ceil(n_epochs * i ** 2 / n_steps ** 2)
            milestones.append(milestone)

        return milestones

    @staticmethod
    def _get_alpha_root(n_epochs, n_steps, curr_epoch, curr_step, quickness=2):
        # curr_step = _get_step_root(n_epochs, n_steps, curr_epoch)

        start_step = math.ceil(n_epochs * curr_step ** 2 / n_steps ** 2)
        end_step = math.ceil(n_epochs * (curr_step + 1) ** 2 / n_steps ** 2)

        dx = (end_step - start_step) / quickness
        dy = 1

        alpha = dy / dx * (curr_epoch - start_step)

        return min(alpha, 1)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    main()
