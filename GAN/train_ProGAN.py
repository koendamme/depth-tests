import torch
from GAN.dataset import PreiswerkDataset
from torch.utils.data import DataLoader, random_split
from ProGAN import Generator, Discriminator
from tqdm import tqdm
import math


def get_alpha(curr_epoch, epochs_per_step, quickness):
    alpha = quickness*(curr_epoch % epochs_per_step)/epochs_per_step

    return alpha if alpha <= 1 else 1


def get_step(n_epochs, total_steps, curr_epoch):
    return int((total_steps-1)/(n_epochs-1) * curr_epoch)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = PreiswerkDataset("B", device=device)
    train_length = int(len(dataset) * .9)
    train, test = random_split(dataset, [train_length, len(dataset) - train_length])
    batch_size = 8
    noise_vector_length = 256
    depth_feature_length = dataset[0][1].shape[0]

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    D = Discriminator(depth_feature_length).to(device)
    G = Generator(depth_feature_length, noise_vector_length).to(device)

    gen_optimizer = torch.optim.Adam(G.parameters(), betas=(0, 0.99), lr=0.001, eps=1e-8)
    discr_optimizer = torch.optim.Adam(D.parameters(), betas=(0, 0.99), lr=0.001, eps=1e-8)

    n_epochs = 100
    desired_resolution = 256
    total_steps = 1 + math.log2(desired_resolution/4)
    epochs_per_step = n_epochs // total_steps
    for i in range(n_epochs):
        # curr_step = curr_step + 1 if i % epochs_per_step == 0 else curr_step
        curr_step = get_step(n_epochs, total_steps, i)
        curr_alpha = get_alpha(i, epochs_per_step, quickness=2)

        for i_batch, (us_batch, depth_batch, mri_batch) in tqdm(enumerate(train_dataloader),
                                                                desc=f"Epoch {i + 1}, step {curr_step}, alpha {round(curr_alpha, 2)}: ",
                                                                total=len(train) // batch_size):

            D.zero_grad()
            noise_batch = torch.randn(us_batch.shape[0], noise_vector_length, 1, 1, device=device)
            fake = G(noise_batch, us_batch, depth_batch, curr_step, curr_alpha)
            d_fake = D(fake, us_batch, depth_batch, curr_step, curr_alpha)

            real_input = torch.nn.functional.adaptive_avg_pool2d(mri_batch, (4*2**curr_step, 4*2**curr_step))
            d_real = D(real_input[:, None], us_batch, depth_batch, curr_step, curr_alpha)
            d_loss = torch.mean(d_real) - torch.mean(d_fake)
            d_loss.backward()
            discr_optimizer.step()

            G.zero_grad()
            g_fake = G(noise_batch, us_batch, depth_batch, curr_step, curr_alpha)
            g_loss = -torch.mean(g_fake)
            g_loss.backward()
            gen_optimizer.step()


if __name__ == '__main__':
    main()
