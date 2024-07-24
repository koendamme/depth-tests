import torch
import numpy as np
from torch.utils.data import DataLoader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def generate_image(amount, G, us, depth, output_shape, noise_vector_length, device):
    noise = torch.normal(0, 1, size=(1, noise_vector_length), device=device)
    G_output = G(noise, us[None], depth[None])
    return torch.reshape(G_output[0], output_shape)


def min_max_scale_tensor(data):
    return (data - torch.min(data))/(torch.max(data) - torch.min(data))


def normalize_tensor(data):
    mu = torch.mean(data)
    sigma = torch.std(data)

    normalized = (data-mu)/sigma
    return normalized, mu, sigma

def denormalize_tensor(data, mu, sigma):
    return data*sigma + mu

def scale_generator_output(data):
    return (data+1)/2

def scale_input(data, new_min, new_max):
    return new_min + ((data - torch.min(data))*(new_max - new_min))/(torch.max(data) - torch.min(data))


def create_video(fake_imgs, real_imgs):
    frames = []
    for fake, real in zip(fake_imgs, real_imgs):
        f = torch.concatenate([scale_generator_output(fake), scale_generator_output(real)[None, :, :]],
                              dim=2).detach().cpu().numpy()
        f = np.concatenate([f, f, f], axis=0)
        f = np.uint8(f * 255)
        frames.append(f)

    return np.array(frames)


def get_mean_std(train_dataset):
    loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    data = next(iter(loader))
    
    heat = data["heat"].mean(), data["heat"].std()
    coil = data["coil"].mean(), data["coil"].std()
    us = data["us_wave"].mean(), data["us_wave"].std()

    return heat, coil, us


