import torch
import numpy as np


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
    print(torch.mean(normalized), torch.std(normalized))
    return normalized

