import torch
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def generate_images(amount, G, y, output_shape, noise_vector_length, device):
    noise = torch.normal(0, 1, size=(amount, noise_vector_length), device=device)
    G_output = G(noise, y)

    images = []

    for i in range(amount):
        images.append(torch.reshape(G_output[i], output_shape))

    return images

