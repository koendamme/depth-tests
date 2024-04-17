import torch


def normalized_mean_squared_error(fake_batch, real_batch):
    nom = torch.linalg.matrix_norm(fake_batch - real_batch) ** 2
    denom = torch.linalg.matrix_norm(real_batch) ** 2
    return nom/denom
