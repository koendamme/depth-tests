import torch


def normalized_mean_squared_error(fake_batch, real_batch):
    nom = torch.linalg.matrix_norm(fake_batch - real_batch) ** 2
    denom = torch.linalg.matrix_norm(real_batch) ** 2
    return nom/denom


def gaussian_window(size, sigma=1.5):
    coords = torch.arange(start=-(size-1)/2, end=(size-1)/2+1, step=1)
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return (gaussian / gaussian.sum())[None, None]


def ssim(fake_batch, real_batch, pixel_range=2):
    k_1, k_2 = .01, .02
    C_1 = (k_1*pixel_range)**2
    C_2 = (k_2*pixel_range)**2
    x = fake_batch
    y = real_batch

    w = gaussian_window(size=11, sigma=1.5)

    mu_x = torch.nn.functional.conv2d(input=x, weight=w)
    mu_y = torch.nn.functional.conv2d(input=y, weight=w)

    sigma_x_sq = torch.nn.functional.conv2d(input=x*x, weight=w) - mu_x**2
    sigma_y_sq = torch.nn.functional.conv2d(input=y*y, weight=w) - mu_y**2

    sigma_xy = torch.nn.functional.conv2d(input=x*y, weight=w) - mu_x*mu_y

    num = (2*mu_x*mu_y + C_1)*(2*sigma_xy + C_2)
    denom = (mu_x**2 + mu_y**2 + C_1)*(sigma_x_sq + sigma_y_sq + C_2)

    return torch.mean(num/denom, dim=(1, 2, 3))


def main():
    fake_batch = torch.randn((8, 1, 256, 256))
    real_batch = torch.randn((8, 1, 256, 256))
    a = torch.concatenate([fake_batch, fake_batch], dim=0)
    b = torch.concatenate([fake_batch, real_batch], dim=0)

    score = ssim(a, b)
    print(score)




if __name__ == '__main__':
    main()
