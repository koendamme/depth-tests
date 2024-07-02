import torch
import torchvision
from GAN.dataset import VeenstraDataset
from GAN.utils import min_max_scale_tensor


def normalized_mean_squared_error(fake_batch, real_batch):
    nom = torch.linalg.matrix_norm(fake_batch - real_batch) ** 2
    denom = torch.linalg.matrix_norm(real_batch) ** 2
    return nom/denom


def gaussian_window(size, sigma=1.5):
    coords = torch.arange(start=-(size-1)/2, end=(size-1)/2+1, step=1)
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return (gaussian / gaussian.sum())[None, None]


def inception_score(fake_images):
    inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)

    # Input must have 3 channels
    fake_images = fake_images.repeat(8, 3, 1, 1)
    # Input must be (299 x 299)
    fake_images = torch.nn.functional.interpolate(input=fake_images, size=(299, 299), mode="bilinear")

    output = inception(fake_images).logits
    p_yx = torch.nn.functional.softmax(output, dim=1)
    p_y = p_yx.mean(dim=0)

    return torch.exp((p_yx * (torch.log2(p_yx) - torch.log2(p_y))).sum(dim=1).mean())


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


def patch_is_ndc(patch):
    L_1 = patch[:, 0, :]
    L_2 = patch[:, :, -1]
    L_3 = patch[:, -1, :]
    L_4 = patch[:, :, 0]

    for edge in [L_1, L_2, L_3, L_4]:
        mu = torch.nn.functional.conv1d(input=edge, weight=torch.ones((1, 1, 6)) / 6)
        sigma_sq = torch.nn.functional.conv1d(input=edge * edge, weight=torch.ones((1, 1, 6)) / 6) - mu ** 2
        if torch.any(sigma_sq < .1):
            return True


def patch_is_nc(patch):
    n = patch.shape[-1]
    S_sur = torch.cat([patch[:, :, :n//2-1], patch[:, :, n//2+1:]], dim=2)
    S_cen = patch[:, :, n//2-1:n//2+1]

    sigma_sur = torch.std(S_sur)
    sigma_cen = torch.std(S_cen)
    sigma_blk = torch.std(patch)

    beta = abs(sigma_cen/sigma_sur - sigma_blk)/max(sigma_cen/sigma_sur, sigma_blk)

    return sigma_blk > 2*beta


def pique(image_batch, n_blocks):
    w = gaussian_window(size=7, sigma=1)

    mu = torch.nn.functional.conv2d(input=image_batch, weight=w, padding=3)
    sigma_sq = torch.nn.functional.conv2d(input=image_batch**2, weight=w, padding=3) - mu**2

    I_hat = (image_batch - mu)/(sigma_sq**.5+1)
    block_size = I_hat.shape[2] // n_blocks
    unfolded = torch.nn.functional.unfold(I_hat, kernel_size=block_size, stride=block_size)

    patches = unfolded.view(image_batch.shape[0], 1, block_size, block_size, -1).permute(0, 4, 1, 2, 3)

    pique_values = []
    for i in range(patches.shape[0]):
        curr_patches = patches[i]
        v_k = torch.var(curr_patches, dim=(1, 2, 3))

        total_Dsk = 0
        N_sa = 0
        for j in range(curr_patches.shape[0]):
            # Check whether block is spatially active
            if v_k[j] >= .1:
                N_sa += 1
                is_ndc = patch_is_ndc(curr_patches[j])
                is_nc = patch_is_nc(curr_patches[j])

                if is_ndc:
                    total_Dsk += 1 - v_k[j]
                elif is_nc:
                    total_Dsk += v_k[j]
                elif is_nc and is_ndc:
                    total_Dsk += 1
                else:
                    pass
                    # raise Exception("Either NC or NDC should be true.")

        pique = (total_Dsk + 1) / (N_sa + 1)
        pique_values.append(pique)

    return torch.tensor(pique_values)


def main():
    data = VeenstraDataset()
    mr = data[0][0]
    # plt.imshow(mr.squeeze(), cmap="gray")
    s = inception_score(mr.squeeze()[None])
    print("Score: ", s)


if __name__ == '__main__':
    main()
