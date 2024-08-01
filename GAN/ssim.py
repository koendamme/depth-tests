import glob
import cv2
from skimage.metrics import structural_similarity
import numpy as np
from metrics import ssim
import torch


def own_ssim(real, fake):
    mu_x = real.mean()
    mu_y = fake.mean()
    sigma_x_sq = real.var()
    sigma_y_sq = fake.var()

    sigma_xy = ((real - real.mean())*(fake - fake.mean())).mean()

    c_1 = (0.01*255)**2
    c_2 = (0.03*255)**2

    o = ((2*mu_x*mu_y + c_1)*(2*sigma_xy + c_2))/((mu_x**2 + mu_y**2 + c_1)*(sigma_x_sq + sigma_y_sq + c_2))

    return o


def main():
    path = "F:\\results\\combined_model\\A1\\Regular Breathing\\*.png"
    paths = glob.glob(path)
    paths.sort(key=lambda p: int(p.split("\\")[-1].replace(".png", "")))

    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fake = img[:, :img.shape[1]//2]
        real = img[:, img.shape[1]//2:]


        y = structural_similarity(real, fake, data_range=256, gaussian_weights=True)
        y2 = own_ssim(real, fake)
        # y3 = ssim(torch.tensor([fake], dtype=torch.uint8), torch.tensor([real], dtype=torch.uint8), device="cpu", pixel_range=255)
        print(y, y2)

        # cv2.imshow("Frame1", fake)
        # cv2.imshow("Frame2", real)
        # cv2.waitKey(0)


if __name__ == "__main__":
    main()