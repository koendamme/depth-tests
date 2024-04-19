from GAN.dataset import PreiswerkDataset
import torch
from torch.utils.data import Subset, DataLoader
from GAN.ProGAN import ConditionalProGAN
import cv2
from GAN.utils import scale_generator_output
import numpy as np
import time

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patient = "A"
    dataset = PreiswerkDataset(patient, device=device)

    cProGAN = ConditionalProGAN(
        noise_vector_length=256,
        device=device,
        desired_resolution=256,
        G_lr=0.001,
        D_lr=0.001,
        us_signal_length=dataset.us.shape[2],
        us_channels=dataset.us.shape[1]
    )
    cProGAN.load_state_dict(torch.load("models/earthy-butterfly-43_epoch949.pt", map_location=device))
    cProGAN.eval()

    train_length = int(len(dataset) * .9)
    test_subset = Subset(dataset, torch.arange(train_length, len(dataset)))

    # for _, mr in test_subset:
    #     mr = scale_generator_output(mr)
    #     cv2.imshow("asdf", mr.detach().numpy().squeeze())
    #     if cv2.waitKey(1000) == ord('q'):
    #         break

    all_us = None
    for us, _ in test_subset:
        if all_us is None:
            all_us = us
        else:
            all_us = torch.concatenate([all_us, us], dim=0)


    for i in range(all_us.shape[0]//5):
        # print(i)
        us_batch = all_us[None, i:i+64, :]
        noise_batch = torch.randn((1, cProGAN.noise_vector_length, 1, 1), device=device)
        # print(noise_batch[0, 0, 0, 0])
        # print(us_batch[0, 0, 0])
        output = cProGAN.G(noise_batch, us_batch, 6, 1)
        output = scale_generator_output(output.squeeze().detach().numpy())
        print(output[120, 120])

        cv2.imshow("frame", output)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()