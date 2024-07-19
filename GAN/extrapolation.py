from GAN.dataset import PreiswerkDataset
import torch
from torch.utils.data import Subset, DataLoader
from models.cProGAN import ConditionalProGAN
import cv2
from GAN.utils import scale_generator_output
import numpy as np

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patient = "A"
    dataset = PreiswerkDataset(patient, device=device, global_scaling=False)

    cProGAN = ConditionalProGAN(
        noise_vector_length=128,
        device=device,
        desired_resolution=256,
        G_lr=0.001,
        D_lr=0.001,
        us_signal_length=dataset.us.shape[2],
        us_channels=dataset.us.shape[1],
        n_critic=1
    )
    cProGAN.load_state_dict(torch.load("model_checkpoints/faithful-voice-65_epoch574.pt", map_location=device))
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

    writer = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter("Real.mp4", writer, 3, (256, 256))

    for row in test_subset:
        _, mri = row
        output = scale_generator_output(mri).numpy()
        rgb_img = np.stack((output,) * 3, axis=-1)
        cv2.imshow("", rgb_img)
        cv2.waitKey(100)
        video.write(np.uint8(rgb_img * 255))

    cv2.destroyAllWindows()
    video.release()

    video = cv2.VideoWriter("Fake.mp4", writer, 3, (256, 256))
    for i in range(all_us.shape[0]//5):
        # print(i)
        us_batch = all_us[None, i:i+64, :].to(device)
        noise_batch = torch.randn((1, cProGAN.noise_vector_length, 1, 1), device=device)
        # print(noise_batch[0, 0, 0, 0])
        # print(us_batch[0, 0, 0])
        output = cProGAN.G(noise_batch, us_batch, 6, 1)
        output = scale_generator_output(output.squeeze().detach().cpu().numpy())
        rgb_img = np.stack((output,) * 3, axis=-1)
        cv2.imshow("frame", output)
        cv2.waitKey(100)
        video.write(np.uint8(rgb_img * 255))




if __name__ == '__main__':
    main()