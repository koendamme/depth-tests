import torch
from GAN.dataset import VeenstraDataset
from GAN.utils import denormalize_tensor, scale_generator_output
from torch.utils.data import DataLoader, Subset
from GAN.models.ProGAN import VanillaProGAN
from tqdm import tqdm
from datetime import datetime
import time
import math
import wandb
from torchvision.utils import save_image
import os


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = dict(
        batch_size=16,
        n_epochs=100,
        desired_resolution=256,
        noise_vector_length=512,
        G_learning_rate=0.001,
        D_learning_rate=0.001,
        GP_lambda=10,
        n_critic=2,
        patient="A",
        surrogates="US"
    )

    wandb.init(project="Veenstra-ProGAN", config=config)

    dataset = VeenstraDataset()
    train_length = int(len(dataset) * .9)

    train_subset = Subset(dataset, torch.arange(0, train_length))
    test_subset = Subset(dataset, torch.arange(train_length, len(dataset)))
    train_dataloader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=len(test_subset), shuffle=False)

    model = VanillaProGAN(
        noise_vector_length=config["noise_vector_length"],
        device=device,
        desired_resolution=config["desired_resolution"],
        G_lr=config["G_learning_rate"],
        D_lr=config["D_learning_rate"],
        n_critic=config["n_critic"]
    )

    datestring = datetime.now().strftime("%d-%m-%H%M")
    os.mkdir(os.path.join("train_video", f"{wandb.run.name}_{datestring}"))

    for i in range(config["n_epochs"]):
        start_time = time.time()
        D_loss, G_loss = model.train_single_epoch(train_dataloader, config["n_epochs"], i, gp_lambda=config["GP_lambda"])
        end_time = time.time()

        test_imgs, real_imgs, nmse = model.evaluate(test_dataloader)

        wandb.log({
            "D_loss": D_loss,
            "G_loss": G_loss,
            "Test_images": [wandb.Image(scale_generator_output(test_imgs[i]), caption=f"Test Image {i}") for i in range(test_imgs.shape[0])],
            "Real_images": [wandb.Image(scale_generator_output(real_imgs[i]), caption=f"Real Image {i}") for i in range(real_imgs.shape[0])],
            "Epoch": i,
            "Training_epoch_time": end_time - start_time,
            "Test_NMSE": nmse.mean()
        })

        save_image(scale_generator_output(test_imgs[0]), os.path.join("train_video", f"{wandb.run.name}_{datestring}", f"Epoch{i}_0_fake.png"))
        save_image(scale_generator_output(real_imgs[0]), os.path.join("train_video", f"{wandb.run.name}_{datestring}", f"Epoch{i}_0_real.png"))
        save_image(scale_generator_output(test_imgs[1]), os.path.join("train_video", f"{wandb.run.name}_{datestring}", f"Epoch{i}_1_fake.png"))
        save_image(scale_generator_output(real_imgs[1]), os.path.join("train_video", f"{wandb.run.name}_{datestring}", f"Epoch{i}_1_real.png"))

        if i in [574, 581, 599]:
            file_path = f"models/{wandb.run.name}_epoch{i}.pt"
            torch.save(model.state_dict(), file_path)
            # wandb.save(file_path)


if __name__ == '__main__':
    main()
