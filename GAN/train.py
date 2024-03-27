# from .GAN import Discriminator, Generator
from GAN.GAN_models import Discriminator, Generator
from GAN.dataset import PreiswerkDataset
import torch
from GAN.utils import weights_init, generate_images
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


config = dict(
    noise_vector_length=150,
    n_epochs=100,
    lr=.001,
    batch_size=8,
    p_dropout_G=.2,
    p_dropout_D=.2,
    digit="all",
    architecture="cGAN"
)

wandb.init(project="Preiswerk-GAN", config=config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = PreiswerkDataset("B", device=device)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

G = Generator(config["noise_vector_length"], dataset.depth.shape[1], dataset.mri[0].shape, p_dropout=config["p_dropout_G"]).to(device)
# G.apply(weights_init)
D = Discriminator(dataset.mri[0].shape, dataset.depth.shape[1], p_dropout=config["p_dropout_G"]).to(device)
# D.apply(weights_init)

criterion = torch.nn.BCELoss()

gen_optimizer = torch.optim.Adam(G.parameters(), lr=config["lr"])
discr_optimizer = torch.optim.Adam(D.parameters(), lr=config["lr"])

wandb.watch(G, criterion, log="all", log_freq=10)
wandb.watch(D, criterion, log="all", log_freq=10)

for i_epoch in range(config["n_epochs"]):
    running_discr_loss, running_gen_loss = 0.0, 0.0

    for i_batch, (us_batch, depth_batch, mri_batch) in tqdm(enumerate(dataloader), desc=f"Epoch {i_epoch+1}: ", total=len(dataset)//config["batch_size"]):
        # Update Discriminator
        D.zero_grad()
        # Train with real batch
        discr_real_output = D(mri_batch.flatten(start_dim=1), us_batch, depth_batch)
        loss_D_real = criterion(discr_real_output, torch.ones(us_batch.shape[0], 1, device=device))
        loss_D_real.backward()

        # Train with fake batch
        noise_batch = torch.normal(0, 1, size=(us_batch.shape[0], config["noise_vector_length"]), device=device)
        fake = G(noise_batch, us_batch, depth_batch)
        discr_fake_output = D(fake.detach(), us_batch, depth_batch)

        loss_D_fake = criterion(discr_fake_output, torch.zeros(us_batch.shape[0], 1, device=device))
        loss_D_fake.backward()
        for param in D.parameters():
            print(param.grad)

        loss_D = loss_D_real + loss_D_fake
        running_discr_loss += loss_D.item()

        discr_optimizer.step()

        # Update generator
        G.zero_grad()
        discr_output = D(fake, us_batch, depth_batch)
        loss_G = criterion(discr_output, torch.ones(us_batch.shape[0], 1, device=device))
        running_gen_loss += loss_G.item()

        loss_G.backward()
        gen_optimizer.step()

        if (i_batch + 1) % 2 == 0:
            wandb.log({
                "D_loss": loss_D,
                "G_loss": loss_G,
                "epoch": i_epoch
            })

    images = []
    for i in range(fake.shape[0]):
        images.append(torch.reshape(fake[i], dataset.mri[0].shape))
    images_to_log = [wandb.Image(image, caption="Example image") for image in images]
    wandb.log({
        f"Fake images": images_to_log
    })
