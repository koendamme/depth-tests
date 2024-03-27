# from .GAN import Discriminator, Generator
from GAN.GAN_models import Discriminator, Generator
from GAN.dataset import PreiswerkDataset
import torch
from GAN.utils import weights_init, generate_images
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb


def log_batch(us_batch, depth_batch, mri_batch, noise_vector_length, generator, device, mode):
    noise = torch.normal(0, 1, size=(us_batch.shape[0], noise_vector_length), device=device)
    fake_data = generator(noise, us_batch, depth_batch)
    fake_image_batch = torch.reshape(fake_data, (us_batch.shape[0], mri_batch.shape[1], mri_batch.shape[2]))

    real_images_to_log = [wandb.Image(mri_batch[i], caption=f"Real {mode} image") for i in
                          range(mri_batch.shape[0])]
    fake_images_to_log = [wandb.Image(fake_image_batch[i], caption=f"Fake {mode} image") for i in
                          range(fake_image_batch.shape[0])]

    wandb.log({
        f"Real {mode} images": real_images_to_log,
        f"Fake {mode} images": fake_images_to_log
    })


config = dict(
    noise_vector_length=150,
    n_epochs=100,
    lr=.0001,
    batch_size=8,
    p_dropout_G=.2,
    p_dropout_D=.2,
    digit="all",
    architecture="cGAN"
)

wandb.init(project="Preiswerk-GAN", config=config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = PreiswerkDataset("B", device=device)
train_length = int(len(dataset)*.9)
train, test = random_split(dataset, [train_length, len(dataset) - train_length])

train_dataloader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(test, batch_size=config["batch_size"], shuffle=True)

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

    for i_batch, (us_batch, depth_batch, mri_batch) in tqdm(enumerate(train_dataloader), desc=f"Epoch {i_epoch+1}: ", total=len(train)//config["batch_size"]):
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

        if i_batch == 0:
            log_batch(us_batch, depth_batch, mri_batch, config["noise_vector_length"], G, device, "train")

    for i_test_batch, (us_test_batch, depth_test_batch, mri_test_batch) in enumerate(test_dataloader):
        if i_test_batch == 0:
            log_batch(us_test_batch, depth_test_batch, mri_test_batch, config["noise_vector_length"], G, device, "test")
