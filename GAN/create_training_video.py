import wandb
import os
import time

api = wandb.Api()

run = api.run("thesis-koen/Preiswerk-GAN/jztj7zpc")

timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

os.mkdir(timestr)

i = 0
for file in run.files():
    if (file.name.endswith(".png") and
            file.name.split("/")[-1].split("_")[0] == "Test images" and
            int(file.name.split("/")[-1].split("_")[1]) % 10 == 0):

        file.name = file.name.replace(".png", "_" + str(i) + ".png")
        file.download(timestr, exist_ok=True)
        i += 1

