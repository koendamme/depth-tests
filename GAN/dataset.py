import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
from GAN.utils import min_max_scale_tensor, normalize_tensor, scale_input, scale_generator_output
import matplotlib.pyplot as plt
import glob
from bs4 import BeautifulSoup
import json
import pydicom
from datetime import datetime
from torchvision import transforms
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PreiswerkDataset(Dataset):
    def __init__(self, patient, device, global_scaling):
        path = {
            "A": os.path.join("..", "Preiswerk", "datasets", "A", "1"),
            "B": os.path.join("..", "Preiswerk", "datasets", "B", "2"),
            "H": os.path.join("..", "Preiswerk", "datasets", "H", "2")
        }

        with h5py.File(os.path.join(path[patient], "mr_data.h5"), "r") as f:
            self.mri = torch.tensor(np.array(f['mr_data']['I'][:, 1]), dtype=torch.float32)[1:]

        with h5py.File(os.path.join(path[patient], "mr2us.h5"), "r") as f:
            mr2us = np.array(f['mr2us']['plane1'])

        with h5py.File(os.path.join(path[patient], "us_data.h5"), "r") as f:
            all_us = np.array(f['us_data']).squeeze()

        with open(os.path.join("..", "Preiswerk", "datasets", patient, "config.xml"), 'r') as f:
            config = f.read()
            config_file = BeautifulSoup(config, "xml")
            us_roi = config_file.find("parameters").find("parameter", {"name": "us_roi"})
            print(us_roi["start"], us_roi["end"])

        grouped = []
        prev_idx = mr2us[0, 0]
        for i in range(1, mr2us.shape[0]):
            curr_idx = mr2us[i, 0]
            grouped.append(all_us[prev_idx:curr_idx, int(us_roi["start"]):int(us_roi["end"])])
            prev_idx = curr_idx

        self.us = torch.tensor(np.array(grouped), dtype=torch.float32)

        if global_scaling:
            self.mri = scale_input(self.mri, -1, 1)
        else:
            for i in range(self.mri.shape[0]):
                self.mri[i] = scale_input(self.mri[i], -1, 1)
        self.us, _, _ = normalize_tensor(self.us)

    def __len__(self):
        return self.mri.shape[0]

    def __getitem__(self, idx):
        return self.us[idx], self.mri[idx]


class VeenstraDataset(Dataset):
    def __init__(self):
        with open(os.path.join("..", "Veenstra", "new_A", "mri.json")) as f:
            self.mri = torch.tensor(json.load(f)["mri_data"], dtype=torch.float32)

        with open(os.path.join("..", "Veenstra", "new_A", "us.json")) as f:
            self.us = json.load(f)

        self.resize = transforms.Resize(256)

    def visualize(self):
        for img in self.mri["mri_data"]:
            img = np.array(img)
            max, min = np.max(img), np.min(img)
            img = (img - min)/(max-min)
            cv2.imshow("Frame", img)
            cv2.waitKey(100)

        cv2.destroyAllWindows()

    def __getitem__(self, idx):
        scaled = scale_input(self.mri[idx], -1, 1)
        resized = self.resize(scaled.unsqueeze(0))
        return resized

    def __len__(self):
        return self.mri.shape[0]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = PreiswerkDataset("A", device, False)

    print(torch.min(dataset[10][1]), torch.max(dataset[10][1]))
    mr_timestamps = [datetime.combine(datetime.today(), datetime.utcfromtimestamp(float(mr_time)).time()) for mr_time in dataset.mri["times"]]
    us_timestamps = [datetime.combine(datetime.today(), datetime.strptime(us_time, "%H:%M:%S").time()) for us_time in dataset.us["times"]]
    #
    # start_us, start_mr = us_timestamps[0], mr_timestamps[0]
    # differences = [(mr_timestamps[i + 1] - mr_timestamps[i]).total_seconds() for i in range(len(mr_timestamps) - 1)]
    #
    # us_aligned = [us_time - diff for us_time in us_timestamps]