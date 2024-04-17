import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
from GAN.utils import min_max_scale_tensor, normalize_tensor, scale_input
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import pydicom
from pydicom.tag import BaseTag
from pydicom.data import get_testdata_files
from pydicom.fileset import FileSet
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PreiswerkDataset(Dataset):
    def __init__(self, patient, device):
        path = {
            "A": os.path.join("..", "Preiswerk", "datasets", "A", "1"),
            "B": os.path.join("..", "Preiswerk", "datasets", "B", "2"),
            "H": os.path.join("..", "Preiswerk", "datasets", "H", "2")
        }

        with h5py.File(os.path.join(path[patient], "mr_data.h5"), "r") as f:
            self.mri = torch.tensor(np.array(f['mr_data']['I'][:, 1]), dtype=torch.float32, device=device)[1:]

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

        self.us = torch.tensor(np.array(grouped), device=device, dtype=torch.float32)
        self.mri = scale_input(self.mri, -1, 1)
        self.us, _, _ = normalize_tensor(self.us)


    def __len__(self):
        return self.mri.shape[0]

    def __getitem__(self, idx):
        return self.us[idx], self.mri[idx]


class VeenstraDataset(Dataset):
    def __init__(self):
        path = os.path.join("..", "A")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # with open(os.path.join("..", "Preiswerk", "datasets", "A", "config.xml"), 'r') as f:
    #     config = f.read()
    #
    # config = BeautifulSoup(config, "xml")
    # roi = config.find("parameters").find("parameter", {"name": "us_roi"})
    # print(type(roi["start"]))
    #
    data = PreiswerkDataset("H", device)
    print(data.us.shape)
    #
    # with h5py.File(os.path.join("..", "Preiswerk", "datasets", "A", "1", "mr2us.h5"), "r") as f:
    #     mr2us = np.array(f['mr2us']['plane1']).squeeze()
    #     print(mr2us[1] - mr2us[0])
    #     print(mr2us.shape)

