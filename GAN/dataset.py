import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
from utils import normalize_tensor, scale_input
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import json
from torchvision import transforms
from feature_extractors.us.madore_wave_extraction import get_wave_from_us
import os
import cv2
import pickle
from dataset_splitter import DatasetSplitter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class PreiswerkDataset(Dataset):
    def __init__(self, patient, global_scaling):
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


class CustomDataset(Dataset):
    def __init__(self, root_path, patient, coil_normalizer=(0, 1), heat_normalizer=(0, 1), us_normalizer=(0, 1)):
        with open(os.path.join(root_path, patient, "settings.json")) as file:
            self.settings = json.load(file)
            self.TR = self.settings["MRI"]["TR"]
            mri_freq = 1/self.TR
            surrogate_freq = 50
            self.signals_between_mrs = int(surrogate_freq//mri_freq)

        with open(os.path.join(root_path, patient, "mr.pickle"), 'rb') as file:
            self.mr = pickle.load(file)["images"]
            self.mr = np.clip(self.mr, a_min=0, a_max=255).astype(np.uint8)
            self.mr = cv2.addWeighted(self.mr, 1.7, np.zeros(self.mr.shape, self.mr.dtype), 0, 0)
            self.mr = torch.from_numpy(self.mr).float()
            self.mr = self.mr * 2 / 255 - 1
            self.mr = self.mr[:, :128, 32:-32]

        with open(os.path.join(root_path, patient, "surrogates.pickle"), 'rb') as file:
            surrogates = pickle.load(file)
            # us_roi = self.settings["US"]["ROI"]["0"], self.settings["US"]["ROI"]["1"]
            # print(us_roi)
            # self.us = np.float32(surrogates["us"])
            # self.us_wave = get_wave_from_us(self.us.T, (0, 1000))
            # self.us_wave = torch.tensor(self.us_wave)
            # self.us_wave = (self.us_wave - us_normalizer[0]) / us_normalizer[1]

            self.heat = torch.tensor(np.float32(surrogates["heat"]))
            self.heat = (self.heat - heat_normalizer[0]) / heat_normalizer[1]

            self.coil = torch.tensor(np.float32(surrogates["coil"]))
            self.coil = (self.coil - coil_normalizer[0]) / coil_normalizer[1]

        with open(os.path.join(root_path, patient, "us_wave_detrended.pickle"), "rb") as file:
            self.us_wave = pickle.load(file)
            self.us_wave = torch.tensor(np.float32(self.us_wave))
            self.us_wave = (self.us_wave - us_normalizer[0]) / us_normalizer[1]

        with open(os.path.join(root_path, patient, "mr2us_new.pickle"), 'rb') as file:
            self.mr2us = pickle.load(file)["mr2us"]

        with open(os.path.join(root_path, patient, "mr_wave.pickle"), 'rb') as file:
            self.mr_wave = torch.Tensor(pickle.load(file)["mri_waveform"])
            self.mr_wave = (self.mr_wave - self.mr_wave.mean()) / self.mr_wave.std() # normalize
            # self.mr_wave = gaussian_filter1d(self.mr_wave, .8) # smoothing

        with open(os.path.join(root_path, patient, "splits.pickle"), 'rb') as file:
            self.splits = pickle.load(file)

    def set_normalizers(self, heat, us, coil):
        self.heat_normalizer = heat
        self.us_normalizer = us
        self.coil_normalizer = coil

    def visualize(self):
        for img in self.mr:
            cv2.imshow("Frame", (img.numpy() + 1)/2)
            cv2.waitKey(30)

    def __getitem__(self, idx):
        mr = self.mr[idx]
        mr2us = self.mr2us[idx]
        # us = self.us[mr2us-self.signals_between_mrs+1:mr2us+1, self.us_roi[0]:self.us_roi[1]]
        heat = self.heat[mr2us-self.signals_between_mrs+1:mr2us+1]
        coil = self.coil[mr2us-self.signals_between_mrs+1:mr2us+1]
        us_wave = self.us_wave[mr2us-self.signals_between_mrs+1:mr2us+1]
        mr_wave = self.mr_wave[idx]

        return {"mr": mr, "heat": heat, "mr_wave": mr_wave, "us_wave": us_wave, "coil": coil}

    def __len__(self):
        return self.mr.shape[0]


if __name__ == '__main__':
    root = os.path.join("C:", os.sep, "data", "Formatted_datasets")
    
    dataset = CustomDataset(root, "C1")
    # splitter = DatasetSplitter(dataset, train_fraction=1, val_fraction=0, test_fraction=0)

    # db = splitter.train_subsets["Deep Breathing"]
    
    plt.plot(dataset.mr_wave)
    # plt.ylim([-.5, .5])
    plt.show()
    



