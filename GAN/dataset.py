import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
from GAN.utils import min_max_scale_tensor, normalize_tensor
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



class PreiswerkDataset(Dataset):
    def __init__(self, patient, device):
        path = {
            "A": os.path.join("..", "Preiswerk", "datasets", "A", "1"),
            "B": os.path.join("..", "Preiswerk", "datasets", "B", "2"),
            "H": os.path.join("..", "Preiswerk", "datasets", "H", "2")
        }

        with open(os.path.join(path[patient], "depth_data.npy"), "rb") as f:
            self.depth = torch.tensor(np.load(f), dtype=torch.float32, device=device)[1:]

        with h5py.File(os.path.join(path[patient], "mr_data.h5"), "r") as f:
            self.mri = torch.tensor(np.array(f['mr_data']['I'][:, 1]), dtype=torch.float32, device=device)[1:]

        with h5py.File(os.path.join(path[patient], "mr2us.h5"), "r") as f:
            mr2us = np.array(f['mr2us']['plane1'])

        with h5py.File(os.path.join(path[patient], "us_data.h5"), "r") as f:
            all_us = np.array(f['us_data']).squeeze()

        grouped = []
        prev_idx = mr2us[0, 0]
        for i in range(1, mr2us.shape[0]):
            curr_idx = mr2us[i, 0]
            grouped.append(all_us[prev_idx:curr_idx, 2000:3000])
            prev_idx = curr_idx

        self.us = torch.tensor(np.array(grouped), device=device, dtype=torch.float32)
        self.mri = normalize_tensor(self.mri)
        self.us = normalize_tensor(self.us)
        self.depth = normalize_tensor(self.depth)
        # self.mri = min_max_scale_tensor(self.mri)
        # self.us = min_max_scale_tensor(self.us)
        # self.depth = min_max_scale_tensor(self.depth)

    def __len__(self):
        return self.mri.shape[0]

    def __getitem__(self, idx):
        return self.us[idx], self.depth[idx], self.mri[idx]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = PreiswerkDataset("B", device)
    print(data.mri[:4, None].shape)
    resized = torch.nn.functional.adaptive_avg_pool2d(data.mri[:4], (32, 32))
    print(resized[:, None].shape)
    # print(len(data))
    # plt.imshow(data.us[0][:, :100].T, cmap="gray")
    # plt.xlabel("Time")
    # plt.show()
    # print(data.us[0].shape)


