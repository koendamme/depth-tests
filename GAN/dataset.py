import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import h5py


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


    def __len__(self):
        return self.mri.shape[0]

    def __getitem__(self, idx):
        return self.us[idx], self.depth[idx], self.mri[idx]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=0, std=1)
    data = PreiswerkDataset("B", device)

    print(torch.min(data.us), torch.max(data.us))
    print(torch.min(data.depth), torch.max(data.depth))
    print(torch.min(data.mri), torch.max(data.mri))

    print(data.mri[0])

    # U_0 = data.us.cpu().numpy()[data.mr2us[0][0]:data.mr2us[1][0], 500:1000]
    # U_0 = data.us.cpu().numpy()[:200, :]
    # min, max = np.min(U_0), np.max(U_0)
    #
    # scaled = (U_0 - min)/(max - min)
    # plt.title("original")
    # plt.imshow(U_0, cmap="gray")
    # plt.show()
    # print(data.us.shape)

