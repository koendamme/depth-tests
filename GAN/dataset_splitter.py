from GAN.dataset import CustomDataset
import torch
from torch.utils.data import Subset, ConcatDataset


class DatasetSplitter:
    def __init__(self, dataset, train_fraction, val_fraction, test_fraction):
        self.dataset = dataset
        self.splits = self.dataset.splits
        self.patterns = list(self.splits.keys())

        self.train_subsets = {}
        self.val_subsets = {}
        self.test_subsets = {}
        for p in self.patterns:
            start_idx = self.splits[p]["start"]
            end_idx = self.splits[p]["end"]
            idxs = torch.arange(start_idx, end_idx + 1)

            train_split = int(len(idxs) * train_fraction)
            train_idxs = idxs[:train_split]

            val_split = int(len(idxs) * (train_fraction + val_fraction))
            val_idxs = idxs[train_split:val_split]

            test_idxs = idxs[val_split:]

            self.train_subsets[p] = Subset(dataset, train_idxs)
            self.val_subsets[p] = Subset(dataset, val_idxs)
            self.test_subsets[p] = Subset(dataset, test_idxs)

    def get_train_dataset(self):
        return ConcatDataset(self.train_subsets.values())


if __name__ == '__main__':
    mri_freq = 2.9
    surrogate_freq = 50

    signals_between_mrs = int(surrogate_freq//mri_freq)

    dataset = CustomDataset(r"C:\data", "A", (500, 1000), signals_between_mrs)

    splitter = DatasetSplitter(dataset, train_fraction=.8, val_fraction=.1, test_fraction=.1)
    train = splitter.get_train_dataset()
    print()