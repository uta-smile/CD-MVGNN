import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VAEDataset(Dataset):

    def __init__(self, data, config):
        self.data = data
        self.col_names = [config.csv_col_names[0]]
        assert len(self.col_names) == 1, "VAE model only needs one SMILE column as input."
        self.lbl_cols = config.design_col_names.copy()
        self.norm = config.auto_norm
        self.max = 1.0
        self.min = 0.0
        if self.data.shape[1] == 1:
            self.has_lbl = False
            assert len(self.lbl_cols) == 0
        else:
            self.has_lbl = True
            if self.norm and self.has_lbl:
                self.max = self.data[self.lbl_cols].std()
                self.min = self.data[self.lbl_cols].mean()

    def update_norm(self, min, max):
        if isinstance(self.min, float):
            self.min = min
        else:
            self.min = pd.Series(min, index=self.min.index)
        if isinstance(self.max, float):
            self.max = max
        else:
            self.max = pd.Series(max, index=self.max.index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.has_lbl:
            x = self.data.iloc[index][self.col_names]
            y = self.data.iloc[index][self.lbl_cols]
            if self.norm and self.has_lbl:
                y = ((y - self.min) / self.max).values
        else:
            x = np.array(self.data.iloc[index])
            y = np.array([None])

        return np.hstack((x, y))


class VAEDesignDataset(Dataset):

    def __init__(self, data, sample_times=1, min_v=None, max_v=None):
        self.data = data
        self.sample_times = sample_times
        if min_v is not None and max_v is not None:
            self.min_v = min_v
            self.max_v = max_v
            self.norm = True
        else:
            self.norm = False

    def __len__(self):
        return self.data.shape[0] * self.sample_times

    def __getitem__(self, index):
        if self.norm:
            return (self.data[int(np.floor(index / self.sample_times))] - self.min_v) * 1.0 \
                   / self.max_v

        return self.data[int(np.floor(index / self.sample_times))]
