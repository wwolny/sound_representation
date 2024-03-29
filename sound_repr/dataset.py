from os import listdir
from os.path import isfile, join

import torch
from scipy.io import wavfile
from torch.utils.data.dataset import Dataset


class Samples(Dataset):
    """Custom dataset for sound dataset."""

    def __init__(self, dataset: Dataset, sr: int, frame_len: int):
        self._dataset = dataset
        self.frame_length = frame_len
        self.sr = sr

    def __getitem__(self, idx: int):
        raw, sr, label, _ = self._dataset[idx]
        raw = raw[0]
        assert sr == self.sr

        n_samples = int(self.frame_length * sr)

        if len(raw) >= n_samples:
            return raw[:n_samples]
        else:
            signal = torch.zeros(n_samples)
            signal[: len(raw)] = raw
            return signal

    def __len__(self):
        return len(self._dataset)


class CustomDataset(Dataset):
    """Custom dataset class filled with sound files from directory."""

    def __init__(self, directory: str):
        self.items = []
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        for file in files:
            samplerate, data = wavfile.read(join(directory, file))
            self.items.append(([data], samplerate, "", ""))

    def __getitem__(self, idx: int):
        return self.items[idx]

    def __len__(self):
        return len(self.items)
