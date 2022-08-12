from torch.utils.data.dataset import Dataset
import torch
from scipy.io import wavfile
from os import listdir
from os.path import isfile, join


class Samples(Dataset):
    def __init__(self, dataset, SR, frame_len):
        self._dataset = dataset
        self.frame_length = frame_len
        self.SR = SR

    def __getitem__(self, idx):
        raw, sr, label, _ = self._dataset[idx]
        raw = raw[0]
        assert sr == self.SR

        n_samples = int(self.frame_length * sr)

        if len(raw) >= n_samples:
            return raw[:n_samples]
        else:
            signal = torch.zeros(n_samples)
            signal[:len(raw)] = raw
            return signal

    def __len__(self):
        return len(self._dataset)


class CustomDataset(Dataset):
    def __init__(self, directory):
        self.items = []
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        for file in files:
            samplerate, data = wavfile.read(join(directory, file))
            self.items.append(([data], samplerate, "", ""))

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)
