from torch.utils.data.dataset import Dataset
import torch


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
