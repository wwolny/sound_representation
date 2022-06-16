import torch
import librosa
import numpy as np


class Sine(torch.nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.sin(self.w0 * x)


def get_power(x):
    return np.log(np.abs(librosa.stft(x, 2048)) ** 2 + 1e-8)


def LSD(preds, targets):
    preds = preds.detach().numpy()
    targets = targets.detach().numpy()
    S1 = get_power(targets)
    S2 = get_power(preds)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=1)), axis=0)
    return torch.tensor([min(lsd, 10.)])
