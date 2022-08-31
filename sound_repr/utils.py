import torch
import librosa
import librosa.display
import numpy as np
from typing import List
from matplotlib import pyplot as plt


class Sine(torch.nn.Module):
    def __init__(self, w0: float = 1.0):
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


def plot_spectrogram_Hz(sample, sample_rate):
    sample = np.hstack(sample)
    sgram = librosa.stft(sample)
    out_sgram = librosa.amplitude_to_db(np.abs(sgram), ref=np.min)
    librosa.display.specshow(out_sgram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Częstotliwość [Hz]')
    plt.xlabel('Czas [s]')


def plot_spectrogram(sample, sample_rate):
    sample = np.hstack(sample)
    sgram = librosa.stft(sample)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram, n_mels=128)
    log_S = librosa.power_to_db(np.abs(mel_scale_sgram) ** 2, ref=np.min)
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Częstotliwość [Hz]')
    plt.xlabel('Czas [s]')


def build_network(
        input_size: int,
        output: int,
        network: List[int],
        activation,
        bias: bool,
        first_activation,
):
    module_list = [
        torch.nn.Linear(input_size, network[0], bias=bias),
        first_activation
    ]
    for module_id in range(1, len(network)):
        module_list.append(
            torch.nn.Linear(network[module_id - 1], network[module_id], bias=bias))
        module_list.append(activation)
    module_list.append(torch.nn.Linear(network[-1], output, bias=bias))
    return module_list
