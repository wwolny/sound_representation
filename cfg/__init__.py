from dataclasses import dataclass, fields, field
from typing import List

@dataclass
class BaseConfig:
    @classmethod
    def from_dict(cls, obj: dict):
        fields_names = [fld.name for fld in fields(cls)]
        dct = {k: v for (k, v) in obj.items() if k in fields_names}
        return cls(**dct)


@dataclass
class MainConfig(BaseConfig):
    # Settings
    wandb: bool = True
    seed: int = 0

    # Dataset
    name: str = "LJSPEECH"
    frame_len: float = 2.0
    SR: int = 22050
    batched: bool = False
    batch_size: int = 64
    samples: int = 5
    t_scale_max: int = 300
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128

    # Train
    mode: str = "default"
    network: List[int] = field(default_factory=lambda: [128])
    epochs: int = 101
    bias: bool = True
    lr: float = 0.001
    optimizer: str = "Adam"
    loss_fn: str = "MSELoss"
    loss_lst: List[str] = field(default_factory=lambda: ["L1Loss", "HuberLoss", "SmoothL1Loss"])
    SNR: bool = True
    LSD: bool = False

    # NERF
    L: int = 32

    # SIREN
    omega: int = 30
    hidden_omega: int = 30
    siren_bias_init: bool = True

    # Output
    log_img: int = 25
    plot: bool = True
