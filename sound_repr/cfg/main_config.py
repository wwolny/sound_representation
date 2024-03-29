from dataclasses import dataclass, field
from typing import List

from sound_repr.cfg.base_config import BaseConfig


@dataclass
class MainConfig(BaseConfig):
    """Dataclass with values to run experiments."""

    # Settings
    wandb: bool = True
    wandb_entity: str = "wwolny"
    wandb_project: str = "sound_representation"
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
    loss_lst: List[str] = field(
        default_factory=lambda: ["L1Loss", "HuberLoss", "SmoothL1Loss"]
    )
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
