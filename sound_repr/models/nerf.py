import torch.nn

from cfg import MainConfig
from sound_repr.models.base import BaseModel
from sound_repr.utils import build_network


class NeRFModel(BaseModel):
    def __init__(self, config: MainConfig):
        super().__init__()
        module_lst = build_network(
            2 * config.L,
            1,
            config.network,
            torch.nn.LeakyReLU(),
            config.bias,
            torch.nn.LeakyReLU(),
        )
        self.add_module_lst(module_lst)
