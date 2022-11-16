import torch.nn

from cfg import MainConfig
from sound_repr.models.base import BaseModel
from sound_repr.utils import build_network


class BaselineModel(BaseModel):
    def __init__(self, config: MainConfig):
        super().__init__()
        module_lst = build_network(
            1,
            1,
            config.network,
            torch.nn.LeakyReLU(),
            config.bias,
            torch.nn.LeakyReLU(),
        )
        print(module_lst)
        self.add_module_lst(module_lst)
