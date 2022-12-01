import numpy as np
import torch.nn

from sound_repr.cfg import MainConfig
from sound_repr.models.base import BaseModel
from sound_repr.utils import Sine, build_network


class SIRENModel(BaseModel):
    def __init__(self, config: MainConfig):
        super().__init__()
        module_lst = build_network(
            1,
            1,
            config.network,
            Sine(config.hidden_omega),
            config.bias,
            Sine(config.omega),
        )

        with torch.no_grad():
            for id_mod, mod in enumerate(module_lst):
                if isinstance(mod, torch.nn.Linear):
                    if id_mod == 0:
                        mod.weight.uniform_(
                            -1 / mod.in_features, 1 / mod.in_features
                        )
                        if config.siren_bias_init:
                            mod.bias.uniform_(
                                -1 / mod.in_features, 1 / mod.in_features
                            )
                    else:
                        mod.weight.uniform_(
                            -np.sqrt(6 / mod.in_features)
                            / config.hidden_omega,
                            np.sqrt(6 / mod.in_features) / config.hidden_omega,
                        )
                        if config.siren_bias_init:
                            mod.bias.uniform_(
                                -np.sqrt(6 / mod.in_features)
                                / config.hidden_omega,
                                np.sqrt(6 / mod.in_features)
                                / config.hidden_omega,
                            )
        self.add_module_lst(module_lst)
