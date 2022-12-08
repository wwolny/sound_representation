from typing import List

import torch.nn as nn

from sound_repr.cfg.main_config import MainConfig
from sound_repr.models.baseline import BaselineModel
from sound_repr.models.nerf import NeRFModel
from sound_repr.models.siren import SIRENModel
from sound_repr.utils import Sine


class TestModulesInit:
    def test_baseline_model(self):
        layer_sizes = [1, 16, 32]
        mc = MainConfig.from_dict({"mode": "default", "network": layer_sizes})
        baseline_model = BaselineModel(mc)
        self.run_setup(baseline_model, layer_sizes, nn.LeakyReLU)

    def test_nerf_model(self):
        layer_sizes = [1, 16, 32]
        mc = MainConfig.from_dict({"mode": "nerf", "network": layer_sizes})
        nerf_model = NeRFModel(mc)
        self.run_setup(nerf_model, layer_sizes, nn.LeakyReLU)

    def test_siren_model(self):
        layer_sizes = [1, 16, 32]
        mc = MainConfig.from_dict({"mode": "siren", "network": layer_sizes})
        siren_model = SIRENModel(mc)
        self.run_setup(siren_model, layer_sizes, Sine)

    @staticmethod
    def run_setup(model: nn.Module, layer_sizes: List, activation: nn.Module):
        # Last layer is always 1
        layer_sizes.append(1)
        # Revers layers size list for popping
        layer_sizes.reverse()
        for _, module in model.named_children():
            if isinstance(module, nn.Linear):
                assert module.out_features == layer_sizes.pop()
            else:
                assert isinstance(module, activation)
