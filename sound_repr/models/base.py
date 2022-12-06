from typing import List

import torch.nn


class BaseModel(torch.nn.Sequential):
    """Model template."""

    def add_module_lst(self, module_lst: List):
        """Add modules from the list to th model."""
        for idx, module in enumerate(module_lst):
            self.add_module(str(idx), module)
