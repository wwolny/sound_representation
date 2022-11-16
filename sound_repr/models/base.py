import torch.nn
from typing import List


class BaseModel(torch.nn.Sequential):
    def add_module_lst(self, module_lst: List):
        for idx, module in enumerate(module_lst):
            self.add_module(str(idx), module)
