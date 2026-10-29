"""Base Model Class"""

from abc import abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs: Any) -> torch.Tensor:  # type: ignore
        """Forward pass logic
        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def param_num(self) -> int:
        """Returns number of parameters"""
        return sum([param.nelement() for param in self.parameters()])

    def flops(self, inputs: Any) -> int:
        """Flop count analysis using fvcore"""
        flops = FlopCountAnalysis(self, inputs)
        return flops.total()

    def __str__(self) -> str:
        """Number of parameters in model"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
