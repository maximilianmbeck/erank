import torch
import logging
from abc import ABC, abstractmethod
from torch import nn

LOGGER = logging.getLogger(__name__)

class Regularizer(nn.Module, ABC):
    """A class defining the interface for a regularizer.

    Args:
        name (str): The name. Will be shown in the run logs.
        loss_coefficient (float): The (initial) loss coefficient. 
            Multiplier for the regularization term in the loss function.
        loss_coefficient_learnable (bool): If the loss_coefficient is learnable. Defaults to False.
    """

    def __init__(self, name: str, loss_coefficient: float, loss_coefficient_learnable: bool = False):
        super().__init__()
        self._name = name
        self._loss_coefficient_learnable = loss_coefficient_learnable
        self._loss_coefficient = nn.Parameter(torch.tensor(loss_coefficient), requires_grad=loss_coefficient_learnable)

    @abstractmethod
    def forward(self, model: nn.Module) -> torch.Tensor:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def loss_coefficient(self) -> torch.Tensor:
        return self._loss_coefficient.data

