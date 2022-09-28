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
        normalize_partial_gradient (bool): If the gradient of this loss term should be computed separately 
            and rescaled to the length of `loss_coefficient`.
    """

    def __init__(self,
                 name: str,
                 loss_coefficient: float,
                 loss_coefficient_learnable: bool = False,
                 normalize_partial_gradient: bool = False):
        super().__init__()
        self._name = name
        self._loss_coefficient_learnable = loss_coefficient_learnable
        self._loss_coefficient = nn.Parameter(torch.tensor(loss_coefficient), requires_grad=loss_coefficient_learnable)
        self._normalize_partial_gradient = normalize_partial_gradient
        assert not (
            loss_coefficient_learnable and normalize_partial_gradient
        ), 'Loss coefficient cannot be learnt, when the partial gradient of this regularizer is computed separately.'

    @abstractmethod
    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute regularization here."""
        pass

    @property
    def normalize_partial_gradient(self) -> bool:
        """Flag, determining wether to normalize the gradient of this term 
        before accumulating the gradient of the total loss."""
        return self._normalize_partial_gradient

    @property
    def name(self) -> str:
        """Name of the regularizer. Used for logging."""
        return self._name

    @property
    def loss_coefficient(self) -> torch.Tensor:
        """Coefficient for the regularization term in the total loss. 
        If `normalize_partial_gradient` is true, this coefficient is used to scale the gradient."""
        return self._loss_coefficient.data
