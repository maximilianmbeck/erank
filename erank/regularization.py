from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Dict, Tuple
import torch
import logging
from torch import nn
from erank.utils import load_directions_matrix_from_task_sweep

LOGGER = logging.getLogger(__name__)

LOG_LOSS_PREFIX = 'loss'
LOG_LOSS_TOTAL_KEY = f'{LOG_LOSS_PREFIX}_total'

class Regularizer(nn.Module, ABC):
    """A class defining the interface for a regularizer.

    Args:
        name (str): The name. Will be shown in the run logs.
        loss_coefficient (float): The loss coefficient. 
            Multiplier for the regularization term in the loss function.
    """

    def __init__(self, name: str, loss_coefficient: float):
        super().__init__()
        self._name = name
        self._loss_coefficient = loss_coefficient

    @abstractmethod
    def forward(self, model: nn.Module) -> torch.Tensor:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def loss_coefficient(self) -> float:
        return self._loss_coefficient


class RegularizedLoss(nn.Module):

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss_module = loss
        self._regularizers: Dict[str, Regularizer] = {}

    def forward(self, y_preds: torch.Tensor, y_labels: torch.Tensor,
                model: nn.Module = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = {}
        loss = self.loss_module(y_preds, y_labels)
        loss_dict[f'{LOG_LOSS_PREFIX}_{self.loss_module.__class__.__name__}'] = loss
        loss_total = loss
        if not model is None:
            for reg_name, reg in self._regularizers.items():
                loss_reg = reg(model)
                loss_dict[f'{LOG_LOSS_PREFIX}_{reg_name}'] = loss_reg
                loss_total += reg.loss_coefficient * loss_reg

        loss_dict[LOG_LOSS_TOTAL_KEY] = loss_total

        return loss_total, loss_dict

    def add_regularizer(self, regularizer: Regularizer) -> None:
        if regularizer.name in self._regularizers.keys():
            raise ValueError(f'Regularizer {regularizer.name} already exists in RegularizedLoss.')
        else:
            LOGGER.info(f'Adding regularizer `{regularizer.name}` to RegularizdLoss')
            self._regularizers[regularizer.name] = regularizer

    def get_regularizer(self, regularizer_name: str) -> Regularizer:
        return self._regularizers.get(regularizer_name, None)

    def contains_regularizer(self, regularizer_name: str) -> bool:
        return regularizer_name in self._regularizers


class EffectiveRankRegularization(Regularizer):
    """Regularization of the effective rank that discourages weights from opening up new dimensions.

    For more information on effective rank, see [#]_.

    Args:
        buffer_size (int): Number of models in the directions matrix. 
        init_model (nn.Module): Initialized model
        loss_coefficient (float): The weighting factor of the Effective Rank regularization term.
        normalize_directions (bool, optional): Normalized all model parameters in the directions matrix, 
                                               i.e. all directions to 1 before computing the erank. Defaults to False.
        use_abs_model_params (bool, optional): If true, use absolute model parameters in for the direction matrix. 
                                               The models are stacked, i.e. each row contains the parameters of a model. 
                                               If false, use model training differences, i.e. for pretrained models: 
                                               best model minus initialization. For the model that is optimized during training 
                                               use current model parameters minus previous model parameters. 

    References:
        .. [#] Roy, Olivier, and Martin Vetterli. "The effective rank: A measure of effective dimensionality."
               2007 15th European Signal Processing Conference. IEEE, 2007.
    """

    def __init__(self,
                 buffer_size: int,
                 init_model: nn.Module,
                 loss_coefficient: float,
                 normalize_directions: bool = False,
                 use_abs_model_params: bool = False,
                 name: str = 'erank'):
        super().__init__(name=name, loss_coefficient=loss_coefficient)
        self.buffer_size = buffer_size  # number of directions in the buffer
        self._device = next(iter(init_model.parameters())).device

        self._normalize_directions = normalize_directions
        self._use_abs_model_params = use_abs_model_params

        # directions buffer is a tensor containing all directions that span the subspace
        # it has shape: buffer_size x n_params of the model
        self._directions_buffer = torch.tensor([], device=self._device)
        # the params to compute the delta of the current model step to
        # We use a deque here as we want to compute the difference to the PREVIOUS step. Therefore we access the 0th element for computing the difference.
        self._delta_start_params_queue = deque(maxlen=2)
        self._delta_start_params_queue.append(nn.utils.parameters_to_vector(init_model.parameters()).detach())

    def update_delta_start_params(self, model: nn.Module) -> None:
        """Updates the start vector to which the delta of the next parameter update of the model will be computed

        Args:
            model (nn.Module): The model containing the parameters.
        """
        self._delta_start_params_queue.append(nn.utils.parameters_to_vector(model.parameters()).detach())

    def update_directions_buffer(self, model: nn.Module) -> None:
        # FIFO queue for _directions_buffer
        raise NotImplementedError()

    def get_param_step_len(self) -> float:
        """Returns the difference between the model parameter vectors in the delta_start_params_queue.
        This corresponds to the optimizer step length, if the queue is updated after every optimizer step.
        """
        if len(self._delta_start_params_queue) < 2:
            return 0.0
        else:
            # subtract the previous added item from the last added item
            delta = self._delta_start_params_queue[-1] - self._delta_start_params_queue[-2]
            return torch.linalg.norm(delta, ord=2).item()

    def get_normalized_erank(self) -> float:
        """Returns the normalized effective rank of matrix composed of the last model and the pretrained models."""
        if self._directions_buffer.shape[0] < self.buffer_size or len(self._delta_start_params_queue) < 2:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        model = self._delta_start_params_queue[-1]
        directions_matrix = self._construct_directions_matrix(model, use_abs_model_params=True, normalize_matrix=True)
        return EffectiveRankRegularization.erank(directions_matrix)

    def init_directions_buffer(self, path_to_buffer_or_runs: str = '', random_buffer: bool = False) -> None:
        if random_buffer:
            n_model_params = self._delta_start_params_queue[0].shape[0]
            directions_buffer = torch.normal(mean=0,
                                             std=1,
                                             size=(self.buffer_size, n_model_params),
                                             device=self._device)
        else:
            assert path_to_buffer_or_runs
            load_path = Path(path_to_buffer_or_runs)
            if load_path.is_dir():

                directions_buffer = load_directions_matrix_from_task_sweep(
                    load_path,
                    num_runs=self.buffer_size,
                    device=self._device,
                    use_absolute_model_params=self._use_abs_model_params)
                LOGGER.info(
                    f'Loaded erank directions from run dir {path_to_buffer_or_runs} (shape {directions_buffer.shape}).')
            else:
                raise NotImplementedError('Loading file not supported.')

        assert directions_buffer.shape[
            0] == self.buffer_size, f'Specified buffer size is {self.buffer_size}, but given directions_buffer as shape {directions_buffer.shape}.'
        self._directions_buffer = directions_buffer

    def _construct_directions_matrix(self, model: nn.Module, use_abs_model_params: bool,
                                     normalize_matrix: bool) -> torch.Tensor:
        """Constructs the directions matrix (which is used to calculate the erank). 
        Each row contains the parameters of a (pretrained) model flattened as a vector."""
        delta_end_params = nn.utils.parameters_to_vector(model.parameters())  # not detached!
        if use_abs_model_params:
            # concatenate absolute model parameters
            delta = delta_end_params
        else:
            # concatenate last update step
            delta = delta_end_params - self._delta_start_params_queue[0]
        directions_matrix = torch.cat([delta.unsqueeze(dim=0), self._directions_buffer],
                                      dim=0)  # shape: (n_directions, n_model_parameters)
        if normalize_matrix:
            directions_matrix = directions_matrix / torch.linalg.norm(directions_matrix, ord=2, dim=1, keepdim=True)
        return directions_matrix

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Calculate the Effective Rank regularization term.

        Args:
            model (nn.Module): the model parameters to regularize

        Returns:
            torch.Tensor: Effective Rank regularization term.
        """
        # Apply erank regularization only if directions buffer is full and we have a first step in the delta_start_params_queue
        if self._directions_buffer.shape[0] < self.buffer_size or len(self._delta_start_params_queue) < 2:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        directions_matrix = self._construct_directions_matrix(model, self._use_abs_model_params,
                                                              self._normalize_directions)
        return EffectiveRankRegularization.erank(directions_matrix)

    @staticmethod
    def erank(matrix_A: torch.Tensor, center_matrix_A: bool = False) -> torch.Tensor:
        """Calculates the effective rank of a matrix.

        Args:
            matrix_A (torch.Tensor): Matrix of shape m x n. 
            center_matrix_A (bool): Center the matrix 

        Returns:
            torch.Tensor: Effective rank of matrix_A
        """
        assert matrix_A.ndim == 2
        _, s, _ = torch.pca_lowrank(matrix_A,
                                    center=center_matrix_A,
                                    niter=1,
                                    q=min(matrix_A.shape[0], matrix_A.shape[1]))  # TODO check with torch doc.

        # normalizes input s -> scale independent!
        return torch.exp(torch.distributions.Categorical(s).entropy())
