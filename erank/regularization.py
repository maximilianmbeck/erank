from collections import deque
from pathlib import Path
import torch
import logging
from torch import nn
from erank.utils import load_directions_matrix_from_task_sweep

LOGGER = logging.getLogger(__name__)


class EffectiveRankRegularization(nn.Module):
    """Regularization of the effective rank that discourages weights from opening up new dimensions.

    For more information on effective rank, see [#]_.

    Args:
        buffer_size (int): Number of models in the directions matrix. 
        init_model (nn.Module): Initialized model
        loss_weight (float): The weighting factor of the Effective Rank regularization term.
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

    def __init__(
            self, buffer_size: int, init_model: nn.Module, loss_weight: float, normalize_directions: bool = False,
            use_abs_model_params: bool = False):
        self.buffer_size = buffer_size  # number of directions in the buffer
        self.loss_weight = loss_weight  # weighting parameter in the loss (will be multiplied with erank term)
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

    @property
    def loss_coefficient(self):
        return self.loss_weight

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

    def init_directions_buffer(self, path_to_buffer_or_runs: str = '', random_buffer: bool = False) -> None:
        if random_buffer:
            n_model_params = self._delta_start_params_queue[0].shape[0]
            directions_buffer = torch.normal(mean=0, std=1, size=(
                self.buffer_size, n_model_params), device=self._device)
        else:
            assert path_to_buffer_or_runs
            load_path = Path(path_to_buffer_or_runs)
            if load_path.is_dir():

                directions_buffer = load_directions_matrix_from_task_sweep(
                    load_path, num_runs=self.buffer_size, device=self._device,
                    use_absolute_model_params=self._use_abs_model_params)
                LOGGER.info(
                    f'Loaded erank directions from run dir {path_to_buffer_or_runs} (shape {directions_buffer.shape}).')
            else:
                raise NotImplementedError('Loading file not supported.')

        assert directions_buffer.shape[
            0] == self.buffer_size, f'Specified buffer size is {self.buffer_size}, but given directions_buffer as shape {directions_buffer.shape}.'
        self._directions_buffer = directions_buffer

    def _construct_directions_matrix(self, model: nn.Module) -> torch.Tensor:
        delta_end_params = nn.utils.parameters_to_vector(model.parameters())  # not detached!
        if self._use_abs_model_params:
            delta = delta_end_params
        else:
            delta = delta_end_params - self._delta_start_params_queue[0]
        directions_matrix = torch.cat([delta.unsqueeze(dim=0), self._directions_buffer],
                                      dim=0)  # shape: (n_directions, n_model_parameters)
        if self._normalize_directions:
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
        directions_matrix = self._construct_directions_matrix(model)
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
        _, s, _ = torch.pca_lowrank(matrix_A, center=center_matrix_A, niter=1,
                                    q=min(matrix_A.shape[0], matrix_A.shape[1]))  # TODO check with torch doc.

        # normalizes input s -> scale independent!
        return torch.exp(torch.distributions.Categorical(s).entropy())
