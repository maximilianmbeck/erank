import torch
import logging
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple
from torch import nn
from erank.utils import load_directions_matrix_from_task_sweep

LOGGER = logging.getLogger(__name__)

LOG_LOSS_PREFIX = 'loss'
LOG_LOSS_TOTAL_KEY = f'{LOG_LOSS_PREFIX}_total'


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


class RegularizedLoss(nn.Module):

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss_module = loss
        self._regularizers: nn.ModuleDict[str, Regularizer] = nn.ModuleDict({})

    def forward(self,
                y_preds: torch.Tensor,
                y_labels: torch.Tensor,
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
        return self._regularizers[regularizer_name]

    def contains_regularizer(self, regularizer_name: str) -> bool:
        return regularizer_name in self._regularizers


class EffectiveRankRegularization(Regularizer):
    """Regularization of the effective rank that discourages weights from opening up new dimensions.
    At the core of this "regularizer" (not sure, if this method can be considered as regularization at all) 
    is a (direction) matrix M, which contains in its rows the weights of a neural network model. 
    The (direction) matrix M can be divided in two parts (The vectors in these parts are stacked in rows.): 
    - The subspace vectors (`subspace_vecs`): Weight vectors determining a subspace of the model being optimized. 
            These vectors are detached and are not optimized (at least so far).
    - The model vector (`optim_model_vec`): The parameters of the model being optimized flattened as a vector.

    For more information on effective rank, see [#]_.

    Args:
        init_model (nn.Module): Initialized model
        loss_coefficient (float): The weighting factor of the Effective Rank regularization term.
        buffer_size (int): Number of subspace vectors in the directions matrix M. 
        buffer_mode (str): Mode for internal buffer.
                           Options:
                             - none: Disable internal buffer. Subspace vecs are kept constant after initialization throughout training.
                             - backlog: Keeps subspace vectors in a separate backlog buffer and swaps them in for directions matrix M 
                                        upon request. The backlog buffer is stored on CPU memory by default.
                             - queue: Uses a queue and keep always the last `buffer_size` subspace vectors. 
                                      The subspace vectors in the queue are stored on CPU memory by default.
        optim_model_vec_mode (str): Preprocessing mode for the flattened model parameter vector (of the model being optimized).
                                    Preprocessing is done before computing the directions matrix M.
                                    Options:
                                      - abs: absolute model parameters
                                      - initdiff: difference between initialization and the current model
                                      - stepdiff: last model step diff, i.e. the difference between the last model 
                                                  and the current model during training
                                      - basediff: difference between current model to a specific set of "base parameters"
        subspace_vecs_mode (str): Preprocessing mode for the subspace vectors. Preprocessing is done before computing the directions
                                  matrix M, in the methods `add_subspace_vec` or `init_substpace_vecs`.
                                  Options:
                                    - abs: absolute model parameters
                                    - initdiff: difference between initialization and trained model
                                    - basediff: difference of the vectors to add to a specific set of "base parameters"
        track_last_n_model_steps (int, optional): Keep a copy of the last n parameter vectors. Use `track_last_n_model_steps=2` 
                                                   if you want to use the update step as `optim_model_vec` or compute the update step length.
        normalize_directions (bool, optional): Normalize all model parameters in the directions matrix M, 
                                               i.e. all row vectors in M are normalized to 1 before computing the erank. 
                                               Defaults to False.

    References:
        .. [#] Roy, Olivier, and Martin Vetterli. "The effective rank: A measure of effective dimensionality."
               2007 15th European Signal Processing Conference. IEEE, 2007.
    """
    buffer_modes = ['none', 'backlog', 'queue']
    optim_model_vec_modes = ['abs', 'initdiff', 'stepdiff', 'basediff']
    subspace_vecs_modes = ['abs', 'initdiff', 'basediff']

    def __init__(
            self,
            init_model: nn.Module,
            device: torch.device,
            loss_coefficient: float,
            buffer_size: int,
            buffer_mode: str,  # none, queue, backlog
            optim_model_vec_mode: str,  # abs, stepdiff, initdiff, basediff
            subspace_vecs_mode: str,  # abs, initdiff, basediff
            track_last_n_model_steps: int = 2,
            normalize_dir_matrix_m: bool = False,
            loss_coefficient_learnable: bool = False,
            name: str = 'erank'):
        super().__init__(name=name,
                         loss_coefficient=loss_coefficient,
                         loss_coefficient_learnable=loss_coefficient_learnable)
        self.buffer_size = buffer_size  # number of subspace vectors / directions in the buffer
        self.buffer_mode = buffer_mode
        self.optim_model_vec_mode = optim_model_vec_mode
        self.subspace_vecs_mode = subspace_vecs_mode
        self.track_last_n_model_steps = track_last_n_model_steps
        self.normalize_dir_matrix_m = normalize_dir_matrix_m
        self._device = device
        self._n_model_params = len(nn.utils.parameters_to_vector(init_model.parameters()))

        assert self.buffer_mode in EffectiveRankRegularization.buffer_modes, f'Unknown buffer mode: {self.buffer_mode}'
        assert self.optim_model_vec_mode in EffectiveRankRegularization.optim_model_vec_modes, f'Unknown optimized model vector mode: {self.optim_model_vec_mode}'
        assert self.subspace_vecs_mode in EffectiveRankRegularization.subspace_vecs_modes, f'Unknwon subspace vectors mode: {self.subspace_vecs_mode}'

        #* subspace vec buffer
        self.subspace_vec_buffer: Deque[torch.Tensor] = None
        if self.buffer_mode == 'backlog':
            self.subspace_vec_buffer = deque()
        elif self.buffer_mode == 'queue':
            self.subspace_vec_buffer = deque(maxlen=self.buffer_size)

        #* init model
        self._init_model: torch.Tensor = None
        if self.subspace_vecs_mode == 'initdiff':
            # use nn.Paramater such, that _init_model is properly registered in Module (e.g. for .to() call)
            self._init_model = nn.Parameter(nn.utils.parameters_to_vector(init_model.parameters()).detach(),
                                            requires_grad=False)

        #* base model
        self._base_model_vec: torch.Tensor = None
        if self.optim_model_vec_mode == 'basediff' or self.subspace_vecs_mode == 'basediff':
            # parameters_to_vector and .detach() return a tensor that shares the same memory, hence self._base_model_vec is only a reference
            self._base_model_vec = nn.Parameter(nn.utils.parameters_to_vector(init_model.parameters()).detach(),
                                                requires_grad=False)

        #* subspace vecs
        # subspace_vecs is a tensor containing all (direction) vectors that span the subspace
        # it has shape: buffer_size x n_params of the model
        self._subspace_vecs = nn.Parameter(torch.tensor([], device=self._device), requires_grad=False)

        #* param queue
        # We use a deque here as we want to compute the difference to the PREVIOUS step.
        self._model_params_queue: Deque[torch.Tensor] = None
        if self.optim_model_vec_mode == 'stepdiff' and self.track_last_n_model_steps < 2:
            # we need to keep at least the last two models to compute the step difference
            self.track_last_n_model_steps = 2
        if self.track_last_n_model_steps > 0:
            self._model_params_queue = deque(maxlen=self.track_last_n_model_steps)
            self._model_params_queue.append(nn.utils.parameters_to_vector(init_model.parameters()).detach())

    def init_subspace_vecs(self, path_to_buffer_or_runs: str = '', random_buffer: bool = False) -> None:
        if random_buffer:
            n_model_params = self._model_params_queue[0].shape[0]
            subspace_vecs = torch.normal(mean=0, std=1, size=(self.buffer_size, n_model_params), device=self._device)
        else:
            assert path_to_buffer_or_runs
            load_path = Path(path_to_buffer_or_runs)
            if load_path.is_dir():
                subspace_vecs = load_directions_matrix_from_task_sweep(
                    load_path,
                    num_runs=self.buffer_size,
                    device=self._device,
                    use_absolute_model_params=self._use_abs_model_params)
                LOGGER.info(
                    f'Loaded erank directions from run dir {path_to_buffer_or_runs} (shape {subspace_vecs.shape}).')
            else:
                raise NotImplementedError('Loading file not supported.')

        assert subspace_vecs.shape[
            0] == self.buffer_size, f'Specified buffer size is {self.buffer_size}, but given directions_buffer as shape {subspace_vecs.shape}.'
        self._subspace_vecs.data = subspace_vecs

    def _update_model_params_queue(self, model: nn.Module) -> None:
        """Updates the model params queue with the parameters in the given model.
        """
        if not self._model_params_queue is None:
            with torch.no_grad():
                self._model_params_queue.append(nn.utils.parameters_to_vector(model.parameters()))

    def add_subspace_vec(self, model: nn.Module) -> None:
        """Depending on `buffer_mode` this method adds a subspace vector to the buffer.
        `buffer_mode`='backlog': Add a new model vector to `subspace_vec_backlog`. If backlog buffer is full, 
                                 buffer content is moved to `_subspace_vecs` and buffer is cleared.
        `buffer_mode`='queue': Add a new model vector to `subspace_vec_queue`
        TODO check if tensors must be moved to cpu memory
        Args:
            model (nn.Module): The model to add.
        """
        if self.buffer_mode in ['backlog', 'queue']:
            assert not self.subspace_vec_buffer is None
        else:
            raise ValueError(f'Behavior not specified for buffer mode: {self.buffer_mode}')

        # compute subspace vector
        with torch.no_grad():
            model_vec = nn.utils.parameters_to_vector(model.parameters())

        # if torch.isnan(model_vec).any():
        #     raise RuntimeError('Model parameter vector contains NaNs.')

        if self.subspace_vecs_mode == 'abs':
            subspace_vec = model_vec
        elif self.subspace_vecs_mode == 'initdiff':
            subspace_vec = model_vec - self._init_model
        elif self.subspace_vecs_mode == 'basediff':
            subspace_vec = model_vec - self._base_model_vec

        # add subspace vector to buffer
        self.subspace_vec_buffer.append(subspace_vec)
        if self.buffer_mode == 'backlog':
            if len(self.subspace_vec_buffer) >= self.buffer_size:
                # move vectors in backlog to _subspace_vecs
                self._subspace_vecs.data = self._create_subspace_vecs_from_buffer(self.subspace_vec_buffer)
                self.subspace_vec_buffer.clear()

    def _create_subspace_vecs_from_buffer(self, subspace_vec_buffer: Deque[torch.Tensor]) -> torch.Tensor:
        new_subspace_vecs = torch.stack(list(subspace_vec_buffer))
        assert new_subspace_vecs.device == self._subspace_vecs.device
        if self._subspace_vecs.shape != (0,) and new_subspace_vecs.shape != self._subspace_vecs.shape:
            raise ValueError(
                f'Wrong shape of new subspace vectors! Previous shape: {self._subspace_vecs.shape}, New (wrong) shape: {new_subspace_vecs.shape}'
            )
        return new_subspace_vecs

    def set_base_model(self, model: nn.Module) -> None:
        """Resets the internal base model vector, which is used to construct directions.

        Args:
            model (nn.Module): The model.
        """
        with torch.no_grad():
            # this makes a copy of the model parameters.
            model_vec = nn.utils.parameters_to_vector(model.parameters())
            assert self._base_model_vec.device == model_vec.device
            self._base_model_vec.data = model_vec

    def get_model_update_step_norm(self, steps_before: int = 1, ord: int = 2) -> float:
        """Returns the norm of the update step. The update step is the difference between the last model and the model at `steps_before` 
        update steps.

        Args:
            steps_before (int, optional): The delta index counting from the last update step backwards. Defaults to 1.
            ord (int, optional): ord of the norm. Defaults to 2.

        Returns:
            float: Norm of the update step.
        """
        prev_index = steps_before + 1
        if self._model_params_queue is None or len(self._model_params_queue) < prev_index:
            raise ValueError('`model_params_queue` not initialized properly. Does it have the correct length?')
        else:
            # subtract the previous added item from the last added item
            delta = self._model_params_queue[-1] - self._model_params_queue[-prev_index]
            return torch.linalg.norm(delta, dim=0, ord=ord).item()

    def get_normalized_erank(self, model: nn.Module) -> float:
        """Returns the normalized effective rank of matrix m composed of the subspace_vecs and the model."""
        if self._subspace_vecs.shape[0] < self.buffer_size:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        directions_matrix = self._construct_directions_matrix_m(model, use_abs_model_params=True, normalize_matrix=True)
        return EffectiveRankRegularization.erank(directions_matrix)

    def _construct_directions_matrix_m(self, model: nn.Module, optim_model_vec_mode: bool,
                                       normalize_matrix: bool) -> torch.Tensor:
        """Constructs the directions matrix M (which is used to calculate the erank). 
        Each row contains the parameters of a (pretrained) model flattened as a vector."""
        # compute model vector for optimization
        model_vec = nn.utils.parameters_to_vector(model.parameters())  # not detached!

        # if torch.isnan(model_vec).any():
        #     raise RuntimeError('Model parameter vector contains NaNs, before constructing the directions matrix M.')
        
        if torch.isinf(model_vec).any():
            raise RuntimeError('Model parameter vector contains infinite values, this will likely cause NaNs.')

        if optim_model_vec_mode == 'abs':
            optim_model_vec = model_vec
        elif optim_model_vec_mode == 'initdiff':
            optim_model_vec = model_vec - self._init_model
        elif optim_model_vec_mode == 'stepdiff':
            optim_model_vec = model_vec - self._model_params_queue[-2]
        elif optim_model_vec_mode == 'basediff':
            optim_model_vec = model_vec - self._base_model_vec

        # update subspace vecs
        if self.buffer_mode == 'queue':
            self._subspace_vecs.data = self._create_subspace_vecs_from_buffer(self.subspace_vec_buffer)

        assert self._subspace_vecs.shape[0] == self.buffer_size
        directions_matrix = torch.cat([optim_model_vec.unsqueeze(dim=0), self._subspace_vecs],
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
        if self.buffer_mode == 'backlog':
            if self._subspace_vecs.shape[0] < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        elif self.buffer_mode == 'queue':
            if len(self.subspace_vec_buffer) < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device)

        self._update_model_params_queue(model)
        directions_matrix = self._construct_directions_matrix_m(model, self.optim_model_vec_mode,
                                                                self.normalize_dir_matrix_m)
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
