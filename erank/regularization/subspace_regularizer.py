import torch
import logging
import numpy as np
from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Union
from torch import nn
from erank.utils import load_directions_matrix_from_task_sweep

from erank.regularization.base_regularizer import Regularizer

LOGGER = logging.getLogger(__name__)

class SubspaceRegularizer(Regularizer):
    """Baseclass for subspace regularization methods.
    This class takes care of collecting gradient (parameter) directions during training and computing the optim model vec mode.

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
        epsilon_origin (float): The std of normal Gaussian which is added to small (in terms of L2-norm) optim_model_vecs
        min_optim_vec_norm (float): The lower bound for optim_model_vec norm. Below this margin, noise will be added to parameter vector.

    References:
        .. [#] Roy, Olivier, and Martin Vetterli. "The effective rank: A measure of effective dimensionality."
               2007 15th European Signal Processing Conference. IEEE, 2007.
    """
    buffer_modes = ['none', 'backlog', 'queue']
    optim_model_vec_modes = ['abs', 'initdiff', 'stepdiff', 'basediff']
    subspace_vecs_modes = ['abs', 'initdiff', 'basediff']

    def __init__(
            self,
            name: str,
            init_model: nn.Module,
            device: torch.device,
            loss_coefficient: float,
            buffer_size: int,
            buffer_mode: str,  # none, queue, backlog
            optim_model_vec_mode: str,  # abs, stepdiff, initdiff, basediff
            subspace_vecs_mode: str,  # abs, initdiff, basediff
            track_last_n_model_steps: int = 2,
            normalize_dir_matrix_m: bool = False,
            loss_coefficient_learnable: bool = False):
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

        assert self.buffer_mode in SubspaceRegularizer.buffer_modes, f'Unknown buffer mode: {self.buffer_mode}'
        assert self.optim_model_vec_mode in SubspaceRegularizer.optim_model_vec_modes, f'Unknown optimized model vector mode: {self.optim_model_vec_mode}'
        assert self.subspace_vecs_mode in SubspaceRegularizer.subspace_vecs_modes, f'Unknwon subspace vectors mode: {self.subspace_vecs_mode}'
        LOGGER.info(f'SubspaceRegularizer buffer_mode: {self.buffer_mode}')
        LOGGER.info(f'SubspaceRegularizer subspace_vecs_mode: {self.subspace_vecs_mode}')
        LOGGER.info(f'SubspaceRegularizer optim_model_vec_mode: {self.optim_model_vec_mode}')

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
            subspace_vecs = torch.normal(mean=0,
                                         std=1,
                                         size=(self.buffer_size, self._n_model_params),
                                         device=self._device)
        else:
            assert path_to_buffer_or_runs
            load_path = Path(path_to_buffer_or_runs)
            if load_path.is_dir():
                subspace_vecs = load_directions_matrix_from_task_sweep(
                    load_path, num_runs=self.buffer_size, device=self._device,
                    use_absolute_model_params=False)  # TODO make configurable
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

    def add_subspace_vec(self, model: nn.Module, ord: int = 2) -> torch.Tensor:
        """Depending on `buffer_mode` this method adds a subspace vector to the buffer.
        `buffer_mode`='backlog': Add a new model vector to `subspace_vec_backlog`. If backlog buffer is full, 
                                 buffer content is moved to `_subspace_vecs` and buffer is cleared.
        `buffer_mode`='queue': Add a new model vector to `subspace_vec_queue`

        Args:
            model (nn.Module): The model to add.
            ord (int): Norm type of the subspace vector.

        Returns:
            torch.Tensor: The norm of the subspace vector. 
        """
        if self.buffer_mode == 'none':
            return
        elif self.buffer_mode in ['backlog', 'queue']:
            assert not self.subspace_vec_buffer is None
        else:
            raise ValueError(f'Behavior not specified for buffer mode: {self.buffer_mode}')

        # compute subspace vector
        with torch.no_grad():
            model_vec = nn.utils.parameters_to_vector(model.parameters())

        if torch.isinf(model_vec).any() or torch.isnan(model_vec).any():
            LOGGER.warning(
                f'Model parameter vector of size {len(model_vec)} contains {torch.isinf(model_vec).sum()} infinite and {torch.isnan(model_vec).sum()} NaN values.'
            )

        if self.subspace_vecs_mode == 'abs':
            subspace_vec = model_vec
        elif self.subspace_vecs_mode == 'initdiff':
            subspace_vec = model_vec - self._init_model
        elif self.subspace_vecs_mode == 'basediff':
            subspace_vec = model_vec - self._base_model_vec

        # norm of subspace_vec
        subspace_vec_norm = torch.linalg.norm(subspace_vec, dim=0, ord=ord)

        # add subspace vector to buffer
        self.subspace_vec_buffer.append(subspace_vec)
        if self.buffer_mode == 'backlog':
            if len(self.subspace_vec_buffer) >= self.buffer_size:
                # move vectors in backlog to _subspace_vecs
                self._subspace_vecs.data = self._create_subspace_vecs_from_buffer(self.subspace_vec_buffer)
                self.subspace_vec_buffer.clear()
        return subspace_vec_norm

    def _create_subspace_vecs_from_buffer(self, subspace_vec_buffer: Deque[torch.Tensor]) -> torch.Tensor:
        new_subspace_vecs = torch.stack(list(subspace_vec_buffer))
        assert new_subspace_vecs.device == self._subspace_vecs.device
        if self._subspace_vecs.shape != (0,) and new_subspace_vecs.shape != self._subspace_vecs.shape:
            raise ValueError(
                f'Wrong shape of new subspace vectors! Previous shape: {self._subspace_vecs.shape}, New (wrong) shape: {new_subspace_vecs.shape}'
            )
        return new_subspace_vecs
    
    def _construct_optim_model_vec(self, model: nn.Module, optim_model_vec_mode: bool) -> torch.Tensor:
        # compute model vector for optimization
        model_vec = nn.utils.parameters_to_vector(model.parameters())  # not detached!

        if torch.isinf(model_vec).any() or torch.isnan(model_vec).any():
            LOGGER.warning(
                f'Model parameter vector of size {len(model_vec)} contains {torch.isinf(model_vec).sum()} infinite and {torch.isnan(model_vec).sum()} NaN values.'
            )

        if optim_model_vec_mode == 'abs':
            optim_model_vec = model_vec
        elif optim_model_vec_mode == 'initdiff':
            optim_model_vec = model_vec - self._init_model
        elif optim_model_vec_mode == 'stepdiff':
            optim_model_vec = model_vec - self._model_params_queue[-2]
        elif optim_model_vec_mode == 'basediff':
            optim_model_vec = model_vec - self._base_model_vec

        return optim_model_vec

    def set_base_model(self, model: nn.Module) -> None:
        """Resets the internal base model vector, which is used to construct directions.

        Args:
            model (nn.Module): The model.
        """
        if self.optim_model_vec_mode == 'basediff' or self.subspace_vecs_mode == 'basediff':
            with torch.no_grad():
                # this makes a copy of the model parameters.
                model_vec = nn.utils.parameters_to_vector(model.parameters())
                assert self._base_model_vec.device == model_vec.device
                self._base_model_vec.data = model_vec
        else:
            # LOGGER.warning('`set_base_model()` called, but no mode is `basediff`.')
            pass

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

    @abstractmethod
    def forward(self, model: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def get_additional_logs(self) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        """Return additional log values. 

        Returns:
            Dict[str, Union[float, np.ndarray, torch.Tensor]]: The dict containing additional log values.
        """
        log_dict = {}

        if self.buffer_mode == 'backlog':
            if self._subspace_vecs.shape[0] < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        elif self.buffer_mode == 'queue':
            if len(self.subspace_vec_buffer) < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        
        # get subspace vecs
        if self.buffer_mode == 'queue':
            subspace_vecs = self._create_subspace_vecs_from_buffer(self.subspace_vec_buffer)
        elif self.buffer_mode == 'backlog':
            subspace_vecs = self._subspace_vecs.data 

        log_dict['subspace_erank'] = SubspaceRegularizer.erank(subspace_vecs)
        return log_dict

    @staticmethod
    def erank(matrix_A: torch.Tensor) -> torch.Tensor:
        """Calculates the effective rank of a matrix.

        Args:
            matrix_A (torch.Tensor): Matrix of shape m x n. 
            center_matrix_A (bool): Center the matrix 

        Returns:
            torch.Tensor: Effective rank of matrix_A
        """
        assert matrix_A.ndim == 2
        # pca_lowrank causes numerical issues when evaluated near the origin
        # _, s, _ = torch.pca_lowrank(matrix_A,
        #                             center=center_matrix_A,
        #                             niter=1,
        #                             q=min(matrix_A.shape[0], matrix_A.shape[1]))  # check with torch doc.
        
        # _, s, _ = torch.linalg.svd(matrix_A, full_matrices=False)
        # newer pytorch version supports svdval computation directly
        s = torch.linalg.svdvals(matrix_A)

        # normalizes input s -> scale independent!
        return torch.exp(torch.distributions.Categorical(probs=s).entropy())
        
        
        
