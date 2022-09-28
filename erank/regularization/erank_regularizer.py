from typing import Dict, Union
import torch
import logging
import numpy as np
from torch import nn

from erank.regularization.subspace_regularizer import SubspaceRegularizer

LOGGER = logging.getLogger(__name__)

EPSILON_ORIGIN = 1e-8
MIN_OPTIM_VEC_NORM = 1e-8

class EffectiveRankRegularizer(SubspaceRegularizer):
    """Regularization of the effective rank that discourages weights from opening up new dimensions.
    At the core of this "regularizer" (not sure, if this method can be considered as regularization at all) 
    is a (direction) matrix M, which contains in its rows the weights of a neural network model. 
    The (direction) matrix M can be divided in two parts (The vectors in these parts are stacked in rows.): 
    - The subspace vectors (`subspace_vecs`): Weight vectors determining a subspace of the model being optimized. 
            These vectors are detached and are not optimized (at least so far).
    - The model vector (`optim_model_vec`): The parameters of the model being optimized flattened as a vector.

    For more information on effective rank, see [#]_.
    Args:
        epsilon_origin_std (float, optional): _description_. Defaults to EPSILON_ORIGIN.
        min_optim_vec_norm (float, optional): _description_. Defaults to MIN_OPTIM_VEC_NORM.
        for others: see class SubspaceRegularizer
    """
    name = 'erank'

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
            normalize_partial_gradient: bool = False,
            epsilon_origin_std: float = EPSILON_ORIGIN,
            min_optim_vec_norm: float = MIN_OPTIM_VEC_NORM):
        super().__init__(name=EffectiveRankRegularizer.name,
                         init_model=init_model,
                         device=device,
                         loss_coefficient=loss_coefficient,
                         buffer_size=buffer_size,
                         buffer_mode=buffer_mode,
                         optim_model_vec_mode=optim_model_vec_mode,
                         subspace_vecs_mode=subspace_vecs_mode,
                         track_last_n_model_steps=track_last_n_model_steps,
                         normalize_dir_matrix_m=normalize_dir_matrix_m,
                         loss_coefficient_learnable=loss_coefficient_learnable,
                         normalize_partial_gradient=normalize_partial_gradient)

        self._epsilon_origin_std = epsilon_origin_std
        self._min_optim_vec_norm = min_optim_vec_norm

    def _get_normalized_erank(self, model: nn.Module) -> float:
        """Returns the normalized effective rank of matrix m composed of the subspace_vecs and the model."""
        if self._subspace_vecs.shape[0] < self.buffer_size:
            return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        directions_matrix = self._construct_directions_matrix_m(model, use_abs_model_params=True, normalize_matrix=True)
        return EffectiveRankRegularizer.erank(directions_matrix)

    def _construct_directions_matrix_m(self, model: nn.Module, optim_model_vec_mode: bool,
                                       normalize_matrix: bool) -> torch.Tensor:
        """Constructs the directions matrix M (which is used to calculate the erank). 
        Each row contains the parameters of a (pretrained) model flattened as a vector."""

        optim_model_vec = self._construct_optim_model_vec(model=model, optim_model_vec_mode=optim_model_vec_mode)

        # avoid numeric instabilities, when evaluating erank gradient
        if torch.linalg.norm(optim_model_vec, ord=2, dim=0) < self._min_optim_vec_norm:
            optim_model_vec += self._epsilon_origin_std * torch.randn_like(optim_model_vec, requires_grad=False)

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
                return torch.tensor(0.0, dtype=torch.float32, device=self._device, requires_grad=True)
        elif self.buffer_mode == 'queue':
            if len(self.subspace_vec_buffer) < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device, requires_grad=True)

        self._update_model_params_queue(model)
        directions_matrix = self._construct_directions_matrix_m(model, self.optim_model_vec_mode,
                                                                self.normalize_dir_matrix_m)
        return SubspaceRegularizer.erank(directions_matrix)

    def get_additional_logs(self) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        return super().get_additional_logs()