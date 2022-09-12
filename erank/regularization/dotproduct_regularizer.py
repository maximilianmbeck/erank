from typing import Dict, Union
import torch
import numpy as np
from torch import nn
from erank.regularization.subspace_regularizer import SubspaceRegularizer

EPSILON_ORIGIN_STD = 1e-8
MIN_MEAN_VEC_NORM = 1e-8


class DotProductRegularizer(SubspaceRegularizer):

    name = 'dotproduct'

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
            epsilon_origin_std: float = EPSILON_ORIGIN_STD,
            min_mean_vec_norm: float = MIN_MEAN_VEC_NORM):
        super().__init__(name=DotProductRegularizer.name,
                         init_model=init_model,
                         device=device,
                         loss_coefficient=loss_coefficient,
                         buffer_size=buffer_size,
                         buffer_mode=buffer_mode,
                         optim_model_vec_mode=optim_model_vec_mode,
                         subspace_vecs_mode=subspace_vecs_mode,
                         track_last_n_model_steps=track_last_n_model_steps,
                         normalize_dir_matrix_m=normalize_dir_matrix_m,
                         loss_coefficient_learnable=loss_coefficient_learnable)

        self._epsilon_origin_std = epsilon_origin_std
        self._min_mean_vec_norm = min_mean_vec_norm

    def _calculate_mean_subspace_vec(self) -> torch.Tensor:
        # update subspace vecs
        if self.buffer_mode == 'queue':
            self._subspace_vecs.data = self._create_subspace_vecs_from_buffer(self.subspace_vec_buffer)

        #! Variant 1: first normalize then average
        # Problem: this might "upweight" unimportant directions
        # subspace_vecs_normalized = self._subspace_vecs / torch.linalg.norm(
        #     self._subspace_vecs, ord=2, dim=1, keepdim=True)
        # mean_subspace_vec_normalized = subspace_vecs_normalized.mean(dim=0)  # shape: (n_model_params,)

        #! Variant 2: first average then normalize
        mean_subspace_vec = self._subspace_vecs.mean(dim=0)
        if torch.linalg.norm(mean_subspace_vec, ord=2, dim=0) < self._min_mean_vec_norm:
            mean_subspace_vec += self._epsilon_origin_std * torch.randn_like(mean_subspace_vec, requires_grad=False)
        
        mean_subspace_vec_normalized = mean_subspace_vec / torch.linalg.norm(
            mean_subspace_vec, ord=2, dim=0, keepdim=True)

        return mean_subspace_vec_normalized

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Calculate the dot-product regularization term. 
        We want to maximize the dot-product to push the update direction into the direction of the subspace vectors."""
        if self.buffer_mode == 'backlog':
            if self._subspace_vecs.shape[0] < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device)
        elif self.buffer_mode == 'queue':
            if len(self.subspace_vec_buffer) < self.buffer_size:
                return torch.tensor(0.0, dtype=torch.float32, device=self._device)

        self._update_model_params_queue(model)

        # shape: (n_model_params,)
        optim_model_vec = self._construct_optim_model_vec(model=model, optim_model_vec_mode=self.optim_model_vec_mode)

        mean_subspace_vec = self._calculate_mean_subspace_vec()  # shape: (n_model_params,)

        # calculate dotproduct regularization term, negative since we want to maximize dotproduct, align vectors more
        return -torch.dot(optim_model_vec, mean_subspace_vec)

    def get_additional_logs(self) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        return super().get_additional_logs()
