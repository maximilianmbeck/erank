from typing import Dict, Tuple, Callable, Union
import sys
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.utils import set_seed, get_device
from tqdm import tqdm
from erank.regularization.subspace_regularizer import SubspaceRegularizer

#### Compare model weights, by treating each model as vector. Similarity functions are distance functions on vectors.


def plot_similarity_between_models(model_matrix: torch.Tensor,
                                   similarity_function: Callable,
                                   fig_title=None,
                                   fig_ax: Tuple[mpl.figure.Figure, mpl.axis.Axis] = None,
                                   cmap='RdBu',
                                   alpha=0.8):
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54))
    else:
        fig, ax = fig_ax

    n_models = model_matrix.shape[0]
    similarity_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(i + 1):
            if i == j:
                similarity_matrix[i, j] = np.nan
            else:
                similarity = similarity_function(model_matrix[i], model_matrix[j]).item()
                similarity_matrix[j, i] = similarity
                similarity_matrix[i, j] = np.nan

    cax = ax.matshow(similarity_matrix, alpha=alpha, cmap=cmap)
    for (i, j), z in np.ndenumerate(similarity_matrix):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_title('Similarity between models')
    fig.colorbar(cax, ax=ax)
    fig.suptitle(fig_title)
    return ax


def plot_similarity_to_model(model_matrix: torch.Tensor,
                             compare_model: torch.Tensor,
                             similarity_function: Callable,
                             ax_title=None,
                             fig_title=None,
                             fig_ax: Tuple[mpl.figure.Figure, mpl.axis.Axis] = None,
                             cmap='viridis',
                             alpha=0.8):
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54))
    else:
        fig, ax = fig_ax

    n_models = model_matrix.shape[0]
    similarity_matrix = np.zeros((n_models, 1))
    for i in range(n_models):
        similarity = similarity_function(model_matrix[i], compare_model).item()
        similarity_matrix[i] = similarity

    cax = ax.matshow(similarity_matrix, alpha=alpha, cmap=cmap)
    for (i, j), z in np.ndenumerate(similarity_matrix):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_models))
    ax.set_title(ax_title)
    fig.colorbar(cax, ax=ax)
    fig.suptitle(fig_title)
    return ax


def plot_similarity_to_mean_model(model_matrix: torch.Tensor,
                                  similarity_function: Callable,
                                  fig_title=None,
                                  fig_ax: Tuple[mpl.figure.Figure, mpl.axis.Axis] = None,
                                  cmap='viridis',
                                  alpha=0.8):
    mean_model = model_matrix.mean(dim=0, keepdim=False)
    return plot_similarity_to_model(model_matrix,
                                    mean_model,
                                    similarity_function,
                                    ax_title='Similarity to mean model',
                                    fig_title=fig_title,
                                    fig_ax=fig_ax,
                                    cmap=cmap,
                                    alpha=alpha)


####

#### plot erank


def plot_models_erank(models: Union[torch.Tensor, Dict[str, torch.Tensor]],
                      model_steps: int = 10,
                      random_baseline: bool = True,
                      random_init_model: BaseModel = None,
                      erank_fn: Callable[[torch.Tensor], torch.Tensor] = SubspaceRegularizer.erank,
                      ax_title: str = None,
                      fig_title: str = None,
                      alpha=0.8,
                      fig_ax: Tuple[mpl.figure.Figure, mpl.axis.Axis] = None,
                      seed: int = 0,
                      device: Union[torch.device, str, int] = "auto"):
    GAUSS_RAND_KEY = 'Gaussian random'
    RAND_MODEL_INIT_KEY = 'Random model initializations'
    TRAINED_MODELS_KEY = 'Trained models'

    def _erank(matrix: torch.Tensor) -> float:
        return erank_fn(matrix).item()

    device = get_device(device)

    # setup
    set_seed(seed)
    erank_data = {}
    model_matrix_shape = None  # (num_models, num_model_parameters)
    if isinstance(models, torch.Tensor):
        model_matrix_shape = models.shape
        models = {TRAINED_MODELS_KEY: models.to(device=device)}
        erank_data.update({TRAINED_MODELS_KEY: []})
    elif isinstance(models, dict):
        model_erank_data = {}
        for descr, model_matrix in models.items():
            models[descr] = model_matrix.to(device=device)
            if not model_matrix_shape:
                model_matrix_shape = model_matrix.shape
            assert model_matrix_shape == model_matrix.shape, f'Shape of `{descr}` model matrix does not match!'
            model_erank_data[descr] = []
        erank_data.update(model_erank_data)

    num_vectors = model_matrix_shape[0]
    assert isinstance(models, dict)

    random_vecs = None
    if random_baseline:
        random_vecs = torch.randn(model_matrix_shape, device=device)
        erank_data.update({GAUSS_RAND_KEY: []})

    if random_init_model:
        random_init_model.to(device=device)
        random_init_model_vecs = []
        for i in tqdm(range(num_vectors), file=sys.stdout, desc='Generate random initializations'):
            random_init_model.reset_parameters()
            random_init_model_vecs.append(
                torch.nn.utils.parameters_to_vector(random_init_model.parameters()))
        random_init_model_vecs = torch.stack(random_init_model_vecs).detach().to(device=device)
        erank_data.update({RAND_MODEL_INIT_KEY: []})

    # data generation loop
    model_idxes = torch.randperm(num_vectors)
    num_vec_idxes = list(range(1, len(model_idxes), model_steps))
    if not num_vectors in num_vec_idxes:
        num_vec_idxes.append(num_vectors)
    for i in tqdm(num_vec_idxes, file=sys.stdout, desc='Calculate eranks'):
        if random_baseline:
            erank_data[GAUSS_RAND_KEY].append(_erank(random_vecs[:i]))
        if random_init_model:
            erank_data[RAND_MODEL_INIT_KEY].append(_erank(random_init_model_vecs[:i]))

        for descr, model_matrix in models.items():
            erank_data[descr].append(_erank(model_matrix[model_idxes[:i]]))

    # plotting
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54))
    else:
        fig, ax = fig_ax

    for descr, erank in erank_data.items():
        ax.plot(num_vec_idxes, erank, label=descr, alpha=alpha)

    ax.set_xlabel('Number of vectors')
    ax.set_ylabel('Effective rank')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, mode='expand')

    return fig, ax
