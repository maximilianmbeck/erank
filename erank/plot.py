from typing import Dict, Tuple, Callable, Union
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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


def plot_models_erank(erank_df: pd.DataFrame,
                      column_name_to_plot_kwargs: Dict[str, str] = {},
                      ax_title: str = None,
                      fig_title: str = None,
                      alpha=0.8,
                      fig_ax: Tuple[mpl.figure.Figure, mpl.axis.Axis] = None):
   

    # plotting
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2 * 12 * 1 / 2.54, 2 * 5 * 1 / 2.54))
    else:
        fig, ax = fig_ax

    for column_name in erank_df:
        if column_name in column_name_to_plot_kwargs:
            plt_kwargs = column_name_to_plot_kwargs[column_name]
        else:
            plt_kwargs = {'label': column_name}
        erank = erank_df[column_name]
        ax.plot(erank.index.to_list(), erank.to_list(), **plt_kwargs, alpha=alpha)

    ax.set_xlabel('Number of vectors')
    ax.set_ylabel('Effective rank')
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, mode='expand')
    return fig, ax
