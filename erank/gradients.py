from pathlib import Path
from typing import Dict, List, Tuple, Union
import torch
import pandas as pd
from omegaconf import DictConfig
from torch.utils import data
from torch import nn
from tqdm import tqdm

from ml_utilities.utils import get_device
from ml_utilities.torch_models import get_model_class
from ml_utilities.torch_utils import get_loss, gradients_to_vector
from ml_utilities.output_loader.model_loader import load_best_model, load_model_from_idx
from erank.data.datasetgenerator import DatasetGenerator


class GradientCalculator:

    def __init__(self,
                 dataset_generator_kwargs: DictConfig,
                 model_name: str = '',
                 model_path: Union[str, Path] = None,
                 run_path: Union[str, Path] = None,
                 model_idx: Union[int, List[int]] = -1,
                 default_loss: Union[str, nn.Module] = None,
                 device: Union[str, int] = 'auto'):
        """This class calculates local gradients at given model checkpoint(s). 
        If multiple model_idxes are specified, it can also load multiple models along a training trajectory. 

        Args:
            dataset_generator_kwargs (DictConfig): Config for dataset generation.
            model_name (str, optional): Typename of the model. Must be specified, if a model path is given. Defaults to ''.
            model_path (Union[str, Path], optional): A path to a model checkpoint. Defaults to None.
            run_path (Union[str, Path], optional): A path to a run directory of an earlier run from where a model checkpoint is loaded. Defaults to None.
            model_idx (Union[int, List[int]], optional): The model checkpoint(s) to be loaded. If -1 it loads the best model. Defaults to -1.
            default_loss (Union[str, nn.Module], optional): A default loss, such that the loss must not specified each time when gradients are computed. Defaults to None.
            device (Union[str, int], optional): The device. Defaults to 'auto'.
        """
        self.device = get_device(device)
        # load model
        if model_name and model_path:
            # load model directly from a path
            model_class = get_model_class(model_name)
            model = model_class.load(model_path, device=self._device)
            models = {-1: model}
        elif run_path:
            if isinstance(model_idx, list):
                # load multiple models from a single run
                models = {idx: load_model_from_idx(run_path, idx=model_idx, device=self.device) for idx in model_idx}
            else:
                # load single model from single run
                assert isinstance(model_idx, int)
                if model_idx == -1:
                    model = load_best_model(run_path, device=self.device)
                else:
                    model = load_model_from_idx(run_path, idx=model_idx, device=self.device)
                models = {-1: model}
        else:
            raise ValueError('No model provided!')
        self.models = models

        # generate dataset
        self.data_cfg = dataset_generator_kwargs
        dataset_generator = DatasetGenerator(**self.data_cfg)
        dataset_generator.generate_dataset()
        self.dataset = dataset_generator.train_split

        # create loss
        if default_loss:
            default_loss = self.__init_loss(default_loss)
        self.default_loss_fn = default_loss

    @property
    def model_idxes(self) -> List[int]:
        return list(self.models.keys())

    def __init_loss(self, loss: Union[str, nn.Module]):
        if isinstance(loss, str):
            loss_cls = get_loss(loss)
            loss_fn = loss_cls(reduction='mean')
        else:
            loss_fn = loss
        return loss_fn

    def compute_gradients(self,
                          batch_size: int,
                          num_gradients: int = -1,
                          loss: Union[str, nn.Module] = None,
                          model_idx: int = -1) -> List[torch.Tensor]:
        """Compute stochastic gradients with a given batch size.

        Args:
            batch_size (int): The batch size.
            num_gradients (int, optional): Number of gradients. If -1, gradients for full pass over dataset. Defaults to -1.
            loss (Union[str, nn.Module], optional): The loss for gradient calculation. If None use the default loss. Defaults to None.
            model_idx (int, optional): Specify the model idx to use.

        Returns:
            List[torch.Tensor]: The gradients.
        """
        assert model_idx in self.models, f'No model found with index {model_idx}! Possible model indices are {list(self.models.keys())}.'
        model = self.models[model_idx]
        model.zero_grad()

        dataloader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        loss_fn = self.default_loss_fn
        if loss_fn is None:
            assert not loss is None, 'No loss function given to compute the gradients!'
            loss_fn = self.__init_loss(loss)

        gradients = []

        it = range(num_gradients) if num_gradients > 0 else range(len(dataloader))
        data_iter = iter(dataloader)

        for batch_idx in tqdm(it):
            batch = next(data_iter)
            xs, ys = batch
            xs, ys = xs.to(self.device), ys.to(self.device)

            ys_pred = model(xs)
            loss = loss_fn(ys_pred, ys)
            model.zero_grad()
            loss.backward()

            grad = gradients_to_vector(model.parameters())
            gradients.append(grad)

        return gradients


class GradientAnalyzer:
    """This class collects all local gradient analysis methods."""

    def __init__(self, gradient_calculator: GradientCalculator):
        self.gradient_calculator = gradient_calculator

    def calculate_erank_vals(self, batch_sizes: List[int], model_idxs: List[int] = []) -> pd.DataFrame:
        pass


class GradientMasker:
    # TODO implement this
    pass


# def apply_gradient_mask():
#     # TODO make this a callable class
#     # find best way to store gradient mask:
#     # option a) store torch tensor where idx to mask are 0,
#     # option b) store parameter dict, [option c) store full model]
#     # probably best way is option b)
#     pass

### Utility functions


def magnitude_pruning_thresholds(vecs: torch.Tensor,
                                 frac_entries_to_prune: float,
                                 element_dim: int = 1) -> torch.Tensor:
    """Determine the threshold(s) for magnitude pruning with a given fraction.

    Args:
        vecs (torch.Tensor): The vectors stacked as a matrix. ndim=2. (If element_dim = 1, shape=(n_vecs, n_vec_elements))
        frac_entries_to_prune (float): The fraction of smallest entries to prune.
        element_dim (int, optional): The dimension of the vector entries. Defaults to 1.

    Returns:
        torch.Tensor: A tensor with the pruning threshold for each input vector.
    """
    assert 0. <= frac_entries_to_prune <= 1.
    assert vecs.ndim == 2
    vecs_abs = vecs.abs()
    # sort entries in the vector
    vec_abs_sorted, _ = vecs_abs.sort(dim=element_dim, descending=False)
    n_elements_to_prune = round(vecs.shape[element_dim] * frac_entries_to_prune) - 1
    # thresholds are at the index, given by the fraction to prune
    thresholds = vec_abs_sorted.select(dim=element_dim, index=n_elements_to_prune)
    thresholds.unsqueeze_(element_dim)
    return thresholds


def magnitude_prune_vectors(vecs: torch.Tensor,
                            frac_entries_to_prune: float,
                            element_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply magnitude pruning to vectors. 

    Args:
        vecs (torch.Tensor): The vectors stacked as a matrix. ndim=2. (If element_dim = 1, shape=(n_vecs, n_vec_elements))
        frac_entries_to_prune (float): The fraction of smallest entries to prune.
        element_dim (int, optional): The dimension of the vector entries. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Pruned vector, Positive mask (True if element is kept.), Thresholds 
    """
    if vecs.ndim == 1:
        vecs = vecs.unsqueeze(0)
    thresholds = magnitude_pruning_thresholds(vecs, frac_entries_to_prune, element_dim)
    pruning_mask_positive = vecs.abs() > thresholds
    pruned_vec = vecs.masked_fill(pruning_mask_positive.logical_not(), 0.0)
    return pruned_vec, pruning_mask_positive, thresholds


def intersectionOverUnion(mask_matrix: torch.Tensor, element_dim: int = 1) -> float:
    """Compute the intersection over union of the vectors in the mask matrix.

    Args:
        mask_matrix (torch.Tensor): Bool tensor (matrix) containing the masks in its dimension `element_dim`
        element_dim (int, optional): Element dimension of the mask. Mask dimension = 1 - element_dimension. Defaults to 1.

    Returns:
        float: Intersection over Union
    """
    assert mask_matrix.ndim == 2
    assert isinstance(element_dim, int) and 0 <= element_dim <= 1
    vec_dim = 1 - element_dim
    return (mask_matrix.all(dim=vec_dim).float().sum() / mask_matrix.any(dim=vec_dim).float().sum()).item()