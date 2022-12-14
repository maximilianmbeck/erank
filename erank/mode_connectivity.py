from collections import OrderedDict
from typing import Any, Dict, List, Union
import sys
import copy
import torch
from torch import nn
from torchmetrics import Metric
import torch.utils.data as data
from tqdm import tqdm


class LinearInterpolator:

    def __init__(self, dataset: data.Dataset):
        pass

    def interpolate(self):
        pass


def interpolate_linear(model_0: nn.Module,
                       model_1: nn.Module,
                       train_dataset: data.Dataset,
                       score_fn: Union[nn.Module, Metric],
                       other_datasets: Dict[str, data.Dataset] = {},
                       interpolation_factors: torch.Tensor = torch.linspace(0.0, 1.0, 5),
                       dataloader_kwargs: Dict[str, Any] = {'batch_size': 256},
                       compute_model_distances: bool = True,
                       interpolation_on_train_data: bool = False) -> Dict[str, Dict[str, float]]:
    """Interpolate linearly between two models. Evaluates the performance of each interpolated model on given datasets.
    
    Note:
        Also computes the instability value according to Frankle et al., 2020, p. 3.
        Instability = max/min [interpolation_scores] - mean[interpolation_score(0.0), interpolation_score(1.0)]
    
    References:
        Frankle, Jonathan, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. 2020. 
            “Linear Mode Connectivity and the Lottery Ticket Hypothesis.” arXiv. http://arxiv.org/abs/1912.05671.

    Args:
        model_0 (nn.Module): First model.
        model_1 (nn.Module): Second model.
        train_dataset (data.Dataset): Dataset on which the models have been trained. 
                                      If applicable, this dataset is used to recompte batch norm statistics.
        score_fn (Union[nn.Module, Metric]): The performance measure on which each model is used.
        other_datasets (Dict[str, data.Dataset], optional): Evaluation dataset with descriptor as key. Defaults to {}.
        interpolation_factors (torch.Tensor, optional): Interpolation factor for linear interpolation. Defaults to torch.linspace(0.0, 1.0, 5).
        dataloader_kwargs (Dict[str, Any], optional): Additional dataloader keyword arguments. Defaults to {'batch_size': 256}.
        compute_model_distances (bool, optional): Computes distance metrics on given models. Defaults to True.
        interpolation_on_train_data (bool, optional): Evaluates interpolation performance on train data too. Defaults to False.

    Raises:
        ValueError: If no eval datasets are given or the model architectures do not match.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing the results.
                                     Example:
                                        {'__weights': [0.0,
                                                       1.0],
                                         '_cosinesimilarity': 0.010202181525528431,
                                         '_l2distance': 30.355226516723633,
                                         'val': {'instability': -0.08413612842559814,
                                                 'interpolation_scores': [0.974958598613739,
                                                                          0.976451575756073]}}
    """
    get_device = lambda model: next(iter(model.parameters())).device
    assert get_device(model_0) == get_device(model_1), f'Models to interpolate not on same device!'
    device = get_device(model_0)
    assert 'train' not in other_datasets, f'`train` is a reserved dataset name. Please rename this evaluation dataset.'
    assert interpolation_factors.dim() == 1, '`interpolation_factors` must be tensor of dimension 1.'
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def reset_bn_running_stats(module: nn.Module) -> None:
        if isinstance(module, bn_types):
            module.reset_running_stats()

    def eval_loop(model: nn.Module, dataloader: data.DataLoader, score_fn: nn.Module) -> float:
        batch_scores = []
        for batch_idx, (xs, ys) in enumerate(dataloader):
            xs, ys = xs.to(device), ys.to(device)
            with torch.no_grad():
                y_pred = model(xs)
                score = score_fn(y_pred, ys)
                batch_scores.append(score)
        return torch.tensor(batch_scores).mean().item()

    # check if models have batch_norm layers
    models_have_batch_norm_layers = False
    for m in model_0.modules():
        if isinstance(m, bn_types):
            models_have_batch_norm_layers = True
            break

    # prepare datasets and results dict
    eval_datasets = copy.copy(other_datasets)  # shallow copy
    if interpolation_on_train_data:
        eval_datasets['train'] = train_dataset  # reference only
    res_dict = {ds_name: [] for ds_name in eval_datasets}
    res_dict['__weights'] = interpolation_factors.tolist()

    # create eval_dataloaders
    eval_dataloaders = {ds_name: data.DataLoader(ds, **dataloader_kwargs) for ds_name, ds in eval_datasets.items()}
    if not eval_dataloaders:
        raise ValueError(
            'No evaluation datasets provided. Pass eval_datasets or set `interpolation_on_train_data=True`.')

    interpolation_factors = interpolation_factors.to(device)
    score_fn = score_fn.to(device)
    # alpha = interpolation factor
    for alpha in tqdm(interpolation_factors, desc=f'Interp. factors', file=sys.stdout):
        # create interpolated model in a memory friendly way (only use memory used for another model instance)
        interp_model = copy.deepcopy(model_0)
        interp_model_state_dict = interp_model.state_dict()
        for (k0, v0), (k1, v1) in zip(model_0.state_dict().items(), model_1.state_dict().items()):
            if k0 != k1:
                raise ValueError(f'Model architectures do not match: {k0} != {k1}')
            torch.lerp(v0, v1, alpha, out=interp_model_state_dict[k0])  # linear interpolation between weights
        interp_model.load_state_dict(interp_model_state_dict)

        if models_have_batch_norm_layers:
            # reset running stats
            interp_model.apply(reset_bn_running_stats)
            # compute batch_norm statistics on train_dataset
            train_loader = data.DataLoader(train_dataset, **dataloader_kwargs)
            interp_model.train(True)
            _ = eval_loop(model=interp_model, dataloader=train_loader, score_fn=score_fn)

        interp_model.train(False)
        # eval on eval_datasets
        for ds_name, dataloader in eval_dataloaders.items():
            score = eval_loop(model=interp_model, dataloader=dataloader, score_fn=score_fn)
            res_dict[ds_name].append(score)

    if compute_model_distances:
        vec_0 = nn.utils.parameters_to_vector(model_0.parameters())
        vec_1 = nn.utils.parameters_to_vector(model_1.parameters())
        # L2 distance
        res_dict['_l2distance'] = torch.linalg.norm(vec_1 - vec_0).item()
        # cosine similarity
        res_dict['_cosinesimilarity'] = nn.functional.cosine_similarity(vec_0, vec_1, dim=0).item()

    # compute instability value
    # find weight indices for base models
    # (necessary if 0. and 1. value not and beginning or end of interpolation_factors tensor)
    original_models_in_interpolation_factors = True
    base_interpolation_factors = [0.0, 1.0]
    interp_factors_list = interpolation_factors.tolist()
    base_interpolation_factor_idxes = []
    for f in base_interpolation_factors:
        try:
            base_interpolation_factor_idxes.append(interp_factors_list.index(f))
        except ValueError:
            original_models_in_interpolation_factors = False
            break

    # compute instability per dataset
    def compute_instability(interp_scores: List[float]) -> float:
        if original_models_in_interpolation_factors:
            interp_scores = torch.tensor(interp_scores)
            base_mean = interp_scores[base_interpolation_factor_idxes].mean()
            torch_minmax = torch.min if score_fn.higher_is_better else torch.max
            instability = torch_minmax(interp_scores) - base_mean
            return instability.item()
        else:
            return float('nan')

    for k, v in res_dict.items():
        if not k.startswith('_'):
            # k is ds_name, v is interpolation_scores
            res_dict[k] = {'instability': compute_instability(v), 'interpolation_scores': v}

    return res_dict