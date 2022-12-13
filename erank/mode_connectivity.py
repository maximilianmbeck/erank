from collections import OrderedDict
from typing import Any, Dict
import copy
import torch
import numpy as np
from torch import nn
import torch.utils.data as data


class LinearInterpolator:

    def __init__(self, dataset: data.Dataset):
        pass

    def interpolate(self):
        pass


def interpolate_linear(model_0: nn.Module,
                       model_1: nn.Module,
                       train_dataset: data.Dataset,
                       score_fn: nn.Module,
                       weights: torch.Tensor,
                       other_datasets: Dict[str, data.Dataset] = {},
                       dataloader_kwargs: Dict[str, Any] = {'batch_size': 256}) -> Dict[str, Dict[str, float]]:
    get_device = lambda model: next(iter(model.parameters())).device
    assert get_device(model_0) == get_device(model_1), f'Models to interpolate not on same device!'
    device = get_device(model_0)
    
    def reset_bn_running_stats(module: nn.Module) -> None:
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()

    def eval_loop(model: nn.Module, dataloader: data.DataLoader, score_fn: nn.Module) -> float:
        batch_scores = []
        for batch_idx, xs, ys in enumerate(dataloader):
            xs, ys = xs.to(device), ys.to(device)
            with torch.no_grad():
                y_pred = model(xs)
                score = score_fn(y_pred, ys)
                batch_scores.append(score)
        return np.array(batch_scores).mean().item()


    # prepare datasets and results dict
    eval_datasets = copy.copy(other_datasets)  # shallow copy
    eval_datasets['train'] = train_dataset  # reference only
    res_dict = {ds_name: [] for ds_name in eval_datasets}
    res_dict['_weights'] = weights.tolist()

    # create eval_dataloaders
    eval_dataloaders = {ds_name: data.DataLoader(ds, **dataloader_kwargs) for ds_name, ds in eval_datasets.items()}
    train_loader = eval_dataloaders['train']

    # alpha = interpolation factor
    for alpha in weights:
        # create interpolated model in a memory friendly way (only use memory used for another model instance)
        interp_model_state_dict = OrderedDict()
        for k0, v0, k1, v1 in zip(model_0.state_dict().items(), model_1.state_dict().items()):
            assert k0 == k1, f'Model architectures do not match: {k0} != {k1}'
            interp_model_state_dict[k0] = torch.lerp(v0, v1, alpha) # linear interpolation between weights
        interp_model = copy.copy(model_0)
        interp_model.load_state_dict(interp_model_state_dict)

        # reset running stats
        interp_model.apply(reset_bn_running_stats)

        # compute batch_norm statistics on train_dataset
        interp_model.train(True)
        _ = eval_loop(model=interp_model, dataloader=train_loader, score_fn=score_fn)
        
        interp_model.train(False)
        # eval on eval_datasets
        # 

        pass

    # use torch.lerp()
    print('Done.')