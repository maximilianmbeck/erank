from typing import Callable
from omegaconf import DictConfig
import torch.utils.data as data

from erank.data.data_preparation import prepare_cifar10, prepare_fashion_mnist

_dataset_registry = {'fashion_mnist': prepare_fashion_mnist, 'cifar10': prepare_cifar10}

def get_dataset_provider(dataset_name: str) -> Callable[[DictConfig], data.Dataset]:
    if dataset_name in _dataset_registry:
        return _dataset_registry[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_dataset_registry.keys())}"