from typing import Callable, Type
from omegaconf import DictConfig
import torch.utils.data as data
from erank.data.basemetadataset import BaseMetaDataset

from erank.data.data_preparation import prepare_cifar10, prepare_fashion_mnist
from erank.data.miniimagenetdataset import MiniImagenetDataset
from erank.data.omniglotdataset import OmniglotDataset
from erank.data.sinusdataset import SinusDataset

_dataset_registry = {'fashion_mnist': prepare_fashion_mnist, 'cifar10': prepare_cifar10}

_metadataset_registry = {'sinus': SinusDataset, 'omniglot': OmniglotDataset, 'mini-imagenet': MiniImagenetDataset}


def get_dataset_provider(dataset_name: str) -> Callable[[DictConfig], data.Dataset]:
    if dataset_name in _dataset_registry:
        return _dataset_registry[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_dataset_registry.keys())}"


def get_metadataset_class(dataset_name: str) -> Type[BaseMetaDataset]:
    if dataset_name in _metadataset_registry:
        return _metadataset_registry[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_metadataset_registry.keys())}"