import logging
from typing import Dict, List, Type
from torch import normal
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

LOGGER = logging.getLogger(__name__)

FMNIST_NORMALIZER = {'mean': [0.2860], 'std': [0.3205]}
CIFAR10_NORMALIZER = {'mean': [0.4913995563983917, 0.48215848207473755, 0.44653093814849854],
                      'std': [0.20230084657669067, 0.19941289722919464, 0.20096157491207123]}

_torch_dataset_classes = {'fashion_mnist': torchvision.datasets.FashionMNIST, 'cifar10': torchvision.datasets.CIFAR10}


def prepare_fashion_mnist(dataset_kwargs: DictConfig) -> data.Dataset:
    LOGGER.info('Prepare Fashion-MNIST dataset.')
    return _prepare_torchdataset(torchvision.datasets.FashionMNIST, dataset_kwargs, FMNIST_NORMALIZER)


def prepare_cifar10(dataset_kwargs: DictConfig) -> data.Dataset:
    LOGGER.info('Prepare CIFAR10 dataset.')
    return _prepare_torchdataset(torchvision.datasets.CIFAR10, dataset_kwargs, CIFAR10_NORMALIZER)


def _prepare_torchdataset(
        torch_dataset_class: Type[data.Dataset],
        dataset_kwargs: DictConfig, default_normalizer: Dict[str, List[float]]) -> data.Dataset:
    data_dir = Path(get_original_cwd()) / dataset_kwargs.dataset_dir
    normalizer = dataset_kwargs.get('normalizer', None)
    if normalizer is None:
        LOGGER.info(f'Using default normalizer: {default_normalizer}')
        normalizer = default_normalizer
    else:
        LOGGER.info('Using custom normalizer.')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(normalizer['mean'], normalizer['std'])])
    train_dataset = torch_dataset_class(root=data_dir, train=True, transform=transform, download=True)
    return train_dataset
