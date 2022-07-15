import logging
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

LOGGER = logging.getLogger(__name__)

FMNIST_NORMALIZER = {'mean': [0.2860], 'std': [0.3205]}

def prepare_fashion_mnist(dataset_kwargs: DictConfig) -> data.Dataset:
    LOGGER.info('Prepare Fashion-MNIST dataset.')
    data_dir = Path(get_original_cwd()) / dataset_kwargs.dataset_dir
    normalizer = dataset_kwargs.normalizer
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(normalizer.mean, normalizer.std)])
    train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
    return train_dataset

def prepare_cifar10(dataset_kwargs: DictConfig) -> data.Dataset:
    LOGGER.info('Prepare CIFAR10 dataset.')
    pass
