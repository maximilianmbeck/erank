import logging
from typing import Callable, Dict, List, Tuple, Type, Union
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path

LOGGER = logging.getLogger(__name__)

FMNIST_NORMALIZER = {'mean': [0.2860], 'std': [0.3205]}
MNIST_NORMALIZER = {'mean': [0.1306605190038681], 'std': [0.3015042245388031]}
CIFAR10_NORMALIZER = {
    'mean': [0.4913995563983917, 0.48215848207473755, 0.44653093814849854],
    'std': [0.20230084657669067, 0.19941289722919464, 0.20096157491207123]
}

_torch_dataset_classes = {'fashion_mnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10, 'mnist': datasets.MNIST}
_default_normalizers = {'fashion_mnist': FMNIST_NORMALIZER, 'cifar10': CIFAR10_NORMALIZER, 'mnist': MNIST_NORMALIZER}

def get_torch_dataset_class(dataset_name: str) -> Type[datasets.VisionDataset]:
    if dataset_name in _torch_dataset_classes:
        return _torch_dataset_classes[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_torch_dataset_classes.keys())}"

def get_default_normalizer(dataset_name: str) -> Callable:
    if dataset_name in _default_normalizers:
        return _default_normalizers[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_default_normalizers.keys())}"

def _prepare_torchdataset(torch_dataset: str,
                          data_root_path: str,
                          train: bool = True,
                          normalizer: Dict[str, Union[float, List[float]]] = None) -> data.Dataset:
    assert Path(data_root_path).is_absolute()
    data_dir = Path(data_root_path)
    if normalizer is None:
        default_normalizer = _default_normalizers[torch_dataset]
        LOGGER.info(f'Using default normalizer: {default_normalizer}')
        normalizer = default_normalizer
    else:
        LOGGER.info('Using custom normalizer.')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normalizer['mean'], normalizer['std'])])
    torch_dataset_class = _torch_dataset_classes[torch_dataset]
    train_dataset = torch_dataset_class(root=data_dir, train=train, transform=transform, download=True)
    return train_dataset


class TorchBuiltInDataset(data.Dataset):

    def __init__(self,
                 dataset: str,
                 data_root_path: Union[str, Path],
                 train: bool = True,
                 normalizer: Dict[str, Union[float, List[float]]] = None):
        self._dataset = _prepare_torchdataset(torch_dataset=dataset,
                                              data_root_path=data_root_path,
                                              train=train,
                                              normalizer=normalizer)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)


class TorchMnist(TorchBuiltInDataset):

    def __init__(self, data_root_path: Union[str, Path], train: bool = True):
        super().__init__('mnist', data_root_path, train)


class TorchFmnist(TorchBuiltInDataset):

    def __init__(self, data_root_path: Union[str, Path], train: bool = True):
        super().__init__('fashion_mnist', data_root_path, train)


class TorchCifar10(TorchBuiltInDataset):

    def __init__(self, data_root_path: Union[str, Path], train: bool = True):
        super().__init__('cifar10', data_root_path, train)