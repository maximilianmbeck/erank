from typing import Dict, List, Tuple
import logging
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from erank.data.torchbuiltindatasets import get_default_normalizer, get_torch_dataset_class
from ml_utilities.data_utils import calculate_dataset_mean_std
from ml_utilities.utils import convert_dict_to_python_types

LOGGER = logging.getLogger(__name__)

NORMALIZER_TYPES = ('none', 'default', 'recompute')


class RotatedVisionDataset(data.Dataset):

    def __init__(self,
                 dataset: str,
                 data_root_path: str,
                 rotation_angle: float,
                 train: bool = True,
                 normalizer_type: str = 'default',
                 normalizer: Dict[str, List[float]] = {}):
        self._dataset_name = dataset
        self._dataset_class = get_torch_dataset_class(self._dataset_name)
        self._data_root_path = data_root_path
        self._rotation_angle = float(rotation_angle)
        self._train = train
        assert normalizer_type in NORMALIZER_TYPES
        LOGGER.info(f'Rotated vision dataset with {self._dataset_name} and rotation {self._rotation_angle}.')

        self._transforms = [
            transforms.ToTensor(),
            transforms.RandomRotation((rotation_angle, rotation_angle), interpolation=InterpolationMode.BILINEAR)
        ]
        normalizer_ = None
        if normalizer_type == 'recompute':
            normalizer_ = self.__compute_normalizer()
        elif normalizer_type == 'default':
            if normalizer:
                normalizer_ = normalizer_
            else:
                normalizer_ = get_default_normalizer(self._dataset_name)

        if normalizer_:
            self._transforms.append(transforms.Normalize(normalizer_['mean'], normalizer_['std']))

        transform = transforms.Compose(self._transforms)
        self._dataset = self._dataset_class(root=self._data_root_path,
                                            train=self._train,
                                            transform=transform,
                                            download=False)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)

    def __compute_normalizer(self) -> Dict[str, List[float]]:
        LOGGER.info('Recomputing normalizer..')
        transform = transforms.Compose(self._transforms)
        unnormalized_dataset = self._dataset_class(root=self._data_root_path,
                                                   train=self._train,
                                                   transform=transform,
                                                   download=False)
        mean, std = calculate_dataset_mean_std(unnormalized_dataset)
        normalizer = dict(mean=mean, std=std)
        normalizer = convert_dict_to_python_types(normalizer, single_vals_as_list=True)
        LOGGER.info(f'using normalizer: {normalizer}')
        return normalizer