from typing import Dict, List, Tuple
import torch
import torch.utils.data as data

from erank.data.torchbuiltindatasets import get_torch_dataset_class

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
        self._rotation_angle = rotation_angle
        self._train = train
        assert normalizer_type in NORMALIZER_TYPES


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def __len__(self):
        pass

    def _compute_normalizer(self) -> Dict[str, List[float]]:
        # TODO from here
        unnormalized_dataset = self._dataset_class(root=self._data_root_path, train=self._train, )
        mean, std = calculate_dataset_mean_std(unnormalized_dataset)
        normalizer = dict(mean=mean, std=std)
        normalizer = convert_dict_to_python_types(normalizer, single_vals_as_list=True)
        normalizer