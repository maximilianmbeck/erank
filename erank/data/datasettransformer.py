import torch
import torchvision
from typing import Any, Dict, List, Tuple, Callable
from torch.utils.data import Dataset
from erank.data.basedataset import BaseDataset
from erank.data.transforms import create_transform


class DatasetTransformer(Dataset):

    def __init__(self,
                 dataset: Dataset,
                 image_transforms: List[Callable] = [],
                 tensor_transforms: List[Callable] = [],
                 joint_tensor_transforms: List[Callable] = [],
                 enable_transforms: bool = True):
        self.dataset = dataset

        self._image_transforms = image_transforms if enable_transforms else []
        self._tensor_transforms = tensor_transforms if enable_transforms else []
        self._joint_tensor_transforms = joint_tensor_transforms if enable_transforms else []

        # find dataset normalizer
        # account for the case that the dataset is a Subset (e.g. when a train_val split is applied)
        if isinstance(dataset, BaseDataset):
            base_dataset = dataset  
        else:
            ds = dataset
            # dataset can be multiple recursive subsets, go along all Subsets
            while True:
                ds = getattr(ds, 'dataset', None)
                if ds is None or isinstance(ds, BaseDataset):
                    break
            base_dataset = ds
        if base_dataset is None or not isinstance(base_dataset, BaseDataset):
            raise ValueError(f'Could not find a `BaseDataset` instance for {dataset}.')

        # add dataset normalizer
        if base_dataset.normalizer:
            self._tensor_transforms.append(base_dataset.normalizer)

        self._composed_image_tensor_transforms = torchvision.transforms.Compose(self._image_transforms +
                                                                                [torchvision.transforms.ToTensor()] +
                                                                                self._tensor_transforms)

    @property
    def image_tensor_transforms(self) -> Callable:
        return self._composed_image_tensor_transforms

    @property
    def joint_tensor_transforms(self) -> List[Callable]:
        return self._joint_tensor_transforms

    def transform(self, input, target=None) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self._composed_image_tensor_transforms(input)

        for joint_tensor_transform in self._joint_tensor_transforms:
            input, target = joint_tensor_transform(input, target)

        return input, target

    def get_raw_item(self, index) -> Tuple[Any, Any]:
        return self.dataset[index]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.dataset[index]
        transformed = self.transform(*raw)
        return transformed

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def create(dataset: Dataset,
               image_transforms: Dict[str, Any] = {},
               tensor_transforms: Dict[str, Any] = {},
               joint_tensor_transforms: Dict[str, Any] = {},
               enable_transforms: bool = True) -> 'DatasetTransformer':
        if dataset is None:
            return None
        
        it = [create_transform(t) for t in image_transforms] if image_transforms else []
        tt = [create_transform(t) for t in tensor_transforms] if tensor_transforms else []
        jtt = [create_transform(t) for t in joint_tensor_transforms] if joint_tensor_transforms else []

        return DatasetTransformer(dataset,
                                  image_transforms=it,
                                  tensor_transforms=tt,
                                  joint_tensor_transforms=jtt,
                                  enable_transforms=enable_transforms)
