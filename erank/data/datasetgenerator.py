from typing import Any, Dict
import logging
import torch.utils.data as data
from abc import ABC, abstractmethod

from erank.data import get_dataset_class
from erank.data.data_utils import random_split_train_tasks
from erank.data.datasettransformer import DatasetTransformer

LOGGER = logging.getLogger(__name__)


class DatasetGeneratorInterface(ABC):

    def generate_dataset(self) -> None:
        pass

    @property
    @abstractmethod
    def train_split(self) -> data.Dataset:
        pass

    @property
    @abstractmethod
    def val_split(self) -> data.Dataset:
        pass


class DatasetGenerator(DatasetGeneratorInterface):
    """A class that generates datasets and its splits trough a common interface.
    
    Args:
        dataset (str): Dataset name.
        dataset_kwargs (Dict[str, Any]): Keyword args for the dataset.
        dataset_split (Dict[str, Any], optional): Keyword args for splitting the dataset. 
                                                  If not provided, the full dataset can be accessed via `val_split`. Defaults to {}.
        train_split_transforms (Dict[str, Any], optional): Keyword args for adding transformation to the train split of the dataset.
                                                             If not provided, just try to normalize the dataset. Defaults to {}.
        val_split_transforms (Dict[str, Any], optional): Keyword args for adding transformation to the val split of the dataset.
                                                           If not provided, just try to normalize the dataset. Defaults to {}.
    """

    def __init__(self,
                 dataset: str,
                 dataset_kwargs: Dict[str, Any],
                 dataset_split: Dict[str, Any] = {},
                 train_split_transforms: Dict[str, Any] = {},
                 val_split_transforms: Dict[str, Any] = {}):
        self.dataset = dataset
        self.dataset_class = get_dataset_class(dataset_name=dataset)
        self.dataset_kwargs = dataset_kwargs
        self.dataset_split = dataset_split
        self.train_dataset_tansforms = train_split_transforms
        self.val_dataset_transforms = val_split_transforms
        self._train_split = None
        self._val_split = None
        self._dataset_generated = False

    def generate_dataset(self) -> None:
        LOGGER.info(f'Generating dataset: {self.dataset}')
        dataset = self.dataset_class(**self.dataset_kwargs)
        self._train_split = None
        self._val_split = dataset
        if self.dataset_split:
            self._train_split, self._val_split = random_split_train_tasks(dataset=dataset, **self.dataset_split)

        # add data normalization and augmentations
        self._train_split = DatasetTransformer.create(self._train_split, **self.train_dataset_tansforms)
        # add only normalization
        self._val_split = DatasetTransformer.create(self._val_split, **self.val_dataset_transforms)

        self._dataset_generated = True

    @property
    def dataset_generated(self) -> bool:
        return self._dataset_generated

    @property
    def train_split(self) -> data.Dataset:
        return self._train_split

    @property
    def val_split(self) -> data.Dataset:
        return self._val_split
