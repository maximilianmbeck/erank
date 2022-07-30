from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple

import torch

SUPPORT_X_KEY = QUERY_X_KEY = 'x'
SUPPORT_Y_KEY = QUERY_Y_KEY = 'y'

def support_query_as_minibatch(set: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Checks dimensions of tensors in the set and moves them to the given device.

    Args:
        set (Tuple[torch.Tensor, torch.Tensor]): support or query set
        device (torch.device): 

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: set on correct device.
    """
    def fun(x):
        assert len(x.shape) > 1, 'Tensor as too less dimensions. Maybe batch dimension missing?'
        return x.to(device)
    return tuple(map(fun, set))

class Task(object):

    def __init__(self, support_set: Dict[str, torch.Tensor] = {}, query_set: Dict[str, torch.Tensor] = {}):
        # Tensors must have batch dimension
        self._support_data = support_set
        self._query_data = query_set

    @property
    def support_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._support_data[SUPPORT_X_KEY], self._support_data[SUPPORT_Y_KEY]

    @property
    def query_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._query_data[QUERY_X_KEY], self._query_data[QUERY_Y_KEY]

    @property
    def support_size(self) -> int:
        return self._support_data[SUPPORT_X_KEY].shape[0]

    @property
    def query_size(self) -> int:
        return self._query_data[QUERY_X_KEY].shape[0]


class BaseMetaDataset(object):
    """
    TODO take care of normalization of inputs

    """

    def __init__(self, support_size: int, query_size: int, num_tasks: int = -1):
        # num_tasks -1 means infinite / specified by dataset
        self.num_tasks = num_tasks
        self.support_size = support_size
        self.query_size = query_size

    @abstractmethod
    def sample_tasks(self, num_tasks: int) -> Iterable[Task]:
        pass

    @abstractmethod
    def get_tasks(self, num_tasks: int = -1) -> List[Task]:
        pass

    def __len__(self) -> int:
        self.num_tasks
