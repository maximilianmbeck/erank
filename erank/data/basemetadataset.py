import sys
import torch
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
from torch.utils.data import IterableDataset, get_worker_info
from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)

SUPPORT_X_KEY = QUERY_X_KEY = 'x'
SUPPORT_Y_KEY = QUERY_Y_KEY = 'y'


def support_query_as_minibatch(set: Tuple[torch.Tensor, torch.Tensor],
                               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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


class Task(ABC):

    def __init__(self,
                 support_set: Dict[str, Union[torch.Tensor, np.ndarray]] = {},
                 query_set: Dict[str, Union[torch.Tensor, np.ndarray]] = {},
                 rng: np.random.Generator = None):
        self._rng = rng
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

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def plot_query_predictions(self, epoch: int, preds: Dict[int, torch.Tensor]) -> Tuple[Figure, str]:
        """Make a figure comparing the predictions on the query set before and after learning on the support set.

        Args:
            epoch (int): The current learning epoch. 
            preds (Dict[int, torch.Tensor]): Predictions on query set of the model after n (the key) 
                steps of finetuning on the support set.

        Returns:
            Tuple[Figure, str]: The matplotlib Figure and its filename.
        """
        return None

    def __str__(self) -> str:
        return self.name


class BaseMetaDataset(ABC, IterableDataset):
    """
    Baseclass for a Metadataset.
    """

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 num_tasks: int = -1,
                 seed: int = None,
                 normalizer: Dict[str, List[float]] = None):
        # num_tasks -1 means infinite / specified by dataset
        self.num_tasks = num_tasks
        self.support_size = support_size
        self.query_size = query_size
        self.normalizer = normalizer
        self._seed = seed
        self._rng: np.random.Generator = None
        self.reset_rng(seed=seed)

    @abstractmethod
    def sample_tasks(self, num_tasks: int = -1) -> List[Task]:
        """Sample `num_tasks` tasks randomly. The tasks (and hence also the order) may be different on each call.

        Args:
            num_tasks (int, optional): The number of tasks to sample. If -1, sample all available tasks in random order. Defaults to -1.

        Returns:
            List[Task]: The sampled tasks.
        """
        pass

    @abstractmethod
    def get_tasks(self, num_tasks: int = -1) -> List[Task]:
        """Get `num_tasks` tasks in a deterministic way. The order and the tasks will always remain the same.
        If `num_tasks` is 5, it returns always the first five tasks. If `num_tasks` is 7, it returns the first 5 plus the next
        2 tasks.

        Args:
            num_tasks (int, optional): Number of tasks to return. If -1, all available tasks are returned. Defaults to -1.

        Returns:
            List[Task]: The selected tasks.
        """
        pass

    @abstractmethod
    def get_task(self, task_name: str) -> Task:
        """Get a task by name. `name` is defined by the property `name` of the `Task` class.

        Args:
            task_name (str): The task name.

        Returns:
            Task: The task.
        """
        pass

    def reset_rng(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed=seed)

    def __iter__(self):
        #* init worker
        # reset random number generater here, such that we get a different task order in each worker process
        # Result: every worker draws different task order. (Otherwise every task will be sampled four times.)
        # worker_info object contains worker attributes, when called in a worker. None otherwise.
        worker_info = get_worker_info()
        if worker_info:
            new_seed = self._seed + worker_info.id
            LOGGER.info(f'Spawning worker {worker_info}: Setting seed={new_seed}')
            self.reset_rng(new_seed)

        #* data sampling loop
        while True:
            yield self.sample_tasks(num_tasks=1)[0]

    def __len__(self) -> int:
        if self.num_tasks == -1:
            return sys.maxsize
        return self.num_tasks
