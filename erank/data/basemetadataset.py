import sys
import torch
import logging
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
from torch.utils.data import IterableDataset, get_worker_info
from matplotlib.figure import Figure
from ml_utilities.data_utils import Scaler, DummyScaler, get_scaler

LOGGER = logging.getLogger(__name__)

DEFAULT_NORMALIZER = {'mean': [0.0], 'std': [1.0]}

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
        # for classification tasks the targets / labels are stored in tensors with dtype=torch.long and shape (batch_size,)
        assert len(
            x.shape) > 1 or x.dtype == torch.long, 'Tensor has too less dimensions. Maybe batch dimension is missing?'
        return x.to(device)

    return tuple(map(fun, set))


class Task(ABC):

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 support_set: Dict[str, Union[torch.Tensor, np.ndarray]] = {},
                 query_set: Dict[str, Union[torch.Tensor, np.ndarray]] = {},
                 regenerate_support_set: bool = False,
                 regenerate_query_set: bool = False,
                 rng: np.random.Generator = None,
                 normalizer: Scaler = DummyScaler()):
        self._rng = rng
        self._support_size = support_size
        self._query_size = query_size
        self.regenerate_support_set = regenerate_support_set
        self.regenerate_query_set = regenerate_query_set
        self._normalizer = copy.deepcopy(normalizer)
        # Tensors must have batch dimension
        self._support_data = support_set
        self._query_data = query_set

    @property
    def support_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.regenerate_support_set:
            self._generate_support_set()
        return self._normalizer(self._support_data[SUPPORT_X_KEY]), self._support_data[SUPPORT_Y_KEY]

    @property
    def query_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.regenerate_query_set:
            self._generate_query_set()
        return self._normalizer(self._query_data[QUERY_X_KEY]), self._query_data[QUERY_Y_KEY]

    @property
    def support_size(self) -> int:
        return self._support_size

    @property
    def query_size(self) -> int:
        return self._query_size

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _generate_support_set(self) -> None:
        """Generate and set the support_set."""
        pass

    def _generate_query_set(self) -> None:
        """Generate and set the query_set."""
        pass

    def _generate_sets(self) -> None:
        """Convenience function to ensure that support and query set are always generated in the correct order upon initialization."""
        self._generate_support_set()  # Mind the order!
        self._generate_query_set()

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

    def __lt__(self, other) -> bool:
        return self.name < other.name


class BaseMetaDataset(ABC, IterableDataset):
    """
    Baseclass for a Metadataset.
    """

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 num_tasks: int = -1,
                 regenerate_task_support_set: bool = True,
                 regenerate_task_query_set: bool = True,
                 seed: int = None,
                 normalizer: Dict[str, List[float]] = None):
        self._seed = seed
        self._rng: np.random.Generator = None
        self.reset_rng(seed=seed)

        # num_tasks -1 means infinite / specified by dataset
        self.num_tasks = num_tasks  # number of pregenerated tasks
        self.support_size = support_size
        self.query_size = query_size
        self.normalizer = get_scaler(normalizer)
        self.regenerate_task_support_set = regenerate_task_support_set
        self.regenerate_task_query_set = regenerate_task_query_set
        assert not (not regenerate_task_support_set and regenerate_task_query_set
                   ), f'Regenerating query set, but not the support set is not possible! Check your selected options!'

        # store pre-generated tasks, to be accessed via `get_tasks()`
        self.pregen_tasks: np.ndarray = None
        self.pregen_task_name_to_index: Dict[str, int] = None

    @abstractmethod
    def sample_tasks(self, num_tasks: int = -1) -> List[Task]:
        """Sample `num_tasks` tasks randomly. The tasks (and hence also the order) may be different on each call.

        Args:
            num_tasks (int, optional): The number of tasks to sample. If -1, sample all available tasks in random order. Defaults to -1.

        Returns:
            List[Task]: The sampled tasks.
        """
        pass

    def get_tasks(self, start_index: int = 0, num_tasks: int = -1) -> List[Task]:
        """Access to pregenerated tasks. Get `num_tasks` tasks in a deterministic way. The order and the tasks will always remain the same.
        If `num_tasks` is 5, it returns always the first five tasks. If `num_tasks` is 7, it returns the first 5 plus the next
        2 tasks.

        Args:
            start_index (int, optional): (Start) Index of task(s) to return. Defaults to 0.
            num_tasks (int, optional): Number of tasks to return. If -1, all available tasks are returned. Defaults to -1.
        
        Returns:
            List[Task]: The selected tasks.
        """
        if num_tasks == -1:
            num_tasks = len(self.pregen_tasks)
        return self.pregen_tasks[start_index:num_tasks].tolist()

    def create_pregen_tasks(self) -> None:
        """Generate `pregen_tasks`. Default behavior samples tasks randomly via `sample_tasks()`.
        Implement and call this method, when accessing tasks via `get_tasks()`."""
        tasks = self.sample_tasks(self.num_tasks)
        task_name_to_index = {task.name: i for i, task in enumerate(tasks)}
        self.pregen_tasks = np.array(tasks)
        self.pregen_tasks.sort()
        self.pregen_task_name_to_index = task_name_to_index

    def compute_normalizer(self) -> Dict[str, List[float]]:
        return {}

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
