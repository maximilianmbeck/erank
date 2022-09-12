from typing import Dict, List, Set, Tuple, Union
import torch
import math
import copy
import logging
import itertools
from abc import abstractmethod
from pathlib import Path
import numpy as np
from erank.data.basemetadataset import QUERY_X_KEY, QUERY_Y_KEY, SUPPORT_X_KEY, SUPPORT_Y_KEY, BaseMetaDataset, Task
from ml_utilities.utils import convert_to_simple_str
from ml_utilities.data_utils import Scaler

LOGGER = logging.getLogger(__name__)
TASK_NAME_SEPARATOR = '#'
CHANNEL_DIM = 1
BATCH_DIM = 0


class ClassificationTask(Task):

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 dataset_name: str,
                 task_data: Dict[str, np.ndarray],
                 rng: np.random.Generator,
                 normalizer: Scaler,
                 regenerate_support_set: bool = True,
                 regenerate_query_set: bool = False):
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         regenerate_support_set=regenerate_support_set,
                         regenerate_query_set=regenerate_query_set,
                         rng=rng, 
                         normalizer=normalizer)
        self._dataset_name = dataset_name
        self._task_data = task_data  # {class_name: data_samples}
        self._task_labels = self._generate_labels()  # {class_name: label_idx}

        init_idxs = {class_name: None for class_name in self._task_data}
        self._support_idxes: Dict[str, np.ndarray] = copy.deepcopy(init_idxs)
        self._query_idxes: Dict[str, np.ndarray] = copy.deepcopy(init_idxs)

        self._generate_sets()

    def _generate_labels(self) -> Dict[str, int]:
        return {class_name: label for label, class_name in enumerate(self._task_data)}

    def _generate_support_set(self) -> None:
        support_x, support_y, self._support_idxes = self._sample_set(set_specifier='support',
                                                                     set_size=self.support_size,
                                                                     exclude_idxes=self._query_idxes)
        self._support_data[SUPPORT_X_KEY] = support_x
        self._support_data[SUPPORT_Y_KEY] = support_y

    def _generate_query_set(self) -> None:
        query_x, query_y, self._query_idxes = self._sample_set(set_specifier='query',
                                                               set_size=self.query_size,
                                                               exclude_idxes=self._support_idxes)
        self._query_data[QUERY_X_KEY] = query_x
        self._query_data[QUERY_Y_KEY] = query_y

    def _sample_set(self, set_specifier: str, set_size: int,
                    exclude_idxes: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
        """Sample support or query set. Make sure sampling is non-overlapping.

        Args:
            set_specifier (str): `support` or `query`
            set_size (int): 
            exclude_idxes (Dict[str, np.ndarray]): The indices of the data samples to exclude (as they are already in the other set).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]: Data and labels for the set; Set indices per class.
        """
        # for every class in task_data, sample set_size data samples
        # generate two tensors: set_x (images), set_y (labels)
        set_x = []
        set_y = []
        set_idxes = {}
        for class_name, data_samples in self._task_data.items():
            #* sample set idxes
            idxes = np.arange(len(data_samples))

            if not exclude_idxes[class_name] is None:
                if (set_specifier == 'support') or (set_specifier == 'query'):
                    # sample set only from samples excluding the other set's samples
                    idxes = np.setdiff1d(idxes, exclude_idxes[class_name], assume_unique=True)

            self._rng.shuffle(idxes)
            set_class_idxes = idxes[:set_size]
            set_idxes[class_name] = set_class_idxes

            #* create data tensors
            x_data = self._task_data[class_name][set_class_idxes]
            y_data = np.repeat(self._task_labels[class_name], set_size)

            set_x.append(torch.tensor(x_data, dtype=torch.float32))
            set_y.append(torch.tensor(y_data, dtype=torch.long))

        set_x = torch.cat(set_x)  # shape: (batch, data_dims), for image data: data_dims=CxHxW
        set_y = torch.cat(set_y)  # shape: (batch,)
        return set_x, set_y, set_idxes

    @property
    def task_classes(self) -> List[str]:
        return list(self._task_data.keys())

    @property
    def n_way(self) -> int:
        return len(self.task_classes)

    @property
    def name(self) -> str:
        return convert_to_simple_str(self.task_classes, separator=TASK_NAME_SEPARATOR)

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _generate_sets(self) -> None:
        super()._generate_sets()
        # check for non-overlapping data samples
        for task_class in self.task_classes:
            assert len(
                np.intersect1d(self._support_idxes[task_class], self._query_idxes[task_class], assume_unique=True)) == 0


class BaseMetaClassificationDataset(BaseMetaDataset):

    def __init__(self,
                 data_root_path: Union[str, Path],
                 n_way_classification: int,
                 support_size: int,
                 query_size: int,
                 split: str,
                 num_tasks: int = -1,
                 regenerate_task_support_set: bool = True,
                 regenerate_task_query_set: bool = True,
                 seed: int = None,
                 normalizer: Dict[str, List[float]] = None):
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         num_tasks=num_tasks,
                         regenerate_task_support_set=regenerate_task_support_set,
                         regenerate_task_query_set=regenerate_task_query_set,
                         seed=seed,
                         normalizer=normalizer)
        #* public attributes
        self.split = split
        self._n_way_classification = n_way_classification
        assert self._n_way_classification > 1, f'Need at least two classes for classification.'
        self.dataset_path: Path = None

        #* private attributes
        # check data path
        if isinstance(data_root_path, str):
            data_root_path = Path(data_root_path)
        assert data_root_path.is_absolute(), f'Data root path `{data_root_path}` is not an absolute path!'
        self._data_root_path = data_root_path

        # make sure dict is sorted, to ensure deterministic behavior (dict(sorted(unsorted_dict.items())))
        # {class_name: class samples}, class_samples.shape = (n_samples, data_dim), data_dim = Channelx???
        self._data: Dict[str, np.ndarray] = None

    @property
    def dataset_classes(self) -> List[str]:
        return list(self._data.keys())

    @property
    def n_way(self) -> int:
        return self._n_way_classification

    @property
    def n_way_combinations(self) -> int:
        return int(
            math.factorial(len(self.dataset_classes)) /
            (math.factorial(self.n_way) * math.factorial(len(self.dataset_classes) - self.n_way)))

    @abstractmethod
    def _load_data(self, split: str) -> Dict[str, np.ndarray]:
        """Load all data from disk into memory."""
        pass

    def sample_tasks(self, num_tasks: int = 1) -> List[Task]:
        assert num_tasks <= self.n_way_combinations, f'Trying to sample more tasks ({num_tasks}) than available ({self.n_way_combinations})!'
        tasks: List[ClassificationTask] = []
        if num_tasks <= 0:
            return tasks

        task_set: Set[Tuple[str, ...]] = set()

        # sample `num_tasks` combinations without replacement, no combination twice within the set
        # sample random combinations on each call
        ds_classes = np.array(list(self._data.keys()))
        self._rng.shuffle(ds_classes)
        # draw n_way classes randomly from all classes
        for i in range(num_tasks):
            # draw classes without replacement, no class twice within a task
            task_classes = self._rng.choice(ds_classes, size=self.n_way, replace=False, shuffle=False)
            # avoid duplicate samples (don't shuffle samples)
            task_set.add(tuple(task_classes))

        if len(task_set) < num_tasks:
            # if this happens the number of possible tasks is probably not much higher than num_tasks
            LOGGER.warning(
                f'Naive sampling yielded less than {num_tasks} tasks. Filling up specified tasks systematically.')
            task_iter = itertools.combinations(ds_classes, self.n_way)
            while len(task_set) < num_tasks:
                task_set.add(next(task_iter))

        assert len(task_set) == num_tasks

        # create tasks
        for task_classes in task_set:
            task = self._create_task(task_classes)
            tasks.append(task)

        return tasks

    def _create_task(self, task_classes: Tuple[str, ...]) -> ClassificationTask:
        ds_name = self.__class__.name
        # collect task_data, no copy for now
        task_data = {task_class: self._data[task_class] for task_class in sorted(task_classes)}

        task = ClassificationTask(support_size=self.support_size,
                                  query_size=self.query_size,
                                  dataset_name=ds_name,
                                  task_data=task_data,
                                  rng=self._rng,
                                  normalizer=self.normalizer,
                                  regenerate_support_set=self.regenerate_task_support_set,
                                  regenerate_query_set=self.regenerate_task_query_set)
        return copy.deepcopy(task) # deepcopy is necessary for multiple dataloader workers to work

    def compute_normalizer(self) -> Dict[str, List[float]]:
        mean = 0.
        std = 0.
        num_dataset_samples = 0
        for class_name, class_data in self._data.items():
            num_class_samples = class_data.shape[BATCH_DIM]
            data_ = class_data.reshape(num_class_samples, class_data.shape[CHANNEL_DIM], -1)
            mean += data_.mean(axis=2).sum(axis=0)
            std += data_.std(axis=2).sum(axis=0)
            num_dataset_samples += num_class_samples

        mean /= num_dataset_samples
        std /= num_dataset_samples

        normalizer_values = {'mean': mean.tolist(), 'std': std.tolist(), 'num_dataset_samples': num_dataset_samples}
        return normalizer_values
