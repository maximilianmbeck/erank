from typing import Dict, List, Set, Tuple, Union
import math
import logging
import itertools
from abc import abstractmethod
from pathlib import Path
import numpy as np
from erank.data.basemetadataset import BaseMetaDataset, Task
from ml_utilities.utils import convert_to_simple_str

LOGGER = logging.getLogger(__name__)
TASK_NAME_SEPARATOR = '#'
SUPPORT_X_IDXS_KEY = QUERY_X_IDXS_KEY = 'x_idxs'


class ClassificationTask(Task):

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 dataset_name: str,
                 task_data: Dict[str, np.ndarray],
                 rng: np.random.Generator,
                 regenerate_support_set: bool = True,
                 regenerate_query_set: bool = False):
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         regenerate_support_set=regenerate_support_set,
                         regenerate_query_set=regenerate_query_set,
                         rng=rng)
        self._dataset_name = dataset_name
        self._task_data = task_data  # {class_name: data_samples}
        self._task_labels = self._generate_labels()  # {class_name: label_idx}

        self._generate_query_set()
        self._generate_support_set()

    def _generate_labels(self) -> Dict[str, int]:
        return {class_name: label for label, class_name in enumerate(self._task_data)}

    def _generate_support_set(self) -> None:
        # for every class in task_data, sample support_size data samples
        for class_name, data_samples in self._task_data.items():
            # TODO from here
            pass

    def _generate_query_set(self) -> None:
        return super()._generate_query_set()

    def _get_available_sample_indices(self) -> List[int]:
        pass

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
        self._data: Dict[str, np.ndarray] = None  # {class_name: class samples}

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
        if num_tasks < 0:
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
                                  regenerate_support_set=self.regenerate_task_support_set,
                                  regenerate_query_set=self.regenerate_task_query_set)
        return task
