from typing import Dict, List, Union
import math
from abc import abstractmethod
from pathlib import Path
import numpy as np
from erank.data.basemetadataset import BaseMetaDataset, Task
from ml_utilities.utils import convert_to_simple_str

TASK_NAME_SEPARATOR = '#'

class ClassificationTask(Task):

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 task_data: Dict[str, np.ndarray],
                 regenerate_support_set: bool = True,
                 regenerate_query_set: bool = False,
                 rng: np.random.Generator = None):
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         regenerate_support_set=regenerate_support_set,
                         regenerate_query_set=regenerate_query_set,
                         rng=rng)

        self._task_data = task_data

    @property
    def task_classes(self) -> List[str]:
        return list(self._task_data.keys())

    @property
    def n_way(self) -> int:
        return len(self.task_classes)

    @property
    def name(self) -> str:
        return convert_to_simple_str(self.task_classes, separator=TASK_NAME_SEPARATOR)



class BaseMetaClassificationDataset(BaseMetaDataset):

    def __init__(self,
                 data_root_path: Union[str, Path],
                 n_way_classification: int,
                 support_size: int,
                 query_size: int,
                 split: str,
                 num_tasks: int = -1,
                 seed: int = None,
                 normalizer: Dict[str, List[float]] = None):
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         num_tasks=num_tasks,
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

        self._data: Dict[str, np.ndarray] = None

    @property
    def dataset_classes(self) -> List[str]:
        return list(self._data.keys())

    @property
    def n_way(self) -> int:
        return self._n_way_classification

    @property
    def n_way_combinations(self) -> int:
        return math.factorial(len(self.dataset_classes)) / (math.factorial(self.n_way) *
                                                            math.factorial(len(self.dataset_classes) - self.n_way))

    @abstractmethod
    def _load_data(self, split: str) -> Dict[str, np.ndarray]:
        """Load all data from disk into memory."""
        pass