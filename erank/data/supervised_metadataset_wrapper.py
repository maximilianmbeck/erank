
from torch.utils import data
from typing import Any, Dict
from erank.data import get_metadataset_class
from erank.data.basemetadataset import Task
from ml_utilities.data.datasetgenerator import DatasetGeneratorInterface


class SupervisedMetaDatasetWrapper(DatasetGeneratorInterface):
    """Wrapper for accessing single tasks of a Meta-Dataset for supervised training."""

    def __init__(self, metadataset: str, metadataset_kwargs: Dict[str, Any], task_idx: int = 0):
        self._task_idx = task_idx
        metadataset_class = get_metadataset_class(metadataset)
        self.metadataset = metadataset_class(**metadataset_kwargs)

    def get_meta_task(self, idx: int = -1) -> Task:
        if idx == -1:
            idx = self._task_idx
        return self.metadataset.get_tasks(start_index=idx, num_tasks=1)[0]

    @property
    def task_name(self) -> str:
        return self.get_meta_task(self._task_idx).name

    @property
    def train_split(self) -> data.TensorDataset:
        return self.get_meta_task(self._task_idx).get_support_dataset()

    @property
    def val_split(self) -> data.TensorDataset:
        return self.get_meta_task(self._task_idx).get_query_dataset()
