import sys
from matplotlib.figure import Figure
import torch
import copy
import numpy as np
from typing import Dict, List, Tuple

from tqdm import tqdm
from erank.data.basemetadataset import QUERY_X_KEY, QUERY_Y_KEY, SUPPORT_X_KEY, SUPPORT_Y_KEY, BaseMetaDataset, Task
from ml_utilities.torch_utils import to_ndarray
import matplotlib.pyplot as plt


class SinusTask(Task):

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 amplitude: float,
                 phase: float,
                 x_range: List[float],
                 regenerate_support_set: bool = False):
        super().__init__()
        self._support_size = support_size
        self._query_size = query_size
        self.amplitude = amplitude
        self.phase = phase
        self.regenerate_support_set = regenerate_support_set
        self.x_range = x_range

        self._generate_support_set()
        self._generate_query_set()

    @property
    def name(self) -> str:
        return f'Ampl_{self.amplitude:2.4f}-Phase_{self.phase:2.4f}'

    def _generate_support_set(self) -> None:
        # uniform sampling of the support set
        support_x = torch.as_tensor(np.random.default_rng().uniform(self.x_range[0], self.x_range[1],
                                                                    (self._support_size, 1)),
                                    dtype=torch.float32)
        self._support_data[SUPPORT_Y_KEY] = self.sinus_func(support_x)
        self._support_data[SUPPORT_X_KEY] = support_x

    def _generate_query_set(self) -> None:
        # query set consists of equidistant distributed points in x
        query_x = torch.linspace(self.x_range[0], self.x_range[1], self._query_size).reshape(-1, 1)
        self._query_data[QUERY_Y_KEY] = self.sinus_func(query_x)
        self._query_data[QUERY_X_KEY] = query_x

    def sinus_func(self, x: torch.Tensor) -> torch.Tensor:
        return self.amplitude * torch.sin(x + self.phase)

    def plot_query_predictions(self, epoch: int, preds_before_learning: torch.Tensor, preds_after_learning: torch.Tensor) -> Tuple[Figure, str]:
        fig, ax = plt.subplots(1,1)
        ax.plot(to_ndarray(self.query_set[0]), to_ndarray(self.query_set[1]), color='red', label='Ground truth')
        # use direct access to _support_data dict to avoid regenerating if `regenerate_support_set` is true
        ax.plot(to_ndarray(self._support_data[SUPPORT_X_KEY]), to_ndarray(self._support_data[SUPPORT_Y_KEY]), 'o', color='black', label='Support samples')
        ax.plot(to_ndarray(self.query_set[0]), to_ndarray(preds_before_learning), color='blue', label='preds before inner-loop')
        ax.plot(to_ndarray(self.query_set[0]), to_ndarray(preds_after_learning), color='orange', label='preds after inner-loop')
        ax.legend()
        ax.set_title(f'Predictions - Epoch: {epoch} | Task: {self.name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fname = f'epoch-{epoch:05d}-querypred-task-{self.name}.png'
        return fig, fname

    @property
    def support_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.regenerate_support_set:
            self._generate_support_set()
        return self._support_data[SUPPORT_X_KEY], self._support_data[SUPPORT_Y_KEY]


class SinusDataset(BaseMetaDataset):

    def __init__(self,
                 support_size: int = 10,
                 query_size: int = 10,
                 num_tasks: int = 10000,
                 amplitude_range: List[float] = [0.1, 5.0],
                 phase_range: List[float] = [0, 2 * 3.14159265359],
                 x_range: List[float] = [-5, 5],
                 regenerate_task_support_set: bool = False, # regenerate support set on each call
                 seed: int = 0): # seed only used for task generation
        super().__init__(support_size=support_size, query_size=query_size, num_tasks=num_tasks)
        assert len(amplitude_range) == 2 and len(phase_range) == 2 and len(x_range) == 2
        rng = np.random.default_rng(seed=seed)
        self.amplitudes = rng.uniform(amplitude_range[0], amplitude_range[1], size=num_tasks)
        self.phases = rng.uniform(phase_range[0], phase_range[1], size=num_tasks)
        self.x_range = x_range
        self.regenerate_task_support_set = regenerate_task_support_set
        # generate all tasks an hold them in memory
        self.tasks : np.ndarray = None
        self.task_name_to_index : Dict[str, int] = None
        self.tasks, self.task_name_to_index = self._generate_tasks()

    def _generate_tasks(self) -> Tuple[np.ndarray, Dict[str, int]]:
        tasks = []
        name_to_index = {}
        for i in tqdm(range(self.num_tasks), file=sys.stdout, desc='Generating Sinus tasks'):
            # we need deepcopy to "deepcopy" the internal dictionaries of the tasks
            task = copy.deepcopy(
                SinusTask(self.support_size,
                          self.query_size,
                          self.amplitudes[i],
                          self.phases[i],
                          self.x_range,
                          regenerate_support_set=self.regenerate_task_support_set))
            tasks.append(task)
            name_to_index[task.name] = i
        return np.array(tasks), name_to_index

    def sample_tasks(self, num_tasks: int) -> List[SinusTask]:
        # task_idxes = np.random.default_rng().integers(len(self.tasks), size=num_tasks)
        if num_tasks == -1:
            num_tasks = len(self.tasks)
        return np.random.default_rng().choice(self.tasks, size=num_tasks, replace=False).tolist()

    def get_tasks(self, num_tasks: int = -1) -> List[Task]:
        if num_tasks == -1:
            num_tasks = len(self.tasks)
        return self.tasks[:num_tasks].tolist()

    def get_task(self, task_name: str) -> Task:
        return self.tasks[self.task_name_to_index[task_name]]