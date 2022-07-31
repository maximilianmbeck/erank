import sys
from matplotlib.figure import Figure
import torch
import copy
import numpy as np
from typing import List, Tuple

from tqdm import tqdm
from erank.data.basemetadataset import QUERY_X_KEY, QUERY_Y_KEY, SUPPORT_X_KEY, SUPPORT_Y_KEY, BaseMetaDataset, Task

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

    def plot_query_predictions(self, preds_before_learning: torch.Tensor, preds_after_learning: torch.Tensor) -> Figure:
        # plt.plot(self.query_set[0].numpy(), self.query_set[1].numpy())
        pass

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
                 regenerate_task_support_set: bool = False):  # regenerate support set on each call
        super().__init__(support_size=support_size, query_size=query_size, num_tasks=num_tasks)
        assert len(amplitude_range) == 2 and len(phase_range) == 2 and len(x_range) == 2
        self.amplitudes = np.random.default_rng().uniform(amplitude_range[0], amplitude_range[1], size=num_tasks)
        self.phases = np.random.default_rng().uniform(phase_range[0], phase_range[1], size=num_tasks)
        self.x_range = x_range
        self.regenerate_task_support_set = regenerate_task_support_set
        # generate all tasks an hold them in memory
        self.tasks = np.array(self._generate_tasks())

    def _generate_tasks(self) -> List[SinusTask]:
        # we need deepcopy to "deepcopy" the internal dictionaries of the tasks
        return [
            copy.deepcopy(
                SinusTask(self.support_size,
                          self.query_size,
                          self.amplitudes[i],
                          self.phases[i],
                          self.x_range,
                          regenerate_support_set=self.regenerate_task_support_set))
            for i in tqdm(range(self.num_tasks), file=sys.stdout, desc='Generating Sinus tasks')
        ]

    def sample_tasks(self, num_tasks: int) -> List[SinusTask]:
        # task_idxes = np.random.default_rng().integers(len(self.tasks), size=num_tasks)
        if num_tasks == -1:
            num_tasks = len(self.tasks)
        return np.random.default_rng().choice(self.tasks, size=num_tasks, replace=False).tolist()

    def get_tasks(self, num_tasks: int = -1) -> List[Task]:
        if num_tasks == -1:
            num_tasks = len(self.tasks)
        return self.tasks[:num_tasks].tolist()