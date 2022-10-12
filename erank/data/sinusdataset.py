from typing import Dict, List, Tuple
import sys
import torch
import copy
import numpy as np

from tqdm import tqdm
from matplotlib.figure import Figure
from erank.data.basemetadataset import BaseMetaDataset, Task, QUERY_X_KEY, QUERY_Y_KEY, SUPPORT_X_KEY, SUPPORT_Y_KEY
from ml_utilities.torch_utils import to_ndarray
import matplotlib.pyplot as plt


class SinusTask(Task):

    def __init__(self,
                 support_size: int,
                 query_size: int,
                 amplitude: float,
                 phase: float,
                 x_range: List[float],
                 rng: np.random.Generator = None,
                 regenerate_support_set: bool = False,
                 regenerate_query_set: bool = False):
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         regenerate_support_set=regenerate_support_set,
                         regenerate_query_set=regenerate_query_set,
                         rng=rng)
        self.amplitude = amplitude
        self.phase = phase
        self.x_range = x_range

        self._generate_sets()

    @property
    def name(self) -> str:
        return f'Ampl_{self.amplitude:2.4f}-Phase_{self.phase:2.4f}'

    def _generate_support_set(self) -> None:
        # uniform sampling of the support set
        support_x = torch.as_tensor(self._rng.uniform(self.x_range[0], self.x_range[1], (self._support_size, 1)),
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

    def plot_query_predictions(self, epoch: int, preds: Dict[int, torch.Tensor]) -> Tuple[Figure, str]:
        fig, ax = plt.subplots(1, 1)
        ax.plot(to_ndarray(self.query_set[0]), to_ndarray(self.query_set[1]), color='red', label='Ground truth')
        # use direct access to _support_data dict to avoid regenerating if `regenerate_support_set` is true
        ax.plot(to_ndarray(self._support_data[SUPPORT_X_KEY]),
                to_ndarray(self._support_data[SUPPORT_Y_KEY]),
                'o',
                color='black',
                label='Support samples')
        for step, pred in preds.items():
            ax.plot(to_ndarray(self.query_set[0]), to_ndarray(pred), label=f'preds after {step} steps')
        ax.legend()
        ax.set_title(f'Predictions - Epoch: {epoch} | Task: {self.name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fname = f'epoch-{epoch:05d}-querypred-task-{self.name}.png'
        return fig, fname


class SinusDataset(BaseMetaDataset):

    def __init__(
            self,
            support_size: int = 10,
            query_size: int = 10,
            num_tasks: int = -1,
            amplitude_range: List[float] = [0.1, 5.0],
            phase_range: List[float] = [0, 2 * 3.14159265359],
            x_range: List[float] = [-5, 5],
            regenerate_task_support_set: bool = False,  # regenerate support set on each call
            regenerate_task_query_set: bool = False,
            seed: int = 0):  # seed used for task generation, support and query samples
        super().__init__(support_size=support_size,
                         query_size=query_size,
                         num_tasks=num_tasks,
                         regenerate_task_support_set=regenerate_task_support_set,
                         regenerate_task_query_set=regenerate_task_query_set,
                         seed=seed)
        assert len(amplitude_range) == 2 and len(phase_range) == 2 and len(x_range) == 2
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.x_range = x_range
        # pre-generate some tasks which are accessed via get task to ensure deterministic behavior
        self.pregen_tasks, self.pregen_task_name_to_index = self.create_pregen_tasks()

    def sample_tasks(self, num_tasks: int) -> List[SinusTask]:
        if num_tasks <= 0:
            return []
        tasks = [self._create_task() for _ in range(num_tasks)]
        return tasks

    def _create_task(self) -> SinusTask:
        amplitude = self._rng.uniform(self.amplitude_range[0], self.amplitude_range[1])
        phase = self._rng.uniform(self.phase_range[0], self.phase_range[1])

        task = SinusTask(self.support_size,
                         self.query_size,
                         amplitude,
                         phase,
                         self.x_range,
                         self._rng,
                         regenerate_support_set=self.regenerate_task_support_set)

        return copy.deepcopy(task) # deepcopy is necessary for multiple dataloader workers to work
