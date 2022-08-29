from typing import Callable, Type
from omegaconf import DictConfig
import torch.utils.data as data
from erank.trainer.reptiletrainer import ReptileTrainer
from erank.trainer.subspacebasetrainer import SubspaceBaseTrainer

from erank.trainer.supervisedtrainer import SupervisedTrainer


_trainer_registry = {'supervised': SupervisedTrainer, 'reptile': ReptileTrainer}

def get_trainer_class(training_setup: str) -> Type[SubspaceBaseTrainer]:
    if training_setup in _trainer_registry:
        return _trainer_registry[training_setup]
    else:
        assert False, f"Unknown training setup \"{training_setup}\". Available training setups are: {str(_trainer_registry.keys())}"