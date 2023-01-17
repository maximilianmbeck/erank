from typing import Type
from omegaconf import OmegaConf, DictConfig

from ml_utilities.runner import Runner

from .mode_connectivity.instability_analysis import InstabilityAnalyzer
from .mode_connectivity.train_instability_analysis import TrainInstabilityAnalysis

KEY_RUN_SCRIPT_NAME = 'run_script_name'
KEY_RUN_SCRIPT_KWARGS = 'run_script_kwargs'

_runner_registry = {InstabilityAnalyzer.str_name: InstabilityAnalyzer, TrainInstabilityAnalysis.str_name: TrainInstabilityAnalysis}

def get_runner_script(run_script: str) -> Type[Runner]:
    if run_script in _runner_registry:
        return _runner_registry[run_script]
    else:
        assert False, f"Unknown run script \"{run_script}\". Available run_script are: {str(_runner_registry.keys())}"


class ScriptRunner(Runner):

    def __init__(self, config: DictConfig):
        OmegaConf.resolve(config)
        self.runner_script = get_runner_script(config[KEY_RUN_SCRIPT_NAME])
        self.runner = self.runner_script(**config[KEY_RUN_SCRIPT_KWARGS])
    
    def run(self) -> None:
        self.runner.run()
