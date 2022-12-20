from typing import Type
from .mode_connectivity import InstabilityAnalyzer
from ml_utilities.runner import Runner
from omegaconf import DictConfig

KEY_RUN_SCRIPT_NAME = 'run_script_name'
KEY_RUN_SCRIPT_KWARGS = 'run_script_kwargs'

_runner_registry = {InstabilityAnalyzer.str_name: InstabilityAnalyzer}

def get_runner_script(run_script: str) -> Type[Runner]:
    if run_script in _runner_registry:
        return _runner_registry[run_script]
    else:
        assert False, f"Unknown run script \"{run_script}\". Available run_script are: {str(_runner_registry.keys())}"


class ScriptRunner(Runner):

    def __init__(self, config: DictConfig):
        self.runner_script = get_runner_script(config[KEY_RUN_SCRIPT_NAME])
        self.runner = self.runner_script(**config[KEY_RUN_SCRIPT_KWARGS])
    
    def run(self) -> None:
        self.runner.run()
