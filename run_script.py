
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from erank.scripts import ScriptRunner
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs_scripts')
def run_script(cfg: DictConfig):
    LOGGER.info(f'Running script with config: \n{OmegaConf.to_yaml(cfg)}')
    cfg = cfg.config
    script_runner = ScriptRunner(cfg)
    script_runner.run()


if __name__=='__main__':
    run_script()