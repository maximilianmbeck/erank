
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf
from erank.scripts import ScriptRunner
from ml_utilities.utils import get_config_file_from_cli, get_config
LOGGER = logging.getLogger(__name__)

def run_script(cfg: DictConfig):
    LOGGER.info(f'Running script with config: \n{OmegaConf.to_yaml(cfg)}')
    cfg = cfg.config
    script_runner = ScriptRunner(cfg)
    script_runner.run()

if __name__=='__main__':
    cfg_file = get_config_file_from_cli(config_folder='configs_scripts', script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run_script(cfg)