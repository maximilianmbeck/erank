
import logging
from omegaconf import DictConfig
from ml_utilities.runner import run_job
from erank.trainer import get_trainer_class
from ml_utilities.utils import get_config_file_from_cli, get_config
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def run(cfg: DictConfig):
    trainer_class = get_trainer_class(cfg.config.trainer.training_setup)
    run_job(cfg=cfg, trainer_class=trainer_class)


if __name__=='__main__':
    cfg_file = get_config_file_from_cli(config_folder='configs', script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run(cfg)