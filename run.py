
import hydra
import logging
from omegaconf import DictConfig
from ml_utilities.runner import run_job
from erank.trainer import get_trainer_class
from hydra.core.hydra_config import HydraConfig

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def run(cfg: DictConfig):
    trainer_class = get_trainer_class(cfg.config.trainer.training_setup)
    run_job(cfg=cfg, trainer_class=trainer_class)


if __name__=='__main__':
    run()