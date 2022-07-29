
from pathlib import Path
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from erank.trainer import Trainer, get_trainer_class
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_experiment(cfg: DictConfig):
    LOGGER.info(f'Starting experiment with config: \n{OmegaConf.to_yaml(cfg)}')
    cfg = cfg.config
    cfg.experiment_data.experiment_dir = Path().cwd()
    trainer_class = get_trainer_class(cfg.trainer.training_setup)
    trainer = trainer_class(config=cfg)
    trainer.train()


if __name__=='__main__':
    run_experiment()