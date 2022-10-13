
from pathlib import Path
import warnings
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from erank.trainer import get_trainer_class
LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_experiment(cfg: DictConfig):
    LOGGER.info(f'Starting experiment with config: \n{OmegaConf.to_yaml(cfg)}')
    warnings.filterwarnings('once')
    cfg = cfg.config
    cfg.experiment_data.experiment_dir = str(Path().cwd().resolve())
    cfg.experiment_data.job_name = HydraConfig.get().job.name
    trainer_class = get_trainer_class(cfg.trainer.training_setup)
    trainer = trainer_class(config=cfg)
    trainer.train()


if __name__=='__main__':
    run_experiment()