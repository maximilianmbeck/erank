import hydra
from omegaconf import DictConfig

from ml_utilities.runner import run_sweep

@hydra.main(version_base=None, config_path='configs', config_name='config.yaml')
def run(cfg: DictConfig):
    run_sweep(cfg)

if __name__=='__main__':
    run()