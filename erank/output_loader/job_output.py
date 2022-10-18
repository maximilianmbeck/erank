from typing import Union
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ml_utilities.torch_models.base_model import BaseModel
from erank.utils import get_best_model_idx, load_model_from_idx

class JobResult:

    def __init__(self, job_dir: Union[str, Path]):
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)
        self.job_dir = job_dir

    
    def get_model_idx(self, idx: int = -1, device: Union[torch.device, str, int] = "auto",) -> BaseModel:
        """If idx is < 0, return best model."""
        if idx < 0:
            idx, _ = get_best_model_idx(self.job_dir)
        return load_model_from_idx(run_path=self.job_dir, idx=idx, device=device)

    def get_config(self) -> DictConfig:
        return OmegaConf.load(self.job_dir / '.hydra' / 'config.yaml')
    
    def get_override_hpparams(self):
        pass

    def is_valid_job(self) -> bool:
        valid_job = True
        try:
            best_idx, specifier = get_best_model_idx(self.job_dir)
        except:
            valid_job = False
        return valid_job

    def __str__(self):
        return str(self.job_dir)
    

class SweepResult:

    def __init__(self):
        pass