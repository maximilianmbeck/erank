from typing import Any, Dict, Union
import torch
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.run_utils.sweep import OVERRIDE_PARAMS_KEY
from ml_utilities.logger import LOG_FOLDERNAME, FN_FINAL_RESULTS, FN_DATA_LOG
from ml_utilities.utils import flatten_hierarchical_dict
from erank.utils import get_best_model_idx, load_model_from_idx


class JobResult:

    def __init__(self, job_dir: Union[str, Path]):
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)
        self.job_dir = job_dir

    def get_model_idx(
        self,
        idx: int = -1,
        device: Union[torch.device, str, int] = "auto",
    ) -> BaseModel:
        """If idx is < 0, return best model."""
        if idx < 0:
            idx, _ = get_best_model_idx(self.job_dir)
        return load_model_from_idx(run_path=self.job_dir, idx=idx, device=device)

    def get_config(self) -> DictConfig:
        return OmegaConf.load(self.job_dir / '.hydra' / 'config.yaml')

    def get_override_hpparams(self) -> Dict[str, Any]:
        cfg = self.get_config()
        override_hpparams = cfg.get(OVERRIDE_PARAMS_KEY, {})
        return flatten_hierarchical_dict(override_hpparams)

    def get_summary(self, append_override_hpparams: bool = True) -> pd.DataFrame:
        summary_dict = OmegaConf.to_container(OmegaConf.load(self.job_dir / LOG_FOLDERNAME /
                                                             f'{FN_FINAL_RESULTS}.yaml'))
        if append_override_hpparams:
            summary_dict.update(self.get_override_hpparams())
        return pd.DataFrame(summary_dict, index=[self.job_dir.stem])

    def get_data_log(self, source: str) -> pd.DataFrame:
        filename = FN_DATA_LOG.format(datasource=source)
        log_file = self.job_dir / LOG_FOLDERNAME / filename
        if not log_file.exists():
            raise ValueError(f'Log file for source `{source}` does not exist at `{str(self.job_dir / LOG_FOLDERNAME)}`!')

        return pd.read_csv(log_file, index_col=0)

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