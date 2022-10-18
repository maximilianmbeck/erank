from typing import Any, Dict, List, Tuple, Union
import torch
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.run_utils.sweep import OVERRIDE_PARAMS_KEY
from ml_utilities.logger import LOG_FOLDERNAME, FN_FINAL_RESULTS, FN_DATA_LOG
from ml_utilities.utils import flatten_hierarchical_dict
from tqdm import tqdm
from erank.utils import get_best_model_idx, load_model_from_idx


class JobResult:

    def __init__(self, job_dir: Union[str, Path]):
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)
        self._job_dir = job_dir

    @property
    def directory(self) -> Path:
        return self._job_dir

    def get_model_idx(
        self,
        idx: int = -1,
        device: Union[torch.device, str, int] = "auto",
    ) -> BaseModel:
        """If idx is < 0, return best model."""
        if idx < 0:
            idx, _ = get_best_model_idx(self._job_dir)
        return load_model_from_idx(run_path=self._job_dir, idx=idx, device=device)

    def get_config(self) -> DictConfig:
        return OmegaConf.load(self._job_dir / '.hydra' / 'config.yaml')

    def get_override_hpparams(self) -> Dict[str, Any]:
        cfg = self.get_config()
        override_hpparams = cfg.get(OVERRIDE_PARAMS_KEY, {})
        return flatten_hierarchical_dict(override_hpparams)

    def get_summary(self,
                    log_source: str = '',
                    row_sel: Tuple[str, int] = (),
                    col_sel: List[str] = [],
                    append_override_hpparams: bool = True,
                    append_seed: bool = True) -> pd.DataFrame:
        summary_dict = OmegaConf.to_container(
            OmegaConf.load(self._job_dir / LOG_FOLDERNAME / f'{FN_FINAL_RESULTS}.yaml'))
        if log_source:
            assert row_sel and col_sel, 'Must provide row and column selection.'
            log_df = self.get_data_log(source=log_source)
            row_df = log_df[log_df[row_sel[0]] == row_sel[1]][col_sel]
            log_dict = row_df.transpose().to_dict()  # this is a dictionary {index: {col: vals}}
            # remove index
            _, log_dict = next(iter(log_dict.items()))
            # add row indicator
            log_dict = {f'{k}-{row_sel[0]}-{row_sel[1]}': v for k, v in log_dict.items()}
            summary_dict.update(log_dict)
        if append_override_hpparams:
            summary_dict.update(self.get_override_hpparams())
        if append_seed:
            cfg = self.get_config()
            seed = cfg.config.experiment_data.seed
            summary_dict.update({'seed': seed})
        return pd.DataFrame(summary_dict, index=[self._job_dir.name])

    def get_data_log(self, source: str) -> pd.DataFrame:
        filename = FN_DATA_LOG.format(datasource=source)
        log_file = self._job_dir / LOG_FOLDERNAME / filename
        if not log_file.exists():
            raise ValueError(
                f'Log file for source `{source}` does not exist at `{str(self._job_dir / LOG_FOLDERNAME)}`!')

        return pd.read_csv(log_file, index_col=0)

    def is_successful_job(self) -> bool:
        valid_job = True
        try:
            best_idx, specifier = get_best_model_idx(self._job_dir)
        except:
            valid_job = False
        return valid_job

    def __str__(self):
        return str(self._job_dir)


class SweepResult:

    def __init__(self, sweep_dir: Union[str, Path], run_folder: str = 'outputs'):
        if isinstance(sweep_dir, str):
            sweep_dir = Path(sweep_dir)
        self._sweep_dir = sweep_dir
        self._sweep_runs_dir = self._sweep_dir / run_folder
        assert self._sweep_runs_dir.exists(), f'Run folder `{run_folder}` does not exist in sweep `{self._sweep_dir}`.'

    @property
    def directory(self) -> Path:
        return self._sweep_dir

    def _get_joblist(self, searchstr: str = '') -> List[Path]:
        if searchstr:
            return list(self._sweep_runs_dir.glob(f'*{searchstr}*'))
        else:
            return list(self._sweep_runs_dir.iterdir())

    def find_jobs(self, searchstr: str) -> List[str]:
        return [str(j) for j in self._get_joblist(searchstr)]

    def get_jobs(self, searchstr: str) -> Union[JobResult, List[JobResult]]:
        joblist = self._get_joblist(searchstr)
        return [JobResult(j) for j in joblist]

    def get_failed_jobs(self) -> List[str]:
        joblist = self._get_joblist()
        failed_jobs = []
        for j in tqdm(joblist):
            if not JobResult(j).is_successful_job():
                failed_jobs.append(str(j))
        return failed_jobs

    def get_summary(self,
                    searchstr: str = '',
                    log_source: str = '',
                    row_sel: Tuple[str, int] = (),
                    col_sel: List[str] = [],
                    append_override_hpparams: bool = True,
                    append_seed: bool = True) -> pd.DataFrame:
        joblist = self._get_joblist(searchstr=searchstr)
        summaries = [
            JobResult(job_dir).get_summary(log_source=log_source,
                                           row_sel=row_sel,
                                           col_sel=col_sel,
                                           append_override_hpparams=append_override_hpparams,
                                           append_seed=append_seed) for job_dir in tqdm(sorted(joblist))
        ]
        return pd.concat(summaries)

    def get_sweep_params(self) -> List[str]:
        cfg = OmegaConf.load(self._sweep_dir / '.hydra' / 'config.yaml')
        return [ax.parameter for ax in cfg.sweep.axes]