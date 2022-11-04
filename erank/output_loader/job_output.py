from typing import Any, Dict, List, Tuple, Union
import torch
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ml_utilities.torch_models.base_model import BaseModel, FN_MODEL_FILE_EXT, FN_MODEL_PREFIX
from ml_utilities.run_utils.sweep import OVERRIDE_PARAMS_KEY
from ml_utilities.logger import LOG_FOLDERNAME, FN_FINAL_RESULTS, FN_DATA_LOG, FN_DATA_LOG_PREFIX
from ml_utilities.trainers.basetrainer import RUN_PROGRESS_MEASURE_STEP
from ml_utilities.utils import flatten_hierarchical_dict
from tqdm import tqdm
from erank.utils import get_best_model_idx, load_model_from_idx


class JobResult:
    """Class providing access to results of a finished job.
    """

    def __init__(self, job_dir: Union[str, Path]):
        """
        Args:
            job_dir (Union[str, Path]): The run directory of the job.
        """
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)
        self._job_dir = job_dir

    @property
    def directory(self) -> Path:
        """The job directory."""
        return self._job_dir

    @property
    def job_config(self) -> DictConfig:
        """Return the job config."""
        return OmegaConf.load(self._job_dir / '.hydra' / 'config.yaml')

    @property
    def override_hpparams(self) -> Dict[str, Any]:
        """Return the hyper-parameters that where overriden by a sweep."""
        cfg = self.job_config
        override_hpparams = cfg.get(OVERRIDE_PARAMS_KEY, {})
        return flatten_hierarchical_dict(override_hpparams)

    @property
    def is_successful_job(self) -> bool:
        """Returns true, if the job has run sucessfully."""
        valid_job = True
        try:
            best_idx, specifier = get_best_model_idx(self._job_dir)
        except:
            valid_job = False
        return valid_job

    @property
    def data_log_sources(self) -> List[str]:
        """The data log sources."""
        log_dir = self._job_dir / LOG_FOLDERNAME
        return [p.stem.replace(FN_DATA_LOG_PREFIX, '') for p in log_dir.glob(pattern=f'{FN_DATA_LOG_PREFIX}*')]

    @property
    def progress_measure(self) -> str:
        """The progress measure used for this run. For e.g. early stopping depends on this."""
        _, progress_measure = get_best_model_idx(self._job_dir)
        return progress_measure

    @property
    def available_model_checkpoint_indices(self) -> List[int]:
        """The available model checkpoints."""
        idxes = [
            int(p.stem.replace(FN_MODEL_PREFIX, '').replace(self.progress_measure, '').replace('_', ''))
            for p in self._job_dir.glob(pattern=f'{FN_MODEL_PREFIX}*{FN_MODEL_FILE_EXT}')
        ]
        idxes.sort()
        return idxes

    @property
    def best_model_idx(self) -> int:
        """The best model index."""
        best_idx, _ = get_best_model_idx(self._job_dir)
        return best_idx

    def get_model_idx(
        self,
        idx: int = -1,
        device: Union[torch.device, str, int] = "auto",
    ) -> BaseModel:
        """Return a model given a checkpoint index.

        Args:
            idx (int, optional): The model checkpoint index. If idx is < 0, return best model.. Defaults to -1.
            device (Union[torch.device, str, int], optional): Device to where the model is loaded to. Defaults to "auto".

        Returns:
            BaseModel: The model.
        """

        if idx < 0:
            idx, _ = get_best_model_idx(self._job_dir)
        return load_model_from_idx(run_path=self._job_dir, idx=idx, device=device)

    def get_summary(self,
                    log_source: str = '',
                    row_sel: Tuple[str, int] = (),
                    col_sel: List[str] = [],
                    append_override_hpparams: bool = True,
                    append_seed: bool = True) -> pd.DataFrame:
        """Return a summary of the run. 

        Args:
            log_source (str, optional): The log source. If specified, it adds metrics from the logsource. Defaults to ''.
            row_sel (Tuple[str, int], optional): Select a row in the log source, if unspecified use the best epoch / step. Defaults to ().
            col_sel (List[str], optional): The columns. Defaults to [].
            append_override_hpparams (bool, optional): Add override params. Defaults to True.
            append_seed (bool, optional): Add a seed column. Defaults to True.

        Returns:
            pd.DataFrame: The summary table.
        """
        summary_dict = OmegaConf.to_container(
            OmegaConf.load(self._job_dir / LOG_FOLDERNAME / f'{FN_FINAL_RESULTS}.yaml'))
        if log_source:
            assert col_sel, 'Must provide a column selection.'
            assert isinstance(col_sel, list), 'Selected columns must be provided as list.'

            if not row_sel:
                idx, progress_measure = get_best_model_idx(self._job_dir)
                if progress_measure == RUN_PROGRESS_MEASURE_STEP:
                    progress_measure = 'train_step'
                row_sel = (progress_measure, idx)

            log_df = self.get_data_log(source=log_source)
            row_df = log_df[log_df[row_sel[0]] == row_sel[1]][col_sel]
            log_dict = row_df.transpose().to_dict()  # this is a dictionary {index: {col: vals}}
            # remove index
            _, log_dict = next(iter(log_dict.items()))
            # add row indicator
            log_dict = {f'{k}-{row_sel[0]}-{row_sel[1]}': v for k, v in log_dict.items()}
            summary_dict.update(log_dict)
        if append_override_hpparams:
            summary_dict.update(self.override_hpparams)
        if append_seed:
            cfg = self.job_config
            seed = cfg.config.experiment_data.seed
            summary_dict.update({'seed': seed})
        return pd.DataFrame(summary_dict, index=[self._job_dir.name])

    def get_data_log(self, source: str) -> pd.DataFrame:
        """Returns the data log table.

        Args:
            source (str): The specifier of the log source, e.g. `val` or `train`.

        Raises:
            ValueError: If the data log file does not exist.

        Returns:
            pd.DataFrame: The log table.
        """
        filename = FN_DATA_LOG.format(datasource=source)
        log_file = self._job_dir / LOG_FOLDERNAME / filename
        if not log_file.exists():
            raise ValueError(
                f'Log file for source `{source}` does not exist at `{str(self._job_dir / LOG_FOLDERNAME)}`!')

        return pd.read_csv(log_file, index_col=0)

    def __str__(self):
        return str(self._job_dir)


class SweepResult:
    """Class providing access to a finished hyperparameter sweep.
    """

    def __init__(self, sweep_dir: Union[str, Path], run_folder: str = 'outputs'):
        if isinstance(sweep_dir, str):
            sweep_dir = Path(sweep_dir)
        self._sweep_dir = sweep_dir
        self._sweep_runs_dir = self._sweep_dir / run_folder
        assert self._sweep_runs_dir.exists(), f'Run folder `{run_folder}` does not exist in sweep `{self._sweep_dir}`.'

    @property
    def directory(self) -> Path:
        return self._sweep_dir

    @property
    def sweep_params(self) -> List[str]:
        cfg = OmegaConf.load(self._sweep_dir / '.hydra' / 'config.yaml')
        return [ax.parameter for ax in cfg.sweep.axes]

    def _get_joblist(self, searchstr: str = '') -> List[Path]:
        if searchstr:
            return list(self._sweep_runs_dir.glob(f'*{searchstr}*'))
        else:
            return list(self._sweep_runs_dir.iterdir())

    def find_jobs(self, searchstr: str = '') -> List[str]:
        """Get matching job names.
        """
        return [str(j) for j in self._get_joblist(searchstr)]

    def get_jobs(self, searchstr: str = '') -> List[JobResult]:
        """Get jobs with matching job name."""
        joblist = self._get_joblist(searchstr)
        return [JobResult(j) for j in joblist]

    def get_failed_jobs(self) -> List[str]:
        """Get failed jobs."""
        joblist = self._get_joblist()
        failed_jobs = []
        for j in tqdm(joblist):
            if not JobResult(j).is_successful_job:
                failed_jobs.append(str(j))
        return failed_jobs

    def get_summary(self,
                    searchstr: str = '',
                    log_source: str = '',
                    row_sel: Tuple[str, int] = (),
                    col_sel: List[str] = [],
                    append_override_hpparams: bool = True,
                    append_seed: bool = True) -> pd.DataFrame:
        """Return a summary table with the job name as index. 
        Calls the `get_summary()` method of each job in the sweep.

        Args:
            searchstr (str, optional): A string to prefilter the runs in the sweep. Defaults to ''.
            log_source (str, optional): The log source. If specified, it adds metrics from the logsource. Defaults to ''.
            row_sel (Tuple[str, int], optional): Select a row in the log source, if unspecified use the best epoch / step. Defaults to ().
            col_sel (List[str], optional): The columns. Defaults to [].
            append_override_hpparams (bool, optional): Add override params. Defaults to True.
            append_seed (bool, optional): Add a seed column. Defaults to True.

        Returns:
            pd.DataFrame: The summary table.
        """
        joblist = self._get_joblist(searchstr=searchstr)
        summaries = [
            JobResult(job_dir).get_summary(log_source=log_source,
                                           row_sel=row_sel,
                                           col_sel=col_sel,
                                           append_override_hpparams=append_override_hpparams,
                                           append_seed=append_seed) for job_dir in tqdm(sorted(joblist))
        ]
        return pd.concat(summaries)
