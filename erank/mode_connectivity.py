from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import logging
import copy
import torch
import itertools
import pickle
import pandas as pd
import numpy as np
import torch.utils.data as data
from torch import nn
from torchmetrics import Metric
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from ml_utilities.runner import Runner
from ml_utilities.output_loader.job_output import JobResult, SweepResult
from ml_utilities.time_utils import FORMAT_DATETIME_SHORT
from ml_utilities.utils import get_device, hyp_param_cfg_to_str, convert_listofdicts_to_dictoflists, setup_logging
from ml_utilities.torch_utils.metrics import get_metric, TAccuracy
from ml_utilities.run_utils.run_handler import EXP_NAME_DIVIDER
from ml_utilities.output_loader.repo import KEY_CFG_CREATED, KEY_CFG_UPDATED, FORMAT_CFG_DATETIME

from ml_utilities.run_utils.sweep import Sweeper
from erank.data.datasetgenerator import DatasetGenerator

LOGGER = logging.getLogger(__name__)

FN_INSTABILITY_ANALYSIS_FOLDER = 'instability_analysis'
PARAM_NAME_INIT_MODEL_IDX_K = 'init_model_idx_k'


class InstabilityAnalyzer(Runner):
    str_name = 'instability_analyzer'
    save_readable_format = 'xlsx'
    save_pickle_format = 'p'
    fn_instability_analysis = FN_INSTABILITY_ANALYSIS_FOLDER
    fn_hp_result_df = 'hp_result_dfs'
    fn_hp_result_readable = f'hp_result_{save_readable_format}'
    fn_combined_results = 'combined_results'
    fn_config = 'config.yaml'
    key_dataset_result_df = 'datasets'
    key_distance_result_df = 'distances'

    def __init__(
        self,
        instability_sweep: Union[SweepResult, str],
        score_fn: Union[nn.Module, Metric, str] = TAccuracy(),
        interpolation_factors: List[float] = list(torch.linspace(0.0, 1.0, 5)),
        interpolate_linear_kwargs: Dict[str, Any] = {},
        init_model_idx_k_param_name: str = 'trainer.init_model_step',
        device: str = 'auto',
        save_results_to_disc: bool = True,
        num_seed_combinations: int = 1,
        init_model_idxes_ks_or_every: Union[List[int], int] = 0,  # 0 use all available, > 0 every nth, list: use subset
        train_model_idxes: List[int] = [-1],  # -1 use best model only, list: use subset
        hpparam_sweep: DictConfig = None,
    ):
        #* save call config
        saved_args = copy.deepcopy(locals())
        saved_args.pop('self')
        config = OmegaConf.create(saved_args)

        #* save start time
        self._start_time = datetime.now()

        if isinstance(instability_sweep, str):
            instability_sweep = SweepResult(sweep_dir=instability_sweep)
        self.instability_sweep = instability_sweep
        self.device = get_device(device)
        self._save_results_to_disc = save_results_to_disc

        #* setup logging / folders etc.
        self._setup(config)
        LOGGER.info(f'Setup instability analysis with config: \n{OmegaConf.to_yaml(config)}')

        LOGGER.info('Loading variables from sweep.')
        #* get k parameter (init_model_idx) values from sweep
        k_param_values = self.instability_sweep.get_sweep_param_values(init_model_idx_k_param_name)
        if len(k_param_values) == 0:
            raise ValueError(f'No hyperparameter found for k parameter name: `{self._init_model_idx_k_param_name}`')

        if len(k_param_values) > 1:
            raise ValueError(
                f'Multiple hyperparemeters found for k parameter name: `{self._init_model_idx_k_param_name}` - Specify further!'
            )

        self._init_model_idx_k_param_name = list(k_param_values.keys())[0]
        k_param_values = list(k_param_values.values())[0]

        #* parameter specifying the rewind point / number of pretraining steps/epochs
        self._all_init_idx_k_param_values = k_param_values
        # find subset of parameter values
        if isinstance(init_model_idxes_ks_or_every, int):
            if init_model_idxes_ks_or_every > 0:
                self._subset_init_idx_k_param_values = self._all_init_idx_k_param_values[::init_model_idxes_ks_or_every]
            else:
                self._subset_init_idx_k_param_values = self._all_init_idx_k_param_values
        elif isinstance(init_model_idxes_ks_or_every, list):
            self._subset_init_idx_k_param_values = np.array(
                self._all_init_idx_k_param_values)[init_model_idxes_ks_or_every].tolist()
        else:
            raise ValueError(
                f'Unsupported type `{type(init_model_idxes_ks_or_every)}` for `init_model_idxes_or_every`.')
        LOGGER.info(f'Using init_model_idxes / k parameters: {self._subset_init_idx_k_param_values}')

        #* model indices for finetuned models
        self._train_model_idxes = train_model_idxes

        LOGGER.info(f'Finding seed combinations..')
        #* find seed combinations
        sweep_seeds = list(self.instability_sweep.get_sweep_param_values('seed').values())[0]
        if len(sweep_seeds) < 2:
            raise ValueError('Sweep contains less than 2 seeds!')
        available_seed_combinations = list(itertools.combinations(sweep_seeds, 2))
        seed_combinations = available_seed_combinations[:num_seed_combinations]
        if len(available_seed_combinations) < num_seed_combinations:
            LOGGER.warning(
                f'Only {len(available_seed_combinations)} seed combinations available, but {num_seed_combinations} were specified.\nUsing all available combinations now.'
            )
        self.seed_combinations = seed_combinations
        # used seeds
        used_seeds = set()
        for sc in self.seed_combinations:
            used_seeds.add(sc[0])
            used_seeds.add(sc[1])
        self._used_seeds = list(used_seeds)
        LOGGER.info(f'Using seed combinations: {self.seed_combinations}')

        #* Linear interpolation specific parameters
        if isinstance(score_fn, str):
            _score_fn = get_metric(score_fn)
        elif isinstance(score_fn, nn.Module):
            _score_fn = score_fn
        else:
            raise ValueError('Unknown type for score_fn!')
        self.score_fn = _score_fn
        self.interpolation_factors = interpolation_factors
        interp_lin_default_kwargs = {'tqdm_desc': ''}
        interp_lin_default_kwargs.update(interpolate_linear_kwargs)
        self._interpolate_linear_kwargs = interp_lin_default_kwargs

        #* sweep hyperparameters
        self._hpparam_sweep_cfg = hpparam_sweep  # TODO check if can be loaded from sweep as default

    def _setup(self, config: DictConfig) -> None:
        self._hp_result_folder_df = self.directory / InstabilityAnalyzer.fn_hp_result_df
        self._hp_result_folder_readable = self.directory / InstabilityAnalyzer.fn_hp_result_readable
        self._combined_results_folder = self.directory / InstabilityAnalyzer.fn_combined_results

        if self._save_results_to_disc:
            from .scripts import KEY_RUN_SCRIPT_KWARGS, KEY_RUN_SCRIPT_NAME
            self.directory.mkdir(parents=True, exist_ok=True)
            # setup logging
            logfile = self.directory / f'output--{self._start_time.strftime(FORMAT_DATETIME_SHORT)}.log'
            setup_logging(logfile)

            # create folders
            self._hp_result_folder_df.mkdir(parents=False, exist_ok=True)
            self._hp_result_folder_readable.mkdir(parents=False, exist_ok=True)
            self._combined_results_folder.mkdir(parents=False, exist_ok=True)

            # save / update config config
            config_file = self.directory / InstabilityAnalyzer.fn_config
            cfg = OmegaConf.create()
            cfg[KEY_RUN_SCRIPT_NAME] = InstabilityAnalyzer.str_name
            cfg[KEY_RUN_SCRIPT_KWARGS] = config
            cfg[KEY_CFG_UPDATED] = self._start_time.strftime(FORMAT_CFG_DATETIME)

            if config_file.exists():
                existing_cfg = OmegaConf.load(config_file)
                cfg_created = existing_cfg[KEY_CFG_CREATED]
            else:
                cfg_created = cfg[KEY_CFG_UPDATED]
            cfg[KEY_CFG_CREATED] = cfg_created
            OmegaConf.save(cfg, config_file)

    @property
    def remaining_hyperparams(self) -> Dict[str, Any]:
        sweep_params = self.instability_sweep.get_sweep_param_values()
        # remove seed and k_param_name
        _ = sweep_params.pop('seed')
        _ = sweep_params.pop(self._init_model_idx_k_param_name)
        return sweep_params

    @property
    def directory(self) -> Path:
        return self.instability_sweep.directory / InstabilityAnalyzer.fn_instability_analysis

    @property
    def hp_result_folder_df(self) -> Path:
        return self._hp_result_folder_df

    @property
    def hp_result_folder_readable(self) -> Path:
        return self._hp_result_folder_readable

    @property
    def combined_results_folder(self) -> Path:
        return self._combined_results_folder

    @property
    def result_dfs(self) -> Dict[str, pd.DataFrame]:
        # TODO load latest results
        pass

    def instability_analysis_for_hpparam(self,
                                         hypparam_sel: Dict[str, Any] = {},
                                         use_tqdm: bool = True) -> Dict[str, pd.DataFrame]:
        # create run_dict: init_model_idx_k_param_value -> runs with different seeds
        run_dict = self._create_run_dict(hypparam_sel=hypparam_sel)

        dataset_dfs, distance_dfs = {}, {}
        it = run_dict.items()
        if use_tqdm:
            it = tqdm(it, file=sys.stdout)
        # iterate over run_dict and do interpolation for seed_combinations and train_model_idxes
        for init_model_idx_k, k_dict in it:
            if isinstance(it, tqdm):
                it.set_description_str(desc=f'init_model_idx_k={init_model_idx_k}')
            # for every init_model_idx_k_param_value, there must be jobs with all used seeds.
            assert set(self._used_seeds).issubset(set(k_dict.keys())), 'Some seeds are missing!'

            k_runs = dataset_dfs.get(init_model_idx_k, None)
            if k_runs is None:
                dataset_dfs[init_model_idx_k] = []
                distance_dfs[init_model_idx_k] = []

            for sc in self.seed_combinations:
                run_0, run_1 = k_dict[sc[0]], k_dict[sc[1]]
                for train_model_idx in self._train_model_idxes:
                    interp_res_ds_df, interp_result_dist_df = interpolate_linear_runs(
                        run_0=run_0,
                        run_1=run_1,
                        score_fn=self.score_fn,
                        model_idx=train_model_idx,
                        interpolation_factors=torch.tensor(self.interpolation_factors),
                        interpolate_linear_kwargs=self._interpolate_linear_kwargs,
                        device=self.device,
                        return_dataframe=True)
                    dataset_dfs[init_model_idx_k].append(interp_res_ds_df)
                    distance_dfs[init_model_idx_k].append(interp_result_dist_df)

            # create a dataframe for every init_model_idx_k
            dataset_dfs[init_model_idx_k] = pd.concat(dataset_dfs[init_model_idx_k])
            distance_dfs[init_model_idx_k] = pd.concat(distance_dfs[init_model_idx_k])

        # concatenate all dataframes
        dataset_result_df = pd.concat(dataset_dfs, names=[PARAM_NAME_INIT_MODEL_IDX_K])
        # TODO check if compute model distances = TRUE
        distance_result_df = pd.concat(distance_dfs, names=[PARAM_NAME_INIT_MODEL_IDX_K])
        return {
            InstabilityAnalyzer.key_dataset_result_df: dataset_result_df,
            InstabilityAnalyzer.key_distance_result_df: distance_result_df
        }

    def _create_run_dict(self, hypparam_sel: Dict[str, Any] = {}) -> Dict[int, Dict[int, JobResult]]:
        """Create a dictionary containing all runs for an instability analysis run."""
        hp_sel = copy.deepcopy(hypparam_sel)
        run_dict = {}
        for k in self._subset_init_idx_k_param_values:
            add_hp_sel = {self._init_model_idx_k_param_name: k, 'seed': self._used_seeds}
            hp_sel.update(add_hp_sel)
            _, jobs = self.instability_sweep.query_jobs(hp_sel)
            k_dict = {job.seed: job for job in jobs}
            run_dict[k] = k_dict

        return run_dict

    def instability_analysis(self, use_tqdm: bool = True, override_files: bool = False) -> Dict[str, pd.DataFrame]:

        # TODO: drop init_model_idx_k_param_name axis from hpparam_sweep_cfg
        LOGGER.info(f'Starting instability analysis..')
        # create sweep
        sweep = Sweeper.create(sweep_config=self._hpparam_sweep_cfg)
        hp_combinations = sweep.generate_sweep_parameter_combinations(flatten_hierarchical_dicts=True)
        hp_combinations_str = [hyp_param_cfg_to_str(hp) for hp in hp_combinations]
        LOGGER.info(f'Number of hyperparameter combinations for instability analysis: {len(hp_combinations)}')

        # perform instability analysis
        dataset_result_dfs = []
        distance_result_dfs = []
        it = zip(hp_combinations, hp_combinations_str)
        if use_tqdm:
            it = tqdm(it, file=sys.stdout, desc='HP combinations')

        for hp_sel, hp_str in it:
            if self._hp_result_df_exists(hp_str) and not override_files:
                LOGGER.info(f'Params `{hp_str}`: load&skip')
                df_dict = self.load_instability_analysis_for_hpparam(hp_str)
            else:
                LOGGER.info(f'Params `{hp_str}`: compute')
                df_dict = self.instability_analysis_for_hpparam(hypparam_sel=hp_sel, use_tqdm=False)
                self._save_hp_result_dfs(df_dict, hp_str)

            dataset_result_dfs.append(df_dict[InstabilityAnalyzer.key_dataset_result_df])
            distance_result_dfs.append(df_dict[InstabilityAnalyzer.key_distance_result_df])

        # create multiindex
        hp_lists = convert_listofdicts_to_dictoflists(hp_combinations)
        # hp_names = [hp_name.split('.')[-1] for hp_name in hp_lists.keys()]
        hp_names = list(hp_lists.keys())
        index = pd.MultiIndex.from_arrays(list(hp_lists.values()), names=hp_names)

        dataset_result_df = pd.concat(dataset_result_dfs, keys=index)
        distance_result_df = pd.concat(distance_result_dfs, keys=index)

        combined_results = {
            InstabilityAnalyzer.key_dataset_result_df: dataset_result_df,
            InstabilityAnalyzer.key_distance_result_df: distance_result_df
        }
        combined_results_file = self._save_combined_result_dfs(combined_results)

        LOGGER.info(f'Done. Combined results in file `{str(combined_results_file)}`.')

        return combined_results

    def load_instability_analysis_for_hpparam(self, hp_sel_str: str) -> Dict[str, pd.DataFrame]:
        # hp_sel_str without file_ending
        load_file = self.hp_result_folder_df / f'{hp_sel_str}.{InstabilityAnalyzer.save_pickle_format}'
        with load_file.open(mode='rb') as f:
            df_dict = pickle.load(f)
        return df_dict

    def _save_combined_result_dfs(self, dataframe_dict: Dict[str, pd.DataFrame]) -> Path:
        fname = f'combined_result{EXP_NAME_DIVIDER}{self._start_time.strftime(FORMAT_DATETIME_SHORT)}'
        pickle_file = self._save_df_dict_pickle(dataframe_dict, self.combined_results_folder, fname)
        self._save_df_dict_readable(dataframe_dict, self.combined_results_folder, fname)
        return pickle_file

    def _save_hp_result_dfs(self, dataframe_dict: Dict[str, pd.DataFrame], hp_sel_str: str) -> Path:
        pickle_file = self._save_df_dict_pickle(dataframe_dict, self.hp_result_folder_df, hp_sel_str)
        self._save_df_dict_readable(dataframe_dict, self.hp_result_folder_readable, hp_sel_str)
        return pickle_file

    def _save_df_dict_pickle(self, dataframe_dict: Dict[str, pd.DataFrame], dir: Path, filename_wo_ending: str) -> Path:
        save_file = dir / f'{filename_wo_ending}.{InstabilityAnalyzer.save_pickle_format}'
        with save_file.open(mode='wb') as f:
            pickle.dump(dataframe_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return save_file

    def _save_df_dict_readable(self, dataframe_dict: Dict[str, pd.DataFrame], dir: Path,
                               filename_wo_ending: str) -> Path:
        save_file = dir / f'{filename_wo_ending}.{InstabilityAnalyzer.save_readable_format}'
        with pd.ExcelWriter(save_file) as excelwriter:
            for df_name, df in dataframe_dict.items():
                df.to_excel(excel_writer=excelwriter, sheet_name=df_name)
        return save_file

    def _hp_result_df_exists(self, hp_sel_str: str) -> bool:
        return (self.hp_result_folder_df / f'{hp_sel_str}.{InstabilityAnalyzer.save_pickle_format}').exists()

    def run(self) -> None:
        self.instability_analysis()

####


def interpolate_linear_runs(
        run_0: JobResult,
        run_1: JobResult,
        score_fn: Union[nn.Module, Metric],
        model_idx: Union[int, List[int]] = -1,
        interpolation_factors: torch.Tensor = torch.linspace(0.0, 1.0, 5),
        interpolate_linear_kwargs: Dict[str, Any] = {},
        device: Union[torch.device, str, int] = 'auto',
        return_dataframe: bool = True) -> Union[Dict[str, Any], Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """Interpolate linearly between models of two runs. 

    Args:
        run_0 (JobResult): Run 0.
        run_1 (JobResult): Run 1.
        score_fn (Union[nn.Module, Metric]): Performance measure for the models.
        model_idx (Union[int, List[int]], optional): The model index/indices used for linear interpolation. Defaults to -1.
                                                     If -1, use the respective best model.
        interpolation_factors (torch.Tensor, optional): Interpolation factors. Defaults to torch.linspace(0.0, 1.0, 5).
        interpolate_linear_kwargs (Dict[str, Any], optional): Some further keyword arguments for `interpolate_linear`. Defaults to {}.
        device (Union[torch.device, str, int], optional): Device for linear interpolation. Defaults to 'auto'.
        return_dataframe (bool, optional): If true, return results as dataframes. Return a dictionary otherwise. Defaults to True.

    Raises:
        ValueError: If a model index is missing in one of the two runs.

    Returns:
        Union[Dict[str, Any], Tuple[pd.DataFrame, Optional[pd.DataFrame]]]: Results as dataframes or a single dictionary.
    """
    device = get_device(device)
    if isinstance(model_idx, int):
        model_idx = [model_idx]

    # use run_0 to determine interpolation name and seeds
    interpolation_name = run_0.experiment_name + EXP_NAME_DIVIDER + hyp_param_cfg_to_str(run_0.override_hpparams)
    interpolation_seeds = (run_0.experiment_data.seed, run_1.experiment_data.seed)

    # use dataset from run_0 for dataset setup
    data_cfg = run_0.config.config.data
    ds_generator = DatasetGenerator(**data_cfg)
    ds_generator.generate_dataset()

    other_datasets = {'val': ds_generator.val_split}
    if 'other_datasets' in interpolate_linear_kwargs:
        other_datasets.update(interpolate_linear_kwargs['other_datasets'])

    res_dict = {}  # contains all interpolation results as dictionary
    dataset_series: List[pd.Series] = []
    distance_series: List[pd.Series] = []

    runs = [run_0, run_1]
    for midx in model_idx:
        models = []
        m_idxes = []
        for i, r in enumerate(runs):
            try:
                m = r.get_model_idx(idx=midx, device=device)
            except FileNotFoundError:
                raise ValueError(f'Missing model_idx={midx} in run_{i}: {r}')
            models.append(m)
            if midx == -1:
                m_idxes.append(r.best_model_idx)
            else:
                m_idxes.append(midx)

        model_0, model_1 = models

        idx_res_dict = interpolate_linear(model_0=model_0,
                                          model_1=model_1,
                                          score_fn=score_fn,
                                          train_dataset=ds_generator.train_split,
                                          interpolation_factors=interpolation_factors,
                                          other_datasets=other_datasets,
                                          **interpolate_linear_kwargs)
        res_dict[tuple(m_idxes)] = idx_res_dict
        # convert result dict into more readable dataframe
        idx_dataset_series, idx_distance_series = interpolation_result2series(idx_res_dict)
        dataset_series.append(idx_dataset_series)
        distance_series.append(idx_distance_series)

    ret_val = res_dict
    # create dataframes
    if return_dataframe:
        ind = pd.MultiIndex.from_product([[interpolation_name], [interpolation_seeds],
                                          list(res_dict.keys())],
                                         names=['job', 'seeds', 'model_idxes'])
        datasets_df = pd.DataFrame(dataset_series, index=ind)
        distances_df = None
        if not distance_series[0] is None:
            distances_df = pd.DataFrame(distance_series, index=ind)
        ret_val = datasets_df, distances_df
    return ret_val
    # return res_dict, dataset_series, distance_series


def interpolation_result2series(result_dict: Dict[str, Any]) -> Tuple[pd.Series, Optional[pd.Series]]:
    ds_key = 'datasets'
    dist_key = 'distances'

    def ds_result2series(ds_result_dict: Dict[str, Any], interp_factors: np.ndarray) -> pd.Series:
        instability_key = 'instability'
        interp_sc_key = 'interpolation_scores'
        interp_scores = np.array(ds_result_dict[interp_sc_key])
        assert len(interp_factors) == len(interp_scores)
        # create results dictionary with necessary values
        res_dict = {alpha: interp_score for alpha, interp_score in zip(interp_factors, interp_scores)}
        res_dict[instability_key] = ds_result_dict[instability_key]
        # create index
        interp_ind = np.full_like(interp_scores, interp_sc_key, dtype=object)
        idx_tuples = list(zip(interp_ind, interp_factors))
        idx_tuples.append((instability_key, None))
        ind = pd.MultiIndex.from_tuples(idx_tuples, names=['score', 'alpha'])
        return pd.Series(res_dict.values(), index=ind)

    interp_factors = np.array(result_dict['interpolation_factors'])
    ds_dict = result_dict[ds_key]
    ds_series_dict = {ds_name: ds_result2series(ds_result, interp_factors) for ds_name, ds_result in ds_dict.items()}
    dataset_series = pd.concat(ds_series_dict, names=[ds_key])

    distances_series = None
    if dist_key in result_dict:
        distances_dict = result_dict[dist_key]
        ind = pd.MultiIndex.from_arrays([distances_dict.keys()], names=[dist_key])
        distances_series = pd.Series(distances_dict.values(), index=ind)

    return dataset_series, distances_series


def interpolate_linear(model_0: nn.Module,
                       model_1: nn.Module,
                       train_dataset: data.Dataset,
                       score_fn: Union[nn.Module, Metric],
                       other_datasets: Dict[str, data.Dataset] = {},
                       interpolation_factors: torch.Tensor = torch.linspace(0.0, 1.0, 5),
                       dataloader_kwargs: Dict[str, Any] = {'batch_size': 256},
                       compute_model_distances: bool = True,
                       interpolation_on_train_data: bool = True,
                       tqdm_desc: str = 'Alphas') -> Dict[str, Any]:
    """Interpolate linearly between two models. Evaluates the performance of each interpolated model on given datasets.
    
    Note:
        Also computes the instability value according to Frankle et al., 2020, p. 3.
        Instability = max/min [interpolation_scores] - mean[interpolation_score(0.0), interpolation_score(1.0)]
    
    References:
        Frankle, Jonathan, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. 2020. 
            “Linear Mode Connectivity and the Lottery Ticket Hypothesis.” arXiv. http://arxiv.org/abs/1912.05671.

    Args:
        model_0 (nn.Module): First model.
        model_1 (nn.Module): Second model.
        train_dataset (data.Dataset): Dataset on which the models have been trained. 
                                      If applicable, this dataset is used to recompte batch norm statistics.
        score_fn (Union[nn.Module, Metric]): The performance measure on which each model is used.
        other_datasets (Dict[str, data.Dataset], optional): Evaluation dataset with descriptor as key. Defaults to {}.
        interpolation_factors (torch.Tensor, optional): Interpolation factor for linear interpolation. Defaults to torch.linspace(0.0, 1.0, 5).
        dataloader_kwargs (Dict[str, Any], optional): Additional dataloader keyword arguments. Defaults to {'batch_size': 256}.
        compute_model_distances (bool, optional): Computes distance metrics on given models. Defaults to True.
        interpolation_on_train_data (bool, optional): Evaluates interpolation performance on train data too. Defaults to False.
        tqdm_desc (str, optional): The description for the tqdm progress bar. If '', then no progress bar is displayed. Defaults to 'Alphas'.

    Raises:
        ValueError: If no eval datasets are given or the model architectures do not match.

    Returns:
        Dict[str, Any]: Dictionary containing the results.
                        Example:
                           {'datasets': {'val': {'instability': -0.1300981044769287,
                                                 'interpolation_scores': [0.974958598613739,
                                                                          0.9691435694694519,
                                                                          0.976451575756073]}},
                            'distances': {'cosinesimilarity': 0.010202181525528431,
                                          'l2distance': 30.355226516723633},
                            'interpolation_factors': [0.0, 0.25, 0.5, 0.75, 1.0]}
    """
    get_model_device = lambda model: next(iter(model.parameters())).device
    assert get_model_device(model_0) == get_model_device(model_1), f'Models to interpolate not on same device!'
    device = get_model_device(model_0)
    assert 'train' not in other_datasets, f'`train` is a reserved dataset name. Please rename this evaluation dataset.'
    assert interpolation_factors.dim() == 1, '`interpolation_factors` must be tensor of dimension 1.'
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

    def reset_bn_running_stats(module: nn.Module) -> None:
        if isinstance(module, bn_types):
            module.reset_running_stats()

    def eval_loop(model: nn.Module, dataloader: data.DataLoader, score_fn: nn.Module) -> float:
        batch_scores = []
        for batch_idx, (xs, ys) in enumerate(dataloader):
            xs, ys = xs.to(device), ys.to(device)
            with torch.no_grad():
                y_pred = model(xs)
                score = score_fn(y_pred, ys)
                batch_scores.append(score)
        return torch.tensor(batch_scores).mean().item()

    # check if models have batch_norm layers
    models_have_batch_norm_layers = False
    for m in model_0.modules():
        if isinstance(m, bn_types):
            models_have_batch_norm_layers = True
            break

    # prepare datasets and results dict
    eval_datasets = copy.copy(other_datasets)  # shallow copy
    if interpolation_on_train_data:
        eval_datasets['train'] = train_dataset  # reference only
    ds_dict = {ds_name: [] for ds_name in eval_datasets}

    res_dict = {}
    res_dict['interpolation_factors'] = interpolation_factors.tolist()

    # create eval_dataloaders
    eval_dataloaders = {ds_name: data.DataLoader(ds, **dataloader_kwargs) for ds_name, ds in eval_datasets.items()}
    if not eval_dataloaders:
        raise ValueError(
            'No evaluation datasets provided. Pass eval_datasets or set `interpolation_on_train_data=True`.')

    interpolation_factors = interpolation_factors.to(device)
    score_fn = score_fn.to(device)
    if tqdm_desc:
        it = tqdm(interpolation_factors, desc=tqdm_desc, file=sys.stdout)
    else:
        it = interpolation_factors
    # alpha = interpolation factor
    for alpha in it:
        # create interpolated model in a memory friendly way (only use memory used for another model instance)
        interp_model = copy.deepcopy(model_0)
        interp_model_state_dict = interp_model.state_dict()
        for (k0, v0), (k1, v1) in zip(model_0.state_dict().items(), model_1.state_dict().items()):
            if k0 != k1:
                raise ValueError(f'Model architectures do not match: {k0} != {k1}')
            torch.lerp(v0, v1, alpha, out=interp_model_state_dict[k0])  # linear interpolation between weights
        interp_model.load_state_dict(interp_model_state_dict)

        if models_have_batch_norm_layers:
            # reset running stats
            interp_model.apply(reset_bn_running_stats)
            # compute batch_norm statistics on train_dataset
            train_loader = data.DataLoader(train_dataset, **dataloader_kwargs)
            interp_model.train(True)
            _ = eval_loop(model=interp_model, dataloader=train_loader, score_fn=score_fn)

        interp_model.train(False)
        # eval on eval_datasets
        for ds_name, dataloader in eval_dataloaders.items():
            score = eval_loop(model=interp_model, dataloader=dataloader, score_fn=score_fn)
            ds_dict[ds_name].append(score)

    if compute_model_distances:
        vec_0 = nn.utils.parameters_to_vector(model_0.parameters())
        vec_1 = nn.utils.parameters_to_vector(model_1.parameters())
        distances = {}
        # L2 distance
        distances['l2distance'] = torch.linalg.norm(vec_1 - vec_0).item()
        # cosine similarity
        distances['cosinesimilarity'] = nn.functional.cosine_similarity(vec_0, vec_1, dim=0).item()
        res_dict['distances'] = distances

    # compute instability value
    # find weight indices for base models
    # (necessary if 0. and 1. value not and beginning or end of interpolation_factors tensor)
    original_models_in_interpolation_factors = True
    base_interpolation_factors = [0.0, 1.0]
    interp_factors_list = interpolation_factors.tolist()
    base_interpolation_factor_idxes = []
    for f in base_interpolation_factors:
        try:
            base_interpolation_factor_idxes.append(interp_factors_list.index(f))
        except ValueError:
            original_models_in_interpolation_factors = False
            break

    # compute instability per dataset
    def compute_instability(interp_scores: List[float]) -> float:
        if original_models_in_interpolation_factors:
            interp_scores = torch.tensor(interp_scores)
            base_mean = interp_scores[base_interpolation_factor_idxes].mean()
            torch_minmax = torch.min if score_fn.higher_is_better else torch.max
            instability = torch_minmax(interp_scores) - base_mean
            return instability.item()
        else:
            return float('nan')

    for k, v in ds_dict.items():
        # k is ds_name, v is interpolation_scores
        ds_dict[k] = {'instability': compute_instability(v), 'interpolation_scores': v}

    res_dict['datasets'] = ds_dict

    return res_dict