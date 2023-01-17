from pathlib import Path
from typing import Any, Dict, List, Union
import sys
import logging
import copy
import torch
import itertools
import pickle
import pandas as pd
from torch import nn
from torchmetrics import Metric
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from datetime import datetime

from ml_utilities.runner import Runner
from ml_utilities.output_loader.result_loader import JobResult, SweepResult
from ml_utilities.time_utils import FORMAT_DATETIME_SHORT
from ml_utilities.utils import get_device, hyp_param_cfg_to_str, convert_listofdicts_to_dictoflists, setup_logging
from ml_utilities.torch_utils.metrics import get_metric, TAccuracy
from ml_utilities.run_utils.run_handler import EXP_NAME_DIVIDER
from ml_utilities.output_loader.repo import KEY_CFG_CREATED, KEY_CFG_UPDATED, FORMAT_CFG_DATETIME

from ml_utilities.run_utils.sweep import Sweeper, SWEEP_TYPE_KEY, SWEEP_TYPE_GRIDVAL
from erank.data.datasetgenerator import DatasetGenerator

from .linear_interpolation import interpolate_linear_runs

LOGGER = logging.getLogger(__name__)

FN_INSTABILITY_ANALYSIS_FOLDER = 'instability_analysis'
PARAM_NAME_INIT_MODEL_IDX_K = 'init_model_idx_k'


class InstabilityAnalyzer(Runner):
    """A class that performs instability analysis outlined in Frankle et al., 2020. 
    Core of this analysis is the linear interpolation of two models traind with different SGD noise from a varying number
    of pretraining steps.
    It takes a SweepResult or the path to the sweep results as input and performs the instability analysis on this sweep.
    The sweep must contain a parameter (`init_model_idx_k_param_name`) that controls the number of pretrain steps on a model
    and finetuning runs with at least two seeds. 

    Note:
        Also computes the instability value according to Frankle et al., 2020, p. 3.
        Instability = max/min [interpolation_scores] - mean[interpolation_score(0.0), interpolation_score(1.0)]

    References:
        Frankle, Jonathan, Gintare Karolina Dziugaite, Daniel M. Roy, and Michael Carbin. 2020. 
            “Linear Mode Connectivity and the Lottery Ticket Hypothesis.” arXiv. http://arxiv.org/abs/1912.05671.

    Args:
        main_training_job (Union[JobResult, str]): The main training run that creates the checkpoints for resuming.
                                                   Provide this for Instability analysis a la Frankle et al. 
        instability_sweep (Union[SweepResult, str]): The (path to the) sweep result. Contains all runs to earlier training checkpoints.
        score_fn (Union[nn.Module, Metric, str], optional): The score function for measuring model performance on the datasets. 
                                                            Defaults to TAccuracy().
        interpolation_factors (List[float], optional): List of interpolation factors. Defaults to list(torch.linspace(0.0, 1.0, 5)).
        interpolate_linear_kwargs (Dict[str, Any], optional): Some keyword arguments for `interpolate_linear()`. Defaults to {}.
        init_model_idx_k_param_name (str, optional): The parameter name that specifies the amount of pretraining in the sweep. 
                                                     Defaults to 'trainer.init_model_step'.
        device (str, optional): The device, e.g. the GPU id. Defaults to 'auto'.
        save_results_to_disc (bool, optional): Save results and logging to disc. Defaults to True.
        override_files (bool, optional): Override results for all hyperparameters, load existing results otherwise. Defaults to False.
        num_seed_combinations (int, optional): Number of seed combinations for linear interpolation. Defaults to 1.
        init_model_idxes_ks_or_every (Union[List[int], int], optional): A list of pretrain model indexes or an interval to use every j-th pretraining model index. 
                                                                        If 0, use all available pretrain indexes. Defaults to 0.
        interpolate_at_model_idxes (List[int], optional): Perform linear interpolation between models with these finetuning indices from runs
                                                 with different seeds. Defaults to [-1, -2].
        save_folder_suffix (str, optional): Suffix for instability analysis result folder in sweep directory. Defaults to ''.
        float_eps_query_job (float, optional): Epsilon for floating point hyperparameter value comparison. Defaults to 1e-3.
        hpparam_sweep (DictConfig, optional): Sweep config producing the hyperparameters on which instability analysis should be performed.
                                              Can be used to provide a subset of hyperparameters to perform instability analysis on.
                                              If None, use sweep config from sweep result, i.e. perform instability analysis on all 
                                              hyperparameter combinations. Defaults to None.
    """
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

    def __init__(self,
                 instability_sweep: Union[SweepResult, Path, str],
                 main_training_job: Union[JobResult, Path, str] = None,
                 score_fn: Union[nn.Module, Metric, str] = TAccuracy(),
                 interpolation_factors: List[float] = list(torch.linspace(0.0, 1.0, 5)),
                 interpolate_linear_kwargs: Dict[str, Any] = {},
                 init_model_idx_k_param_name: str = 'trainer.resume_training.checkpoint_idx',
                 device: str = 'auto',
                 save_results_to_disc: bool = True,
                 override_files: bool = False,
                 num_seed_combinations: int = 1,
                 init_model_idxes_ks_or_every: Union[List[int], int] = 0,
                 interpolate_at_model_idxes: List[int] = [-1, -2], # TODO rename: interpolate at model_idxes
                 save_folder_suffix: str = '',
                 float_eps_query_job: float = 1e-3,
                 hpparam_sweep: DictConfig = None):
        #* save call config
        saved_args = copy.deepcopy(locals())
        saved_args.pop('self')
        if '__class__' in saved_args:
            saved_args.pop('__class__')
        config = OmegaConf.create(saved_args)

        #* save start time
        self._start_time = datetime.now()
        if isinstance(main_training_job, (Path, str)):
            main_training_job = JobResult(job_dir=main_training_job)
        self.main_training_job = main_training_job
        if isinstance(instability_sweep, (Path, str)):
            instability_sweep = SweepResult(sweep_dir=instability_sweep)
        self.instability_sweep = instability_sweep
        super().__init__(runner_dir=instability_sweep.directory)
        self.device = get_device(device)
        self._save_results_to_disc = save_results_to_disc
        self._override_files = override_files
        if save_folder_suffix == '':
            self._save_folder_name = InstabilityAnalyzer.fn_instability_analysis
        else:
            self._save_folder_name = f'{InstabilityAnalyzer.fn_instability_analysis}{EXP_NAME_DIVIDER}{save_folder_suffix}'
        self._float_eps_query_job = float_eps_query_job

        #* setup logging / folders etc.
        self._setup(config)
        LOGGER.info(f'Setup instability analysis with config: \n{OmegaConf.to_yaml(config)}')

        LOGGER.info('Loading variables from sweep.')
        #* get k parameter (init_model_idx) values from sweep
        k_param_values = self.instability_sweep.get_sweep_param_values(init_model_idx_k_param_name)
        if len(k_param_values) == 0:
            raise ValueError(f'No successful hyperparameter job found for k parameter name: `{init_model_idx_k_param_name}`')

        if len(k_param_values) > 1:
            raise ValueError(
                f'Multiple hyperparemeters found for k parameter name: `{init_model_idx_k_param_name}` - Specify further!'
            )

        self._init_model_idx_k_param_name = list(k_param_values.keys())[0]

        #* parameter specifying the rewind point / number of pretraining steps/epochs
        self._all_init_idx_k_param_values = list(k_param_values.values())[0]
        # find subset of parameter values
        if isinstance(init_model_idxes_ks_or_every, int):
            if init_model_idxes_ks_or_every > 0:
                self._subset_init_idx_k_param_values = self._all_init_idx_k_param_values[::init_model_idxes_ks_or_every]
            else:
                self._subset_init_idx_k_param_values = self._all_init_idx_k_param_values
        elif isinstance(init_model_idxes_ks_or_every, (list, ListConfig)):
            # TODO add assert, check that all idxes available
            self._subset_init_idx_k_param_values = list(init_model_idxes_ks_or_every)
        else:
            raise ValueError(
                f'Unsupported type `{type(init_model_idxes_ks_or_every)}` for `init_model_idxes_or_every`.')
        LOGGER.info(f'Using init_model_idxes / k parameters: {self._subset_init_idx_k_param_values}')

        #* model indices for finetuned models
        self._interpolate_at_model_idxes = interpolate_at_model_idxes

        LOGGER.info(f'Finding seed combinations..')
        #* find seed combinations
        sweep_seeds = list(self.instability_sweep.get_sweep_param_values('seed').values())[0]
        if len(sweep_seeds) < 1:
            raise ValueError('Instability sweep contains no successfull runs!')
        # add seed from main_training run. 
        if self.main_training_job:
            sweep_seeds.append(self.main_training_job.seed)
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
        if hpparam_sweep is None and self.instability_sweep.sweep_cfg[SWEEP_TYPE_KEY] == SWEEP_TYPE_GRIDVAL:
            # use sweep parameters only if gridsearch was used
            hpparam_sweep = self.instability_sweep.sweep_cfg
            
        self._hpparam_sweep_cfg = hpparam_sweep

    def _setup(self, config: DictConfig) -> None:
        self._hp_result_folder_df = self.directory / InstabilityAnalyzer.fn_hp_result_df
        self._hp_result_folder_readable = self.directory / InstabilityAnalyzer.fn_hp_result_readable
        self._combined_results_folder = self.directory / InstabilityAnalyzer.fn_combined_results

        if self._save_results_to_disc:
            from erank.scripts import KEY_RUN_SCRIPT_KWARGS, KEY_RUN_SCRIPT_NAME
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

    @staticmethod
    def reload(sweep_result_dir: Union[SweepResult, Path, str],
               instability_folder_suffix: str = '') -> "InstabilityAnalyzer":
        """Reload an InstabilityAnalyzer from disc.

        Args:
            sweep_result_dir (Union[SweepResult, Path, str]): The sweep of this instability analysis
            instability_folder_suffix (str, optional): The instability analysis folder suffix. Defaults to ''.

        Returns:
            InstabilityAnalyzer: The InstabilityAnalyzer configured from files.
        """
        from erank.scripts import KEY_RUN_SCRIPT_KWARGS
        # load config from disc
        if isinstance(sweep_result_dir, str):
            sweep_result_dir = Path(sweep_result_dir)
        elif isinstance(sweep_result_dir, SweepResult):
            sweep_result_dir = sweep_result_dir.directory
        assert isinstance(sweep_result_dir, Path)
        insta_folder_name = f'{InstabilityAnalyzer.fn_instability_analysis}{EXP_NAME_DIVIDER}{instability_folder_suffix}' if instability_folder_suffix else InstabilityAnalyzer.fn_instability_analysis
        instability_analysis_folder = sweep_result_dir / insta_folder_name
        config_file = instability_analysis_folder / InstabilityAnalyzer.fn_config
        cfg = OmegaConf.load(config_file)
        cfg_instability = cfg[KEY_RUN_SCRIPT_KWARGS]

        # no logging and no changes on files
        cfg_instability.override_files = False
        cfg_instability.save_results_to_disc = False

        insta = InstabilityAnalyzer(**cfg_instability)
        return insta

    @property
    def remaining_hyperparams(self) -> Dict[str, Any]:
        """The remaining hyperparameters of the sweep. 
        These are the sweep hyperparameters without the hyperparameters necessary for the instability analysis, 
        i.e. the init_model_idx_k_param_name and the seed."""
        sweep_params = self.instability_sweep.get_sweep_param_values()
        # remove seed and k_param_name
        _ = sweep_params.pop('seed')
        _ = sweep_params.pop(self._init_model_idx_k_param_name)
        return sweep_params

    @property
    def directory(self) -> Path:
        """The root directory of the instability analysis."""
        return self.instability_sweep.directory / self._save_folder_name

    @property
    def hp_result_folder_df(self) -> Path:
        """Folder of single hyperparameter results in pickle format."""
        return self._hp_result_folder_df

    @property
    def hp_result_folder_readable(self) -> Path:
        """Folder of single hyperparameter results in readable format."""
        return self._hp_result_folder_readable

    @property
    def combined_results_folder(self) -> Path:
        """The (combined) results folder."""
        return self._combined_results_folder

    @property
    def combined_results(self) -> List[str]:
        """Return the names of combined results in ascending order (latest results last)."""
        return sorted(
            [p.stem for p in self.combined_results_folder.glob(f'*.{InstabilityAnalyzer.save_pickle_format}')],
            reverse=False)

    @property
    def combined_results_dfs(self) -> Dict[str, pd.DataFrame]:
        """The latest result dataframes."""
        return self.get_combined_results_dfs(idx=-1)

    def get_combined_results_dfs(self, idx: int = -1) -> Dict[str, pd.DataFrame]:
        """Return a dictionary with all result tables (DataFrames).
        List is sorted chronologically after creation time. See `combined_results`.

        Args:
            idx (int, optional): The index in the `combined_results` list. Defaults to -1.

        Returns:
            Dict[str, pd.DataFrame]: The combined results.
        """
        combined_result_name = self.combined_results[idx]
        combined_result_file = self.combined_results_folder / f'{combined_result_name}.{InstabilityAnalyzer.save_pickle_format}'
        with combined_result_file.open(mode='rb') as f:
            combined_result_dfs = pickle.load(f)
        return combined_result_dfs

    def instability_analysis_for_hpparam(self,
                                         hypparam_sel: Dict[str, Any] = {},
                                         use_tqdm: bool = True,
                                         float_eps: float = None) -> Dict[str, pd.DataFrame]:
        """Run the instability analysis for a single hyperparameter configuration.
        Results will be stored separately, if configured.

        Args:
            hypparam_sel (Dict[str, Any], optional): The hyperparameter selection dictionary. 
                                                     Keys are dot-separated hyperparameters. Defaults to {}.
            use_tqdm (bool, optional): Show progress bar. Defaults to True.
            float_eps (float, optional): Epsilon value for float hyperparameter value comparison for querying jobs from the sweep. 
                                         If None use preconfigured value. Defaults to None.

        Returns:
            Dict[str, pd.DataFrame]: The result dataframes for this hyperparameter configuration.
        """
        # create run_dict: init_model_idx_k_param_value -> runs with different seeds
        f_eps = self._float_eps_query_job if float_eps is None else float_eps
        run_dict = self._create_run_dict(hypparam_sel=hypparam_sel, float_eps=f_eps)

        # create dataset_generator and avoid reloading the dataset on every iteration
        # Get any jobresult from the run_dict. They all have the same data config.
        first_jobresult = list(list(run_dict.values())[0].values())[0]
        data_cfg = copy.deepcopy(first_jobresult.config.config.data)
        # set training augmentations to the validation augmentations
        # typically we use augmentations on train data; we do not want to have them for instability analysis on train data
        with open_dict(data_cfg):
            data_cfg.train_split_transforms = data_cfg.get('val_split_transforms', {})
        ds_generator = DatasetGenerator(**data_cfg)
        ds_generator.generate_dataset()

        dataset_dfs, distance_dfs = {}, {}
        it = run_dict.items()
        if use_tqdm:
            it = tqdm(it, file=sys.stdout)
        # iterate over run_dict and do interpolation for seed_combinations and train_model_idxes
        for init_model_idx_k, k_dict in it:
            if isinstance(it, tqdm):
                it.set_description_str(desc=f'init_model_idx_k={init_model_idx_k}')
            # for every init_model_idx_k_param_value, there must be jobs with all used seeds.
            assert set(self._used_seeds).issubset(set(
                k_dict.keys())), f'Some seeds are missing for hyperparameter selection: {hypparam_sel}, init_model_idx_k: {init_model_idx_k}'

            k_runs = dataset_dfs.get(init_model_idx_k, None)
            if k_runs is None:
                dataset_dfs[init_model_idx_k] = []
                distance_dfs[init_model_idx_k] = []

            for sc in self.seed_combinations:
                run_0, run_1 = k_dict[sc[0]], k_dict[sc[1]]
                for interpolate_at_model_idx in self._interpolate_at_model_idxes:
                    interp_result_ds_df, interp_result_dist_df = interpolate_linear_runs(
                        run_0=run_0,
                        run_1=run_1,
                        score_fn=self.score_fn,
                        model_idx=interpolate_at_model_idx,
                        interpolation_factors=torch.tensor(self.interpolation_factors),
                        interpolate_linear_kwargs=self._interpolate_linear_kwargs,
                        device=self.device,
                        return_dataframe=True,
                        dataset_generator=ds_generator)
                    dataset_dfs[init_model_idx_k].append(interp_result_ds_df)
                    if not interp_result_dist_df is None:
                        distance_dfs[init_model_idx_k].append(interp_result_dist_df)

            # create a dataframe for every init_model_idx_k
            dataset_dfs[init_model_idx_k] = pd.concat(dataset_dfs[init_model_idx_k])
            if distance_dfs[init_model_idx_k]:
                distance_dfs[init_model_idx_k] = pd.concat(distance_dfs[init_model_idx_k])

        # concatenate all dataframes
        dataset_result_df = pd.concat(dataset_dfs, names=[PARAM_NAME_INIT_MODEL_IDX_K])
        combined_results = {
            InstabilityAnalyzer.key_dataset_result_df: dataset_result_df,
        }
        if len(list(distance_dfs.values())[0]) > 0:
            distance_result_df = pd.concat(distance_dfs, names=[PARAM_NAME_INIT_MODEL_IDX_K])
            combined_results[InstabilityAnalyzer.key_distance_result_df] = distance_result_df

        return combined_results

    def _create_run_dict(self,
                         hypparam_sel: Dict[str, Any] = {},
                         float_eps: float = 1e-3) -> Dict[int, Dict[int, JobResult]]:
        """Create a dictionary containing all runs for an instability analysis run.
        If hypparam_sel={}, add the main_training_job_dir

        Returns:
            Dict[int, Dict[int, JobResult]]: Dictionary with all runs necessary for instability analysis.
                                             Hierarchy: init_model_idx_k -> seed -> jobresult
        """
        hp_sel = copy.deepcopy(hypparam_sel)
        run_dict = {}
        for k in self._subset_init_idx_k_param_values:
            add_hp_sel = {self._init_model_idx_k_param_name: k, 'seed': self._used_seeds}
            hp_sel.update(add_hp_sel)
            _, jobs = self.instability_sweep.query_jobs(hp_sel, float_eps=float_eps)
            k_dict = {job.seed: job for job in jobs}
            if self.main_training_job:
                k_dict[self.main_training_job.seed] = self.main_training_job
            run_dict[k] = k_dict

        return run_dict

    def instability_analysis(self, use_tqdm: bool = True, override_files: bool = False) -> Dict[str, pd.DataFrame]:
        """Perform mutiple instability analyses for all hyperparameter configurations.

        Args:
            use_tqdm (bool, optional): Show progress bar. Defaults to True.
            override_files (bool, optional): Override hyperparameter result files on re-run, use previous resuls otherwise. 
                                             Defaults to False.

        Returns:
            Dict[str, pd.DataFrame]: Return combined result tables.
        """
        LOGGER.info(f'Starting instability analysis..')
        if self._hpparam_sweep_cfg is not None:
            # create sweep
            sweep = Sweeper.create(sweep_config=copy.deepcopy(self._hpparam_sweep_cfg))
            sweep.drop_axes(ax_param_names=[self._init_model_idx_k_param_name, 'experiment_data.seed'])
            hp_combinations = sweep.generate_sweep_parameter_combinations(flatten_hierarchical_dicts=True)
            hp_combinations_str = [hyp_param_cfg_to_str(hp) for hp in hp_combinations]
        else:
            LOGGER.info('No sweep hyperparameters specified or found: Doing single instability analysis.')
            hp_combinations = [{}]
            hp_combinations_str = ['default_params']
        LOGGER.info(f'Number of hyperparameter combinations for instability analysis: {len(hp_combinations)}')

        # perform instability analysis
        dataset_result_dfs = []
        distance_result_dfs = []
        hps = list(zip(hp_combinations, hp_combinations_str))
        if use_tqdm:
            hps = tqdm(hps, file=sys.stdout, desc='HP combinations')

        for hp_sel, hp_str in hps:
            if self._hp_result_df_exists(hp_str) and not (override_files or self._override_files):
                LOGGER.info(f'Params `{hp_str}`: load&skip')
                df_dict = self.load_instability_analysis_for_hpparam(hp_str)
            else:
                LOGGER.info(f'Params `{hp_str}`: compute')
                df_dict = self.instability_analysis_for_hpparam(hypparam_sel=hp_sel, use_tqdm=True)
                self._save_hp_result_dfs(df_dict, hp_str)

            dataset_result_dfs.append(df_dict[InstabilityAnalyzer.key_dataset_result_df])
            dist_df = df_dict.get(InstabilityAnalyzer.key_distance_result_df, None)
            if not dist_df is None:
                distance_result_dfs.append(dist_df)

        # create multiindex
        if hp_combinations[0] == {}:
            # only default parameters are used
            index = pd.MultiIndex.from_arrays([['default_params']], names=['default_params'])
        else:
            hp_lists = convert_listofdicts_to_dictoflists(hp_combinations)
            hp_names = list(hp_lists.keys())
            index = pd.MultiIndex.from_arrays(list(hp_lists.values()), names=hp_names)

        dataset_result_df = pd.concat(dataset_result_dfs, keys=index)
        combined_results = {InstabilityAnalyzer.key_dataset_result_df: dataset_result_df}
        if len(distance_result_dfs) > 0:
            distance_result_df = pd.concat(distance_result_dfs, keys=index)
            combined_results[InstabilityAnalyzer.key_distance_result_df] = distance_result_df

        combined_results_file = self._save_combined_result_dfs(combined_results)

        LOGGER.info(f'Done. \nCombined results in file `{str(combined_results_file)}`.')

        return combined_results

    def load_instability_analysis_for_hpparam(self, hp_sel_str: str) -> Dict[str, pd.DataFrame]:
        """Load instability analyis result table for a single hyperparameter selection.
        TODO: this method should also take hyper parameter dictionary as argument.
        Args:
            hp_sel_str (str): The hyperparameter selection string. Converted from dictionary.

        Returns:
            Dict[str, pd.DataFrame]: Result dataframes for this hyperparameter configuration.
        """
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
