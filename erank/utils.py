import torch
import logging
from pathlib import Path
from typing import Dict, Tuple, Union
from torch import nn
from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.torch_models import get_model_class
from ml_utilities.trainers.basetrainer import BEST_MODEL_FILENAME, RUN_PROGRESS_MEASURE_STEP, RUN_PROGRESS_MEASURE_EPOCH
from omegaconf import OmegaConf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def get_best_model_idx(
    run_path: Union[str, Path],
    possible_specifiers: Tuple[str] = (RUN_PROGRESS_MEASURE_EPOCH, RUN_PROGRESS_MEASURE_STEP)
) -> Tuple[int, str]:
    if isinstance(run_path, str):
        run_path = Path(run_path)
    for specifier in possible_specifiers:
        best_model_file = run_path / BEST_MODEL_FILENAME.format(specifier=specifier)
        if best_model_file.exists():
            with best_model_file.open('r') as f:
                best_idx = int(f.read())
            return (best_idx, specifier)
    raise ValueError(f'No best file found for run: {run_path}.')


def load_model_from_idx(run_path: Union[str, Path],
                        idx: int,
                        device: Union[torch.device, str, int] = "auto",
                        save_name_num_epoch_digits: int = -1) -> BaseModel:
    if isinstance(run_path, str):
        run_path = Path(run_path)
    # load config
    loaded_config = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    config = loaded_config.config
    # get model class
    model_name = config.model.name
    model_class = get_model_class(model_name)
    # get run progress measure

    progress_measure = RUN_PROGRESS_MEASURE_EPOCH if config.trainer.get('n_epochs',
                                                                        -1) > 0 else RUN_PROGRESS_MEASURE_STEP

    model = model_class.load(run_path,
                             model_class.model_save_name(idx,
                                                         specifier=progress_measure,
                                                         num_digits=save_name_num_epoch_digits),
                             device=device)
    return model


def load_best_model(run_path: Union[str, Path],
                    device: Union[torch.device, str, int] = "auto",
                    save_name_num_epoch_digits: int = -1) -> BaseModel:
    best_idx, _ = get_best_model_idx(run_path)

    model = load_model_from_idx(run_path,
                                best_idx,
                                save_name_num_epoch_digits=save_name_num_epoch_digits,
                                device=device)
    return model


def load_directions_matrix_from_task_sweep(path_to_runs: Union[str, Path],
                                           num_runs: int = -1,
                                           device: Union[torch.device, str, int] = "auto",
                                           use_absolute_model_params: bool = False,
                                           glob_pattern: str = '*',
                                           save_name_num_epoch_digits: int = -1) -> torch.Tensor:
    """Load parameter matrix, where ´num_runs´ models are stacked. 

    Args:
        path_to_runs (Union[str, Path]): Path to the runs. 
        num_runs (int, optional): Number of runs to stack. If num_runs = -1, use all runs. Defaults to -1.
        device (Union[torch.device, str, int], optional): The device. Defaults to "auto".
        use_absolute_model_params (bool, optional): Whether to use the absolute parameters of the best models 
                or the difference between the best model and its initialization. Defaults to False.
        glob_pattern (str, optional): A glob pattern. Can be used to filter runs in the run directory.
        save_name_num_epoch_digits (int, optional): Number of digits in model save name (mainly for compatibility).

    Returns:
        torch.Tensor: The directions matrix.
    """
    if isinstance(path_to_runs, str):
        path_to_runs = Path(path_to_runs)

    assert path_to_runs.exists() and path_to_runs.is_dir(), f'Load path {path_to_runs} is no directory.'

    run_list = list(path_to_runs.glob(glob_pattern))

    if num_runs < 0:
        num_runs = len(run_list)
    elif num_runs > len(run_list):
        raise ValueError(
            f'Try to load {num_runs} runs, but the directory {str(path_to_runs)} contains only {len(run_list)} runs with glob pattern `{glob_pattern}`!'
        )

    directions = []
    pbar = tqdm(sorted(run_list[:num_runs]))
    for run_path in pbar:
        pbar.set_description_str(f'Loading {run_path}')

        best_model = load_best_model(run_path, save_name_num_epoch_digits=save_name_num_epoch_digits, device=device)
        with torch.no_grad():
            best_model_vec = nn.utils.parameters_to_vector(best_model.parameters())

        if not use_absolute_model_params:
            init_model = load_model_from_idx(run_path,
                                             0,
                                             save_name_num_epoch_digits=save_name_num_epoch_digits,
                                             device=device)
            # compute direction vec
            with torch.no_grad():
                init_model_vec = nn.utils.parameters_to_vector(init_model.parameters())
                direction = best_model_vec - init_model_vec
        else:
            direction = best_model_vec

        directions.append(direction)

    directions_matrix = torch.stack(directions)
    return directions_matrix


def load_multiple_dir_matrices_from_sweep(path_to_runs: Union[str, Path],
                                          name_run_glob_pattern_dict: Dict[str, str],
                                          combine_name_pattern: bool = True,
                                          num_runs: int = -1,
                                          use_absolute_model_params: bool = False,
                                          device: Union[torch.device, str, int] = "auto") -> Dict[str, torch.Tensor]:
    """Returns multiple direction matrices in a dictionary. 
    The dictionary has as keys the name (and the glob pattern for the runs (optionally)) and as values the respective matrix.
    """
    model_dict = {}
    LOGGER.info(f'Loading {len(name_run_glob_pattern_dict)} matrices from directory: {str(path_to_runs)}.')
    for i, (name, glob_pattern) in enumerate(name_run_glob_pattern_dict.items()):
        LOGGER.info(f'Matrix {i+1}/{len(name_run_glob_pattern_dict)}:')
        if combine_name_pattern:
            key = f'{name}#{glob_pattern}'
        else:
            key = name
        model_matrix = load_directions_matrix_from_task_sweep(path_to_runs=path_to_runs,
                                                              glob_pattern=glob_pattern,
                                                              num_runs=num_runs,
                                                              use_absolute_model_params=use_absolute_model_params,
                                                              device=device)
        model_dict[key] = model_matrix
    return model_dict