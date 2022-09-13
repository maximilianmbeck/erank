import torch
import logging
from pathlib import Path
from typing import Dict, Union
from torch import nn
from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.torch_models import get_model_class
from omegaconf import OmegaConf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def load_model_from_epoch(run_path: Union[str, Path],
                          epoch: int,
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

    model = model_class.load(run_path, model_class.model_save_name(epoch, save_name_num_epoch_digits), device=device)
    return model


def load_best_model(run_path: Union[str, Path],
                    device: Union[torch.device, str, int] = "auto",
                    save_name_num_epoch_digits: int = -1) -> BaseModel:
    # get best epoch
    best_epoch_file = run_path / 'best_epoch.txt'
    if not best_epoch_file.exists():
        raise ValueError(f'No best epoch file found for run: {run_path}.')

    with best_epoch_file.open('r') as f:
        best_epoch = int(f.read())

    model = load_model_from_epoch(run_path,
                                  best_epoch,
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
            f'Try to load {num_runs} runs, but the directory {str(path_to_runs)} contains only {len(run_list)} runs with glob patter `{glob_pattern}`!'
        )

    directions = []
    pbar = tqdm(sorted(run_list[:num_runs]))
    for run_path in pbar:
        pbar.set_description_str(f'Loading {run_path}')

        best_model = load_best_model(run_path, save_name_num_epoch_digits=save_name_num_epoch_digits, device=device)
        with torch.no_grad():
            best_model_vec = nn.utils.parameters_to_vector(best_model.parameters())

        if not use_absolute_model_params:
            init_model = load_model_from_epoch(run_path,
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