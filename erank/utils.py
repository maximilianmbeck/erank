import torch
import logging
from pathlib import Path
from typing import Union
from torch import nn
from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.torch_models import get_model_class
from omegaconf import OmegaConf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def load_model_from_epoch(
        run_path: Union[str, Path],
        epoch: int, device: Union[torch.device, str, int] = "auto") -> BaseModel:
    if isinstance(run_path, str):
        run_path = Path(run_path)
    # load config
    loaded_config = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    config = loaded_config.config
    # get model class
    model_name = config.model.name
    model_class = get_model_class(model_name)

    model = model_class.load(run_path, model_class.model_save_name(epoch), device=device)
    return model


def load_best_model(run_path: Union[str, Path], device: Union[torch.device, str, int] = "auto") -> BaseModel:
    # get best epoch
    best_epoch_file = run_path / 'best_epoch.txt'
    if not best_epoch_file.exists():
        raise ValueError(f'No best epoch file found for run: {run_path}.')

    with best_epoch_file.open('r') as f:
        best_epoch = int(f.read())

    model = load_model_from_epoch(run_path, best_epoch, device=device)
    return model


def load_directions_matrix_from_task_sweep(
        path_to_runs: Union[str, Path],
        device: Union[torch.device, str, int] = "auto", use_absolute_model_params: bool = False) -> torch.Tensor:
    if isinstance(path_to_runs, str):
        path_to_runs = Path(path_to_runs)

    assert path_to_runs.exists() and path_to_runs.is_dir(), f'Load path {path_to_runs} is no directory.'

    directions = []

    pbar = tqdm(list(path_to_runs.iterdir()))
    for run_path in pbar:
        pbar.set_description_str(f'Loading {run_path}')

        best_model = load_best_model(run_path, device=device)
        with torch.no_grad():
            best_model_vec = nn.utils.parameters_to_vector(best_model.parameters())
        
        if not use_absolute_model_params:
            init_model = load_model_from_epoch(run_path, 0, device=device)
            # compute direction vec
            with torch.no_grad():
                init_model_vec = nn.utils.parameters_to_vector(init_model.parameters())
                direction = best_model_vec - init_model_vec   
        else:
            direction = best_model_vec

        directions.append(direction)

    directions_matrix = torch.stack(directions)
    return directions_matrix
