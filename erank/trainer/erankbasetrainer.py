import logging
from typing import Any, Dict, List, Union
import torch
import wandb
import pandas as pd
from torch import nn
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from ml_utilities.utils import convert_dict_to_python_types
from ml_utilities.torch_models import get_model_class
from ml_utilities.torch_utils import get_loss
from ml_utilities.trainers.basetrainer import BaseTrainer
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler
from hydra.core.hydra_config import HydraConfig
from erank.regularization import EffectiveRankRegularization

LOGGER = logging.getLogger(__name__)

class ErankBaseTrainer(BaseTrainer):
    """Abstract trainer for this project. Collects all common functionalitites across trainers.

    Functionality is further specialized in child classes.

    Args:
        config (DictConfig): The configuration.
    """

    def __init__(self, config: DictConfig):

        self.config = config
        super().__init__(experiment_dir=config.experiment_data.experiment_dir,
                         seed=config.experiment_data.seed,
                         gpu_id=config.experiment_data.gpu_id,
                         n_epochs=config.trainer.n_epochs,
                         val_every=config.trainer.val_every,
                         save_every=config.trainer.save_every,
                         early_stopping_patience=config.trainer.early_stopping_patience)
        #
        self._erank_regularizer: EffectiveRankRegularization = None

    def _setup(self):
        LOGGER.info('Starting wandb.')
        exp_data = self.config.experiment_data
        wandb.init(entity=exp_data.get('entity', None),
                   project=exp_data.project_name,
                   name=HydraConfig.get().job.name,
                   dir=Path.cwd(),
                   config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
                   **self.config.wandb.init,
                   settings=wandb.Settings(start_method='fork'))

    def _create_model(self) -> None:
        LOGGER.info(f'Creating model: {self.config.model.name}')
        model_class = get_model_class(self.config.model.name)
        if self.config.trainer.init_model:
            LOGGER.info(f'Loading model {self.config.trainer.init_model} to device {self.device}.')
            self._model = model_class.load(self.config.trainer.init_model, device=self.device)
        else:
            self._model = model_class(**self.config.model.model_kwargs)

        wandb.watch(self._model, **self.config.wandb.watch)

    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        LOGGER.info('Creating optimizer and scheduler.')
        self._optimizer, self._lr_scheduler = create_optimizer_and_scheduler(model.parameters(),
                                                                             **self.config.trainer.optimizer_scheduler)

    def _create_loss(self) -> None:
        LOGGER.info('Creating loss.')
        loss_cls = get_loss(self.config.trainer.loss)
        self._loss = loss_cls(reduction='mean')

        self._erank_regularizer = self._create_erank_regularizer(self._model)

    def _create_erank_regularizer(self, model: nn.Module) -> EffectiveRankRegularization:
        erank_cfg = self.config.trainer.get('erank', None)
        erank_reg = None
        if erank_cfg is None or erank_cfg.type == 'none':
            LOGGER.info('No erank regularizer.')
        elif erank_cfg.type in ['random', 'pretraindiff']:
            LOGGER.info(f'Erank regularization of type {erank_cfg.type}.')
            erank_reg = EffectiveRankRegularization(buffer_size=erank_cfg.buffer_size,
                                                    init_model=model,
                                                    loss_weight=erank_cfg.loss_weight,
                                                    normalize_directions=erank_cfg.get('norm_directions', False),
                                                    use_abs_model_params=erank_cfg.get('use_abs_model_params', True))
            if erank_cfg.type == 'random':
                erank_reg.init_directions_buffer(random_buffer=True)
            elif erank_cfg.type == 'pretraindiff':
                erank_reg.init_directions_buffer(path_to_buffer_or_runs=erank_cfg.dir_buffer)
        else:
            raise ValueError(f'Unknown erank type: {erank_cfg.type}')
        return erank_reg

    def _finish_train_epoch(self,
                            epoch: int,
                            losses_epoch: Dict[str, List[float]] = {},
                            metrics_epoch: Dict[str, Union[float, torch.Tensor]] = {}) -> None:
        for loss_name, loss_val_list in losses_epoch.items():
            if isinstance(loss_val_list, list):
                losses_epoch[loss_name] = torch.tensor(loss_val_list).mean().item()

        log_dict = {'epoch': epoch, 'train_step': self._train_step, **losses_epoch, **metrics_epoch}
        wandb.log({'train_epoch/': log_dict})

        LOGGER.info(f'Train epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

        self._reset_metrics()

    def _finish_val_epoch(self,
                          epoch: int,
                          losses_epoch: Dict[str, List[float]] = {},
                          metrics_epoch: Dict[str, Union[float, torch.Tensor]] = {}) -> float:
        for loss_name, loss_val_list in losses_epoch.items():
            losses_epoch[loss_name] = torch.tensor(loss_val_list).mean().item()

        # log epoch
        log_dict = {'epoch': epoch, **losses_epoch, **metrics_epoch}
        wandb.log({'val/': log_dict})

        LOGGER.info(f'Val epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

        # val_score is first metric in self._val_metrics
        val_score = metrics_epoch[next(iter(self._val_metrics.items()))[0]].item()
        self._reset_metrics()
        return val_score

    def _final_hook(self, final_results: Dict[str, Any], *args, **kwargs):
        wandb.run.summary.update(final_results)
