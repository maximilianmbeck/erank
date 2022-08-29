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
from ml_utilities.torch_utils.metrics import get_metric_collection
from hydra.core.hydra_config import HydraConfig
from erank.regularization.regularized_loss import RegularizedLoss
from erank.regularization.subspace_regularizer import SubspaceRegularizer
from erank.regularization.erank_regularizer import EffectiveRankRegularizer

LOGGER = logging.getLogger(__name__)


class SubspaceBaseTrainer(BaseTrainer):
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
        self._subspace_regularizer: SubspaceRegularizer = None
        self._log_train_epoch_every = self.config.trainer.get('log_train_epoch_every', 1)


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
        loss_module = loss_cls(reduction='mean')

        self._subspace_regularizer = self._create_erank_regularizer(self._model)
        self._loss = RegularizedLoss(loss_module)
        if self._subspace_regularizer:
            self._loss.add_regularizer(self._subspace_regularizer)

    def _create_erank_regularizer(self, model: nn.Module) -> EffectiveRankRegularizer:
        erank_cfg = self.config.trainer.get('erank', None)
        erank_reg = None
        if erank_cfg is None or erank_cfg.type == 'none':
            LOGGER.info('No erank regularizer.')
        elif erank_cfg.type in ['random', 'weightsdiff', 'buffer']:
            LOGGER.info(f'Erank regularization of type {erank_cfg.type}.')
            erank_kwargs = erank_cfg.erank_kwargs
            erank_reg = EffectiveRankRegularizer(init_model=model, device=self.device, **erank_kwargs)
            if erank_cfg.type == 'random':
                erank_reg.init_subspace_vecs(random_buffer=True)
            elif erank_cfg.type == 'weightsdiff':
                dir_buffer_path = erank_cfg.get('dir_buffer', None)
                if dir_buffer_path is None:
                    raise ValueError(f'Erank type is `weightsdiff`, but no buffer path is given!')
                erank_reg.init_subspace_vecs(path_to_buffer_or_runs=dir_buffer_path)
            elif erank_cfg.type == 'buffer':
                pass # does nothing, buffer is filled during training
        else:
            raise ValueError(f'Unknown erank type: {erank_cfg.type}')
        return erank_reg

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        metric_names = self.config.trainer.metrics
        metrics = get_metric_collection(metric_names=metric_names)
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _finish_train_epoch(self,
                            epoch: int,
                            losses_epoch: Dict[str, Union[List[float], float]] = {},
                            metrics_epoch: Dict[str, Union[float, torch.Tensor]] = {}) -> None:
        if self._log_train_epoch_every > 0 and epoch % self._log_train_epoch_every == 0:
            self._log_losses_metrics('train', epoch, losses_epoch, metrics_epoch)
            self._reset_metrics()

    def _finish_val_epoch(self,
                          epoch: int,
                          losses_epoch: Dict[str, Union[List[float], float]] = {},
                          metrics_epoch: Dict[str, Union[float, torch.Tensor]] = {}) -> float:
        self._log_losses_metrics('val', epoch, losses_epoch, metrics_epoch)

        # val_score is first metric in self._val_metrics
        val_score = metrics_epoch[next(iter(self._val_metrics.items()))[0]].item()
        self._reset_metrics()
        return val_score

    def _log_losses_metrics(self,
                            prefix: str,
                            epoch: int,
                            losses_epoch: Dict[str, Union[List[float], float]] = {},
                            metrics_epoch: Dict[str, Any] = {},
                            log_to_console: bool = True) -> None:
        for loss_name, loss_vals in losses_epoch.items():
            if isinstance(loss_vals, list):
                losses_epoch[loss_name] = torch.tensor(loss_vals).mean().item()

        # log epoch
        log_dict = {'epoch': epoch, 'train_step': self._train_step, **losses_epoch, **metrics_epoch}
        wandb.log({f'{prefix}/': log_dict})
        if log_to_console:
            LOGGER.info(f'{prefix} epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

    def _final_hook(self, final_results: Dict[str, Any], *args, **kwargs):
        wandb.run.summary.update(final_results)
