import logging
from typing import Any, Dict, List, Union
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from ml_utilities.logger import Logger
from ml_utilities.torch_models import get_model_class
from ml_utilities.torch_utils import get_loss
from ml_utilities.trainers.basetrainer import BaseTrainer
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler
from ml_utilities.torch_utils.metrics import get_metric_collection
from erank.regularization import get_regularizer_class
from erank.regularization.regularized_loss import RegularizedLoss
from erank.regularization.subspace_regularizer import SubspaceRegularizer

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
                         early_stopping_patience=config.trainer.early_stopping_patience,
                         num_workers=config.trainer.num_workers,
                         save_every_best_model=config.trainer.get('save_every_best_model', False))
        #
        self._subspace_regularizer: SubspaceRegularizer = None
        self._log_train_step_every = self.config.trainer.get('log_train_step_every', 1)
        self._log_additional_train_step_every_multiplier = self.config.trainer.get(
            'log_additional_train_step_every_multiplier', 1)
        self._log_additional_logs = self.config.trainer.get('log_additional_logs', False)

        exp_data = self.config.experiment_data
        wandb_args = self.config.get('wandb', {})
        if isinstance(wandb_args, DictConfig):
            wandb_args = OmegaConf.to_container(wandb_args, resolve=True, throw_on_missing=True)
            
        self._logger = Logger(job_name=exp_data.job_name,
                              job_dir=exp_data.experiment_dir,
                              project_name=exp_data.project_name,
                              entity_name=exp_data.get('entity', None),
                              config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
                              wandb_args=wandb_args)
        self._logger.setup_logger()

    def _create_model(self) -> None:
        LOGGER.info(f'Creating model: {self.config.model.name}')
        model_class = get_model_class(self.config.model.name)
        if self.config.trainer.init_model:
            LOGGER.info(f'Loading model {self.config.trainer.init_model} to device {self.device}.')
            self._model = model_class.load(self.config.trainer.init_model, device=self.device)
        else:
            self._model = model_class(**self.config.model.model_kwargs)

        self._logger.watch_model(self._model)

    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        LOGGER.info('Creating optimizer and scheduler.')
        self._optimizer, self._lr_scheduler = create_optimizer_and_scheduler(model.parameters(),
                                                                             **self.config.trainer.optimizer_scheduler)

    def _create_loss(self) -> None:
        LOGGER.info('Creating loss.')
        loss_cls = get_loss(self.config.trainer.loss)
        loss_module = loss_cls(reduction='mean')
        self._loss = RegularizedLoss(loss_module)

        self._subspace_regularizer = self._create_subspace_regularizer(self._model)
        if self._subspace_regularizer:
            self._loss.add_regularizer(self._subspace_regularizer)

    def _create_subspace_regularizer(self, model: nn.Module) -> SubspaceRegularizer:
        subspace_reg_cfg = self.config.trainer.get('regularizer', None)
        subspace_reg = None

        if subspace_reg_cfg is None or subspace_reg_cfg.type == 'none':
            LOGGER.info('No regularizer.')
        elif subspace_reg_cfg.init_type in ['random', 'weightsdiff', 'buffer']:
            LOGGER.info(f'Subspace regularizer: {subspace_reg_cfg.type}')
            regularizer_cls = get_regularizer_class(subspace_reg_cfg.type)
            regularizer_kwargs = subspace_reg_cfg.regularizer_kwargs
            subspace_reg = regularizer_cls(init_model=model, device=self.device, **regularizer_kwargs)

            LOGGER.info(f'Subspace regularization of init_type {subspace_reg_cfg.init_type}.')
            if subspace_reg_cfg.init_type == 'random':
                subspace_reg.init_subspace_vecs(random_buffer=True)
            elif subspace_reg_cfg.init_type == 'weightsdiff':
                dir_buffer_path = subspace_reg_cfg.get('init_dir_buffer', None)
                if dir_buffer_path is None:
                    raise ValueError(f'Subspace init_type is `weightsdiff`, but no buffer path is given!')
                subspace_reg.init_subspace_vecs(path_to_buffer_or_runs=dir_buffer_path)
            elif subspace_reg_cfg.init_type == 'buffer':
                pass  # does nothing, buffer is filled during training
        else:
            raise ValueError(f'Unknown regularizer type: {subspace_reg_cfg.init_type}')
        return subspace_reg

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        metric_names = self.config.trainer.metrics
        metrics = get_metric_collection(metric_names=metric_names)
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _finish_train_epoch(self,
                            epoch: int,
                            losses_epoch: Union[Dict[str, Union[List[torch.Tensor], torch.Tensor]],
                                                List[Dict[str, torch.Tensor]]] = {},
                            metrics_epoch: Dict[str, Union[float, torch.Tensor]] = {}) -> None:
        metrics_epoch.update({'time_last_train_epoch_in_s': self._time_last_train_epoch})
        self._log_losses_metrics('train', epoch, losses_epoch, metrics_epoch)
        self._reset_metrics()

    def _finish_val_epoch(self,
                          epoch: int,
                          losses_epoch: Union[Dict[str, Union[List[torch.Tensor], torch.Tensor]],
                                              List[Dict[str, torch.Tensor]]] = {},
                          metrics_epoch: Dict[str, Union[float, torch.Tensor]] = {}) -> float:
        metrics_epoch.update({'time_last_val_epoch_in_s': self._time_last_val_epoch})
        self._log_losses_metrics('val', epoch, losses_epoch, metrics_epoch)

        # val_score is first metric in self._val_metrics
        val_score = metrics_epoch[next(iter(self._val_metrics.items()))[0]].item()
        self._reset_metrics()
        return val_score

    def _log_losses_metrics(self,
                            prefix: str,
                            epoch: int,
                            losses_epoch: Union[Dict[str, Union[List[torch.Tensor], torch.Tensor]],
                                                List[Dict[str, torch.Tensor]]] = {},
                            metrics_epoch: Dict[str, Any] = {},
                            log_to_console: bool = True) -> None:
        self._logger.log_keys_vals(prefix=prefix,
                                   epoch=epoch,
                                   train_step=self._train_step,
                                   keys_multiple_vals=losses_epoch,
                                   keys_val=metrics_epoch,
                                   log_to_console=log_to_console)

    def _final_hook(self, final_results: Dict[str, Any], *args, **kwargs):
        self._logger.finish(final_results=final_results)
