import logging
import sys
from typing import Any, Dict
import wandb
import torch
import torchmetrics
import torch.utils.data as data
from tqdm import tqdm
from torch import nn
from omegaconf import DictConfig

from ml_utilities.torch_utils.metrics import EntropyCategorical, MaxClassProbCategorical
from erank.trainer.erankbasetrainer import ErankBaseTrainer
from erank.data import get_dataset_provider
from erank.data.data_utils import random_split_train_tasks

LOGGER = logging.getLogger(__name__)


class SupervisedTrainer(ErankBaseTrainer):
    """Class for training in a supervised setting.

    Args:
        config (DictConfig): Configuration.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Supervised Trainer.')

    def _create_datasets(self) -> None:
        LOGGER.info('Loading train/val dataset.')
        data_cfg = self.config.data
        provide_dataset = get_dataset_provider(dataset_name=data_cfg.dataset)
        train_dataset = provide_dataset(data_cfg.dataset_kwargs)
        train_set, val_set = random_split_train_tasks(train_dataset, **data_cfg.dataset_split)
        LOGGER.info(f'Size of training/validation set: ({len(train_set)}/{len(val_set)})')
        self._datasets = dict(train=train_set, val=val_set)

    def _create_dataloaders(self) -> None:
        train_loader = data.DataLoader(
            dataset=self._datasets['train'],
            batch_size=self.config.trainer.batch_size, shuffle=True, drop_last=False,
            num_workers=self.config.trainer.num_workers)
        val_loader = data.DataLoader(dataset=self._datasets['val'], batch_size=self.config.trainer.batch_size,
                                     shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)
        self._loaders = dict(train=train_loader, val=val_loader)

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        metrics = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(),
             EntropyCategorical(),
             MaxClassProbCategorical()])
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        losses_epoch = dict(loss_total=[], loss_ce=[])
        if self._erank_regularizer is not None:
            losses_epoch.update(dict(loss_erank=[]))

        # training loop (iterations per epoch)
        pbar = tqdm(self._loaders['train'], desc=f'Train epoch {epoch}', file=sys.stdout)
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)
            # forward pass
            y_pred = self._model(xs)

            loss = self._loss(y_pred, ys)

            # add erank regularizer
            loss_reg = torch.tensor(0.0).to(loss)
            loss_weight = 0.0
            if self._erank_regularizer is not None:
                loss_reg = self._erank_regularizer.forward(self._model)
                loss_weight = self._erank_regularizer.loss_weight

            loss_total = loss + loss_weight * loss_reg

            # backward pass
            self._optimizer.zero_grad()
            loss_total.backward()
            self._optimizer.step()
            self._train_step += 1

            # update regularizer
            if self._erank_regularizer is not None:
                self._erank_regularizer.update_delta_start_params(self._model)

            # metrics & logging
            losses_step = dict(loss_total=loss_total.item(), loss_ce=loss.item())
            if self._erank_regularizer is not None:
                losses_step.update(dict(loss_erank=loss_reg.item()))
            with torch.no_grad():
                metric_vals = self._train_metrics(y_pred, ys)
            additional_logs = self._get_additional_train_step_log()
            # log step
            wandb.log({'train_step/': {'epoch': epoch, 'train_step': self._train_step,
                      **losses_step, **metric_vals, **additional_logs}})
            # save batch losses
            for loss_name in losses_epoch:
                losses_epoch[loss_name].append(losses_step[loss_name])

        # log epoch
        metrics_epoch = self._train_metrics.compute()
        
        self._finish_train_epoch(epoch, losses_epoch, metrics_epoch)


    def _get_additional_train_step_log(self) -> Dict[str, Any]:
        # norm of model parameter vector
        model_param_vec = nn.utils.parameters_to_vector(self._model.parameters())
        model_param_norm = torch.linalg.norm(model_param_vec, ord=2).item()
        log_dict = {'weight_norm': model_param_norm}

        # length of optimizer step
        if self._erank_regularizer is not None:
            model_step_len = self._erank_regularizer.get_param_step_len()
            log_dict.update({'optim_step_len': model_step_len})

            # erank of normalized models
            if self.config.trainer.erank.get('log_normalized_erank', False):
                normalized_erank = self._erank_regularizer.get_normalized_erank()
                log_dict.update({'normalized_erank': normalized_erank})

        return log_dict

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:

        losses_epoch = dict(loss=[]) # add more losses here if necessary

        pbar = tqdm(self._loaders['val'], desc=f'Val epoch {epoch}', file=sys.stdout)
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)

            with torch.no_grad():
                y_pred = trained_model(xs)

                loss = self._loss(y_pred, ys)
                losses_step = dict(loss=loss.item())
                for loss_name in losses_epoch:
                    losses_epoch[loss_name].append(losses_step[loss_name])
                m_val = self._val_metrics(y_pred, ys)

        # compute mean metrics over dataset
        metrics_epoch = self._val_metrics.compute()
        val_score = self._finish_val_epoch(epoch, losses_epoch, metrics_epoch)
        return val_score