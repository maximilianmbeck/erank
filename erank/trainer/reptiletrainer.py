import logging
import sys
import pandas as pd
import torch
from torch import nn

from omegaconf import DictConfig
from tqdm import tqdm
import wandb
from erank.trainer.erankbasetrainer import ErankBaseTrainer
from ml_utilities.utils import convert_dict_to_python_types

LOGGER = logging.getLogger(__name__)

class ReptileTrainer(ErankBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Reptile Trainer.')

    def _create_datasets(self) -> None:
        LOGGER.info('Loading train/val dataset.')
        data_cfg = self.config.data
        # these are datasets of tasks
        # each task has a support and a query set
        train_tasks = ...
        val_tasks = ...
        self._datasets = dict(train=train_tasks, val=val_tasks)

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        metrics = ...
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        loss_vals = dict(loss_total=[], loss_ce=[])
        if self._erank_regularizer is not None:
            loss_vals.update(dict(loss_erank=[]))

        # training loop (iterate over a batch of tasks)
        pbar = tqdm(self._loaders['train'], desc=f'Train epoch {epoch}', file=sys.stdout)
        for task in pbar:
            # TODO reptile algorithm with erank regularization
            
            pass
        
        # log epoch
        metric_vals = self._train_metrics.compute()

        for loss_name, loss_val_list in loss_vals.items():
            loss_vals[loss_name] = torch.tensor(loss_val_list).mean().item()

        log_dict = {'epoch': epoch, 'train_step': self._train_step,
                    **loss_vals, **metric_vals}
        wandb.log({'train_epoch/': log_dict})

        LOGGER.info(f'Train epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

        self._reset_metrics()

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:

        val_losses = []

        pbar = tqdm(self._loaders['val'], desc=f'Val epoch {epoch}', file=sys.stdout)
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)

            with torch.no_grad():
                y_pred = trained_model(xs)

                loss = self._loss(y_pred, ys)
                val_losses.append(loss.item())
                m_val = self._val_metrics(y_pred, ys)

        # compute mean metrics over dataset
        metric_vals = self._val_metrics.compute()

        # log epoch
        log_dict = {'epoch': epoch, 'loss': torch.tensor(val_losses).mean().item(), **metric_vals}
        wandb.log({'val/': log_dict})

        LOGGER.info(f'Val epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

        # val_score is first metric in self._val_metrics
        val_score = metric_vals[next(iter(self._val_metrics.items()))[0]].item()

        self._reset_metrics()
        return val_score