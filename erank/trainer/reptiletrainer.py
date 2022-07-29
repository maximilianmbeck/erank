import logging
import sys
import pandas as pd
import torch
import torch.utils.data as data

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
        # each task returns two dataloaders for support and query set
        train_tasks = ...
        val_tasks = ...
        self._datasets = dict(train=train_tasks, val=val_tasks)

    def _create_dataloaders(self) -> None:
        # these dataloaders return a batch of tasks
        # train_loader = data.DataLoader(
        #     dataset=self._datasets['train'],
        #     batch_size=self.config.trainer.batch_size, shuffle=True, drop_last=False,
        #     num_workers=self.config.trainer.num_workers)
        # val_loader = data.DataLoader(dataset=self._datasets['val'], batch_size=self.config.trainer.batch_size,
        #                              shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)
        
        self._loaders = dict(train=train_loader, val=val_loader)

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
