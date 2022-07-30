import logging
import sys
import copy
import torch
from torch import nn
import pandas as pd

from omegaconf import DictConfig
from tqdm import tqdm
import wandb
from erank.data import get_metadataset_class
from erank.trainer.erankbasetrainer import ErankBaseTrainer
from ml_utilities.utils import convert_dict_to_python_types, zip_strict
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler


LOGGER = logging.getLogger(__name__)

class ReptileTrainer(ErankBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Reptile Trainer.')
        self._task_batch_size = self.config.trainer.batch_size
        self._inner_optimizer = self.config.trainer.inner_optimizer
        self._n_inner_iter = self.config.trainer.n_inner_iter

    def _create_datasets(self) -> None:
        LOGGER.info('Loading train/val dataset.')
        data_cfg = self.config.data
        # these are datasets of tasks
        # each task has a support and a query set
        metadataset_class = get_metadataset_class(data_cfg.metadataset)
        train_tasks = metadataset_class(**data_cfg.train_metadataset_kwargs)
        val_tasks = metadataset_class(**data_cfg.val_metadataset_kwargs)
        self._datasets = dict(train=train_tasks, val=val_tasks)

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        metrics = ... # TODO add metrics
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        loss_vals = dict(loss_total=[], loss_ce=[])
        if self._erank_regularizer is not None:
            loss_vals.update(dict(loss_erank=[]))

        # zero grad of model, since the task updates/gradients will be accumulated there
        self._model.zero_grad()

        # parallel version of Reptile (iterate over a batch of tasks)
        task_batch = self._datasets['train'].sample_tasks(self._task_batch_size)
        pbar = tqdm(task_batch, desc=f'Train epoch {epoch}', file=sys.stdout)
        for task in pbar:
            # sample support and query set
            support_set, query_set = task.support_set, task.query_set
            
            # copy model parameters and init inner optimizer
            inner_model = copy.deepcopy(self._model) # TODO check if model still on same device
            inner_model.train(True)
            inner_optimizer, _ = create_optimizer_and_scheduler(inner_model.parameters(), **self._inner_optimizer)
            inner_optimizer.zero_grad()
            # do inner-loop optimization
            for i in range(self._n_inner_iter):
                xs, ys = support_set[0].to(self.device), support_set[1].to(self.device)
                # forward pass
                ys_pred = inner_model(xs)
                loss = self._loss(ys_pred, ys)
                # TODO erank regularization
                # backward pass
                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

            # after inner-loop optimization: accumulate gradients in self._model.grad / meta_model
            # calculate meta-gradient: meta_model - task_model | paramters(self._model) - parameters(inner_model) | g = phi - phi_i
            for meta_model_param, task_model_param in zip(self._model.parameters(), inner_model.parameters()):
                g = meta_model_param - task_model_param
                if meta_model_param.grad is None:
                    meta_model_param.grad = g
                else:
                    meta_model_param.grad.add_(g)

            # eval on query set with inner-loop optimized model
            xq, yq = query_set[0].to(self.device), query_set[1].to(self.device)
            inner_model.train(False)
            yq_pred = inner_model(xq)
            meta_loss = self._loss(yq_pred, yq)

        
        # average self._model.grad, i.e. divide by number of tasks
        for param in self._model.parameters():
            param.grad.div_(self._task_batch_size)

        # outer loop step / update meta-parameters
        # TODO gradient clipping here
        self._optimizer.step()
        self._optimizer.zero_grad()        
        self._train_step += 1

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
        # TODO implement val_epoch
        # val_losses = []

        # pbar = tqdm(self._loaders['val'], desc=f'Val epoch {epoch}', file=sys.stdout)
        # for xs, ys in pbar:
        #     xs, ys = xs.to(self.device), ys.to(self.device)

        #     with torch.no_grad():
        #         y_pred = trained_model(xs)

        #         loss = self._loss(y_pred, ys)
        #         val_losses.append(loss.item())
        #         m_val = self._val_metrics(y_pred, ys)

        # # compute mean metrics over dataset
        # metric_vals = self._val_metrics.compute()

        # # log epoch
        # log_dict = {'epoch': epoch, 'loss': torch.tensor(val_losses).mean().item(), **metric_vals}
        # wandb.log({'val/': log_dict})

        # LOGGER.info(f'Val epoch \n{pd.Series(convert_dict_to_python_types(log_dict), dtype=float)}')

        # # val_score is first metric in self._val_metrics
        # val_score = metric_vals[next(iter(self._val_metrics.items()))[0]].item()

        # self._reset_metrics()
        val_score = 0
        return val_score