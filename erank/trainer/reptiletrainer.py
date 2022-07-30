import logging
import sys
import copy
from typing import Dict, Tuple, List
import torch
from torch import nn
import pandas as pd
from omegaconf import DictConfig
import torchmetrics
from tqdm import tqdm
import wandb
from erank.data import get_metadataset_class
from erank.data.basemetadataset import support_query_as_minibatch
from erank.trainer.erankbasetrainer import ErankBaseTrainer
from ml_utilities.utils import convert_dict_to_python_types
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler


LOGGER = logging.getLogger(__name__)

class ReptileTrainer(ErankBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Reptile Trainer.')
        self._task_batch_size = self.config.trainer.batch_size
        self._inner_optimizer = self.config.trainer.inner_optimizer
        self._n_inner_iter = self.config.trainer.n_inner_iter

        self._inner_train_step = 0

    def _create_datasets(self) -> None:
        LOGGER.info('Loading train/val dataset.')
        data_cfg = self.config.data
        # these are datasets of tasks
        # each task has a support and a query set
        metadataset_class = get_metadataset_class(data_cfg.metadataset)
        train_tasks = metadataset_class(**data_cfg.train_metadataset_kwargs)
        val_tasks = metadataset_class(**data_cfg.val_metadataset_kwargs)
        self._datasets = dict(train=train_tasks, val=val_tasks)

    def _create_dataloaders(self) -> None:
        # does nothing, for compatibility reasons
        self._loaders = {}

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError()]) # TODO make generic and configurable
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        losses_inner_learning, losses_inner_eval = [], []

        # zero grad of model, since the task updates/gradients will be accumulated there
        self._model.zero_grad()

        # parallel version of Reptile (iterate over a batch of tasks)
        task_batch = self._datasets['train'].sample_tasks(self._task_batch_size)
        pbar = tqdm(task_batch, desc=f'Train epoch {epoch}', file=sys.stdout)
        for task in pbar:
            # sample support and query set
            support_set, query_set = task.support_set, task.query_set
            
            # copy model parameters and do inner-loop learning
            inner_model = copy.deepcopy(self._model)
            inner_model, log_losses_inner_learning = self._inner_loop_learning(inner_model, support_set)

            # eval on query set with inner-loop optimized model
            log_losses_inner_eval = self._inner_loop_eval(inner_model, query_set)

            # after inner-loop optimization: accumulate gradients in self._model.grad / meta_model
            # calculate meta-gradient: meta_model - task_model | paramters(self._model) - parameters(inner_model) | g = phi - phi_i
            for meta_model_param, task_model_param in zip(self._model.parameters(), inner_model.parameters()):
                g = meta_model_param - task_model_param
                # average self._model.grad, i.e. divide by number of tasks
                g.div_(self._task_batch_size)
                if meta_model_param.grad is None:
                    meta_model_param.grad = g
                else:
                    meta_model_param.grad.add_(g)
            
            # track all logs
            losses_inner_learning.append(log_losses_inner_learning)
            losses_inner_eval.append(log_losses_inner_eval)

        # outer loop step / update meta-parameters
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._train_step += 1

        # log epoch
        # prefix 'query_': losses_inner_eval 
        losses_inner_eval = pd.DataFrame(losses_inner_eval).mean().add_prefix('query_').add_suffix('_taskmean').to_dict()
        # prefix 'support_': losses_inner_learning
        losses_inner_learning = self.__process_log_inner_learning(losses_inner_learning)
        losses_inner_learning = pd.DataFrame(losses_inner_learning).mean().add_prefix('support_').add_suffix('_taskmean').to_dict()
        losses_epoch = dict(inner_train_step=self._inner_train_step,**losses_inner_eval, **losses_inner_learning)

        self._finish_train_epoch(epoch, losses_epoch)

    def __process_log_inner_learning(self, log_dicts_losses_inner_learning: List[Dict[str, List[float]]]) -> List[Dict[str, float]]:
        # TODO jit
        list_log_dict = []
        for task_log_dict in log_dicts_losses_inner_learning:
            log_dict = {}
            for k, loss_list in task_log_dict.items():
                # extract mean loss accross steps
                log_dict[f'{k}_stepmean'] = torch.tensor(loss_list).mean().item()
                # extract loss last step
                log_dict[f'{k}_steplast'] = loss_list[-1]
            list_log_dict.append(log_dict)
        return list_log_dict

    def _inner_loop_learning(self, inner_model: nn.Module, support_set: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[nn.Module, Dict[str, List[float]]]:
        # log dict
        losses_inner = dict(loss=[])
        
        inner_model.train(True)
        inner_optimizer, _ = create_optimizer_and_scheduler(inner_model.parameters(), **self._inner_optimizer)
        inner_optimizer.zero_grad()
        # do inner-loop optimization
        for i in range(self._n_inner_iter):
            xs, ys = support_query_as_minibatch(support_set, self.device)
            # forward pass
            ys_pred = inner_model(xs)
            loss = self._loss(ys_pred, ys)
            # TODO erank regularization

            # backward pass
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            self._inner_train_step += 1

            # metrics & logging
            losses_inner['loss'].append(loss.item())

        return inner_model, losses_inner

    def _inner_loop_eval(self, inner_model: nn.Module, query_set: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        xq, yq = support_query_as_minibatch(query_set, self.device)
        inner_model.train(False)
        yq_pred = inner_model(xq)
        meta_loss = self._loss(yq_pred, yq)
        # metrics & logging
        losses_inner_eval = dict()
        losses_inner_eval['loss'] = meta_loss.item()
        metric_vals = self._train_metrics(yq_pred, yq)
        # put train metrics into log dict
        losses_inner_eval.update(convert_dict_to_python_types(metric_vals))
        return losses_inner_eval

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:
        # TODO implement val_epoch
        val_score = 0.
        return val_score