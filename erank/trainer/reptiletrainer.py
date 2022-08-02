import logging
import sys
import copy
from typing import Dict, Tuple, List
import wandb
import torch
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig
from erank.data import get_metadataset_class
from erank.data.basemetadataset import support_query_as_minibatch
from erank.trainer.erankbasetrainer import ErankBaseTrainer
from ml_utilities.utils import convert_dict_to_python_types
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler

LOGGER = logging.getLogger(__name__)

LOG_SEP_SYMBOL = '-'
SAVEDIR_LOSSES_INNER = 'inner_losses/'
SAVEDIR_PRED_PLOT = 'pred_plots/'
SAVEDIR_RESULTS = 'results/'
SAVEFNAME_RESULTS_EVAL_TABLE = 'epoch-{epoch}-valtasks_results.csv'
SAVEFNAME_LOSSES_INNER = 'epoch-{epoch}-losses_inner.png'
DPI = 300


class ReptileTrainer(ErankBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Reptile Trainer.')
        # TODO add doc for parameters
        self._task_batch_size = self.config.trainer.batch_size
        self._inner_optimizer = self.config.trainer.inner_optimizer
        self._n_inner_iter = self.config.trainer.n_inner_iter
        self._val_plots = self.config.trainer.val_plots

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
        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError()])  # TODO make generic and configurable
        self._train_metrics = metrics.clone()  # prefix='train_'
        self._val_metrics = metrics.clone()  # prefix='val_'

    def _train_epoch(self, epoch: int) -> None:
        # setup logging
        losses_inner_learning, losses_inner_eval = {}, {}

        # zero grad of model, since the task updates/gradients will be accumulated there
        self._model.zero_grad()

        # parallel version of Reptile (iterate over a batch of tasks)
        task_batch = self._datasets['train'].sample_tasks(self._task_batch_size)
        # pbar = tqdm(task_batch, desc=f'Train epoch {epoch}', file=sys.stdout) # don't use tqdm for performance reasons
        for task in task_batch:
            # sample support and query set
            support_set, query_set = task.support_set, task.query_set

            # copy model parameters and do inner-loop learning
            inner_model = copy.deepcopy(self._model)
            inner_model, log_losses_inner_learning = self._inner_loop_learning(inner_model, support_set)

            # eval on query set with inner-loop optimized model
            log_losses_inner_eval, query_preds = self._inner_loop_eval(inner_model, query_set)

            #! meta-model gradient update
            # after inner-loop optimization: accumulate gradients in self._model.grad / meta_model
            # calculate meta-gradient: meta_model - task_model | paramters(self._model) - parameters(inner_model) | g = phi - phi_i
            for meta_model_param, task_model_param in zip(self._model.parameters(), inner_model.parameters()):
                g = meta_model_param - task_model_param  # partial meta-gradient
                # average self._model.grad, i.e. divide by number of tasks
                g.div_(self._task_batch_size)
                if meta_model_param.grad is None:
                    meta_model_param.grad = g
                else:
                    meta_model_param.grad.add_(g)

            # track all logs
            losses_inner_learning[task.name] = log_losses_inner_learning
            losses_inner_eval[task.name] = log_losses_inner_eval

        #! meta-model gradient step
        # outer loop step / update meta-parameters
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._train_step += 1

        # log epoch
        self.__log_train_epoch(epoch, losses_inner_learning, losses_inner_eval)

    def _inner_loop_learning(
            self, inner_model: nn.Module, support_set: Tuple[torch.Tensor,
                                                             torch.Tensor]) -> Tuple[nn.Module, Dict[str, List[float]]]:
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

    def _inner_loop_eval(self, inner_model: nn.Module,
                         query_set: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], torch.Tensor]:
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
        return losses_inner_eval, yq_pred.detach()

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:
        # setup logging
        losses_inner_learning, losses_inner_eval_before, losses_inner_eval_after = {}, {}, {}
        preds_plot_log = {}

        # get eval tasks
        eval_tasks = self._datasets['val'].get_tasks()
        pbar = tqdm(eval_tasks, desc=f'Val epoch {epoch}', file=sys.stdout)
        for i, task in enumerate(pbar):
            # sample support and query set
            support_set, query_set = task.support_set, task.query_set

            # copy model parameters and do inner-loop learning
            inner_model = copy.deepcopy(trained_model)
            # eval on query set before inner-loop optimization
            log_losses_inner_eval_before, predictions_before = self._inner_loop_eval(inner_model, query_set)

            # fine-tune model for `n_inner_iter steps`
            inner_model, log_losses_inner_learning = self._inner_loop_learning(inner_model, support_set)

            # eval on query set with inner-loop optimized model
            log_losses_inner_eval_after, predictions_after = self._inner_loop_eval(inner_model, query_set)

            # track all logs
            losses_inner_eval_before[task.name] = log_losses_inner_eval_before
            losses_inner_learning[task.name] = log_losses_inner_learning
            losses_inner_eval_after[task.name] = log_losses_inner_eval_after

            # make plot of predictions
            if i < self._val_plots:
                fig, fname = task.plot_query_predictions(epoch, predictions_before, predictions_after)
                # save fig & log to wandb
                save_path = self._experiment_dir / SAVEDIR_PRED_PLOT
                save_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path / fname, bbox_inches='tight', dpi=DPI)
                preds_plot_log[task.name] = fig
                # plt.close(fig)

        #! log val epoch
        return self.__log_val_epoch(epoch, losses_inner_learning, losses_inner_eval_before, losses_inner_eval_after,
                                    preds_plot_log)

    #### 

    def __log_train_epoch(self, epoch: int, losses_inner_learning: Dict[str, Dict[str, List[float]]],
                          losses_inner_eval: Dict[str, Dict[str, float]]) -> None:
        """Log results of a training iteration (epoch):

        Args:
            epoch (int): epoch
            losses_inner_learning (Dict[str, Dict[str, List[float]]]): Dict levels: Task->metric/loss->Values at inner steps
            losses_inner_eval (Dict[str, Dict[str, float]]): Dict levels: Task->metric/loss->Value
        """
        # prefix 'query{LOG_SEP_SYMBOL}': losses_inner_eval
        losses_inner_eval = pd.DataFrame(losses_inner_eval).transpose().mean().add_prefix(
            f'query{LOG_SEP_SYMBOL}').add_suffix(f'{LOG_SEP_SYMBOL}taskmean').to_dict()
        # prefix 'support{LOG_SEP_SYMBOL}': losses_inner_learning
        losses_inner_learning = self.__process_log_inner_learning(losses_inner_learning)
        losses_inner_learning = pd.DataFrame(losses_inner_learning).mean().add_prefix(
            f'support{LOG_SEP_SYMBOL}').add_suffix(f'{LOG_SEP_SYMBOL}taskmean').to_dict()
        losses_epoch = dict(inner_train_step=self._inner_train_step, **losses_inner_eval, **losses_inner_learning)
        self._finish_train_epoch(epoch, losses_epoch)

    def __process_log_inner_learning(
            self, log_dicts_losses_inner_learning: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, float]]:
        """We get a log_dict for every task (outer Dict[task.name, log_dict]). Each log_dict contains all losses and metrics as keys and a list of values corresponding to each 
        inner update step. 
        This method extracts meaningful information from these data that can be logged to a wandb panel.
        For now, for example: the metrics mean accross all inner step and the metrics after the last inner step."""
        list_log_dict = []
        for task_name, task_log_dict in log_dicts_losses_inner_learning.items():
            log_dict = {}
            for k, loss_list in task_log_dict.items():
                # extract mean loss accross steps
                log_dict[f'{k}_stepmean'] = torch.tensor(loss_list).mean().item()
                # extract loss last step
                log_dict[f'{k}_steplast'] = loss_list[-1]
            list_log_dict.append(log_dict)
        return list_log_dict

    def __log_val_epoch(self, epoch: int, losses_inner_learning: Dict[str, Dict[str, List[float]]],
                        losses_inner_eval_before: Dict[str, Dict[str, float]],
                        losses_inner_eval_after: Dict[str, Dict[str, float]], preds_plot_log: Dict[str,
                                                                                                   Figure]) -> float:
        """Log results of a validation iteration (epoch).

        Args:
            epoch (int): epoch
            losses_inner_learning (Dict[str, Dict[str, List[float]]]): Dict levels: Task->metric/loss->Values at inner steps
            losses_inner_eval_before (Dict[str, Dict[str, float]]): Dict levels: Task->metric/loss->Value
            losses_inner_eval_after (Dict[str, Dict[str, float]]): Dict levels: Task->metric/loss->Value
            preds_plot_log (Dict[str, Figure]): Dict levels: Task->Plot

        Returns:
            float: The validation score used for early stopping.
        """
        # EVAL: prefix 'query{LOG_SEP_SYMBOL}': losses_inner_eval
        losses_eval_before_df = pd.DataFrame(losses_inner_eval_before).transpose().add_suffix(f'{LOG_SEP_SYMBOL}before')
        losses_eval_after_df = pd.DataFrame(losses_inner_eval_after).transpose().add_suffix(f'{LOG_SEP_SYMBOL}after')
        # this is a table containing all tasks as rows and the metrics as columns
        losses_eval_df = pd.concat([losses_eval_after_df, losses_eval_before_df],
                                   axis=1).add_prefix(f'query{LOG_SEP_SYMBOL}')
        save_path = self._experiment_dir / SAVEDIR_RESULTS
        save_path.mkdir(parents=True, exist_ok=True)
        losses_eval_df.to_csv(save_path / SAVEFNAME_RESULTS_EVAL_TABLE.format(epoch=epoch))
        # extract global metrics accross all tasks
        losses_eval_taskmean_df = losses_eval_df.mean().add_suffix(f'{LOG_SEP_SYMBOL}taskmean')
        losses_eval_taskmedian_df = losses_eval_df.median().add_suffix(f'{LOG_SEP_SYMBOL}taskmedian')
        log_dict_losses_eval: Dict[str, float] = pd.concat([losses_eval_taskmean_df, losses_eval_taskmedian_df],
                                                           axis=0).to_dict()

        # LEARNING: prefix 'support{LOG_SEP_SYMBOL}': losses_inner_learning
        processed_losses_inner_learning = self.__process_log_inner_learning(losses_inner_learning)
        log_dict_losses_inner_learning: Dict[str, float] = pd.DataFrame(processed_losses_inner_learning).mean(
        ).add_prefix(f'support{LOG_SEP_SYMBOL}').add_suffix(f'{LOG_SEP_SYMBOL}taskmean').to_dict()

        # make plot of losses_inner_learning for each task / a subset of each task
        losses_inner_plot_log = {}
        fig, fname = self.__plot_inner_learning_curves(epoch, losses_inner_learning)
        save_path = self._experiment_dir / SAVEDIR_LOSSES_INNER
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / fname, bbox_inches='tight', dpi=DPI)
        losses_inner_plot_log['inner-losses'] = fig
        # plt.close(fig)

        # log to wandb
        losses_epoch = dict(inner_train_step=self._inner_train_step,
                            **log_dict_losses_eval,
                            **log_dict_losses_inner_learning)
        self._log_losses_metrics(prefix='val', epoch=epoch, losses_epoch=losses_epoch)
        self._log_losses_metrics(prefix='preds', epoch=epoch, metrics_epoch=preds_plot_log, log_to_console=False)
        self._log_losses_metrics(prefix='val-inner',
                                 epoch=epoch,
                                 metrics_epoch=losses_inner_plot_log,
                                 log_to_console=False)
        val_score = losses_eval_after_df.mean()['loss-after']  # TODO make configurable
        self._reset_metrics()
        return val_score

    def __plot_inner_learning_curves(self, epoch: int,
                                    losses_inner_learning: Dict[str, Dict[str, List[float]]]) -> Tuple[Figure, str]:
        fig, ax = plt.subplots(1, 1)
        # TODO adapt if log_dict contains multiple log losses
        for task_name, log_dict in losses_inner_learning.items():
            ax.plot(log_dict['loss'], label=task_name)
        ax.set_ylabel('inner-loss')
        ax.set_xlabel('inner-steps')
        ax.set_title(f'Epoch {epoch}: Loss curves inner-loop')
        # ax.legend() # wandb displays label, when hovering
        return fig, SAVEFNAME_LOSSES_INNER.format(epoch=epoch)