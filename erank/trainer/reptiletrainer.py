import logging
import sys
import copy
from typing import Dict, Tuple, List, Deque
import torch
import pandas as pd
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
from erank.data import get_metadataset_class
from erank.data.basemetadataset import support_query_as_minibatch
from erank.regularization import LOG_LOSS_TOTAL_KEY
from erank.trainer.erankbasetrainer import ErankBaseTrainer
from ml_utilities.utils import convert_dict_to_python_types, convert_listofdicts_to_dictoflists
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler
from ml_utilities.torch_utils import compute_grad_norm, compute_weight_norm

LOGGER = logging.getLogger(__name__)

LOG_SEP_SYMBOL = '-'
SAVEDIR_LOSSES_INNER = 'inner_losses/'
SAVEDIR_PRED_PLOT = 'pred_plots/'
SAVEDIR_RESULTS = 'results/'
SAVEFNAME_RESULTS_EVAL_TABLE = 'epoch-{epoch}-valtasks_results.csv'
SAVEFNAME_LOSSES_INNER = 'epoch-{epoch}-losses_inner.png'
SAVEFNAME_LOSSES_INNER_AVG = 'epoch-{epoch}-task_avg_losses_inner.png'
DPI = 150


class ReptileTrainer(ErankBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Reptile Trainer.')
        # TODO add doc for parameters
        self._task_batch_size = self.config.trainer.task_batch_size
        self._inner_optimizer = self.config.trainer.inner_optimizer
        self._n_inner_iter = self.config.trainer.n_inner_iter
        self._val_pred_plots_for_tasks = self.config.trainer.val_pred_plots_for_tasks
        self._log_plot_inner_learning_curves = self.config.trainer.get('log_plot_inner_learning_curves', False)
        self._verbose = self.config.trainer.get('verbose', False)

        self._inner_eval_after_steps = self.config.trainer.get('inner_eval_after_steps', None)
        if self._inner_eval_after_steps is None:
            # use a default list
            self._inner_eval_after_steps = [0, 1, 2, 3, 5, 10, 20, 30, 50]
        else:
            if not isinstance(self._inner_eval_after_steps, (list, ListConfig)):
                assert isinstance(self._inner_eval_after_steps, (float, int))
                self._inner_eval_after_steps = [self._inner_eval_after_steps]
            # make sure to always evaluate the meta/base model before finetuning
            if not 0 in self._inner_eval_after_steps:
                self._inner_eval_after_steps.append(0)
            # make sure to always evaluate the finetuned model after the finetuning
            if not self._n_inner_iter in self._inner_eval_after_steps:
                self._inner_eval_after_steps.append(self._n_inner_iter)

        self._inner_train_step = 0
        self.__inner_learning_curves_ylim_upper = None

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

    def _train_epoch(self, epoch: int) -> None:
        LOGGER.debug(f'--Train epoch: {epoch}')
        # setup logging
        losses_inner_learning, losses_inner_eval, epoch_stats = {}, {}, {}

        # zero grad of model, since the task updates/gradients will be accumulated there
        self._model.zero_grad()

        # parallel version of Reptile (iterate over a batch of tasks)
        task_batch = self._datasets['train'].sample_tasks(self._task_batch_size)
        # pbar = tqdm(task_batch, desc=f'Train epoch {epoch}', file=sys.stdout) # don't use tqdm for performance reasons
        for task_idx, task in enumerate(task_batch):
            LOGGER.debug(f'----Task idx: {task_idx}')
            # sample support and query set
            support_set, query_set = task.support_set, task.query_set

            # copy model parameters and do inner-loop learning
            inner_model = copy.deepcopy(self._model)
            inner_model, log_losses_inner_learning, _, _ = self._inner_loop_learning(
                'train', inner_model, support_set, eval_after_steps=[])  # no evaluation during inner-loop training

            # eval on query set with inner-loop optimized model
            log_losses_inner_eval, query_preds = self._inner_loop_eval(inner_model, query_set)
            LOGGER.debug(
                f'Train inner eval losses: \n{pd.Series(convert_dict_to_python_types(log_losses_inner_eval), dtype=float)}'
            )

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

            #? ERANK: add to buffer
            if self._erank_regularizer:
                subspace_vec_norm = self._erank_regularizer.add_subspace_vec(inner_model)
                log_losses_inner_eval.update({'param_subspace_vec_norm': subspace_vec_norm})

            # track all logs
            losses_inner_learning[task.name] = log_losses_inner_learning
            losses_inner_eval[task.name] = log_losses_inner_eval

        #! meta-model gradient step
        epoch_stats['meta-grad-norm'] = compute_grad_norm(self._model)
        # outer loop step / update meta-parameters
        self._optimizer.step()
        self._optimizer.zero_grad()
        self._train_step += 1

        #? ERANK: sets a reference to base model
        if self._erank_regularizer:
            self._erank_regularizer.set_base_model(self._model)

        # log epoch
        self.__log_train_epoch(epoch, losses_inner_learning, losses_inner_eval, epoch_stats)

    def _inner_loop_learning(
        self,
        mode: str,
        inner_model: nn.Module,
        support_set: Tuple[torch.Tensor, torch.Tensor],
        query_set: Tuple[torch.Tensor, torch.Tensor] = None,
        eval_after_steps: List[int] = [0, 1, 2, 3, 5, 10, 20, 30, 50]
    ) -> Tuple[nn.Module, List[Dict[str, torch.Tensor]], Dict[int, Dict[str, float]], Dict[int, torch.Tensor]]:
        """Inner learning loop. Training on `support_set` and evaluation on `query_set`. 
        Evaluation is performed after every step in `eval_after_steps`.

        Args:
            mode (str): `train` or `val`. Might have effect on the loss used for example.
            inner_model (nn.Module): The base/meta model to finetune.
            support_set (Tuple[torch.Tensor, torch.Tensor]): The training set.
            query_set (Tuple[torch.Tensor, torch.Tensor]): The validation/test set.
            eval_after_steps (List[int]): Does evaluation after steps in this list.

        Returns:
            Tuple[nn.Module, List[Dict[str, torch.Tensor]], Dict[int, Dict[str, float]], Dict[int, torch.Tensor]]: 
                - finetuned model
                - inner training losses
                - inner evaluation losses and metrics
                - inner evaluation predictions
        """

        def do_eval_after_step(step: int):
            if step in eval_after_steps:
                loss_metric_dict, preds = self._inner_loop_eval(inner_model, query_set)
                losses_inner_eval[step] = loss_metric_dict
                inner_eval_preds[step] = preds
                LOGGER.debug(
                    f'Inner eval losses: \n{pd.Series(convert_dict_to_python_types(loss_metric_dict), dtype=float)}')

        # log dicts
        losses_inner: List[Dict[str, torch.Tensor]] = []
        losses_inner_eval, inner_eval_preds = {}, {}

        inner_model.train(True)
        inner_optimizer, _ = create_optimizer_and_scheduler(inner_model.parameters(), **self._inner_optimizer)
        inner_optimizer.zero_grad()
        # eval before fine-tuning
        i = 0
        do_eval_after_step(i)

        # do inner-loop optimization
        for i in range(self._n_inner_iter):
            LOGGER.debug(f'------Inner iter: {i}')
            xs, ys = support_query_as_minibatch(support_set, self.device)
            # forward pass
            ys_pred = inner_model(xs)
            if mode == 'train':
                loss, loss_dict = self._loss(ys_pred, ys, inner_model)  # use regularization
                self._inner_train_step += 1
            elif mode == 'val':
                loss, loss_dict = self._loss(ys_pred, ys)  # no regularization
            else:
                raise ValueError(f'Unsupported inner-loop learning mode: `{mode}`')

            if torch.isnan(loss):
                raise RuntimeError(
                    f'Loss NaN in inner iteration {i} of epoch {self._epoch}. Single Loss Terms: \n{pd.Series(convert_dict_to_python_types(loss_dict), dtype=float)}'
                )

            loss_dict['weight_norm'] = compute_weight_norm(inner_model)

            # backward pass
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

            loss_dict['grad_norm'] = compute_grad_norm(inner_model)
            if self._verbose:
                LOGGER.debug(f'Inner losses, weights, grads: \n{pd.Series(convert_dict_to_python_types(loss_dict), dtype=float)}')
            
            # eval during fine-tuning
            do_eval_after_step(i)

            # metrics & logging
            losses_inner.append(loss_dict)

        # eval after fine-tuning
        i += 1
        do_eval_after_step(i)

        return inner_model, losses_inner, losses_inner_eval, inner_eval_preds

    def _inner_loop_eval(self, inner_model: nn.Module,
                         query_set: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], torch.Tensor]:
        xq, yq = support_query_as_minibatch(query_set, self.device)
        inner_model.train(False)
        with torch.no_grad():
            yq_pred = inner_model(xq)
            meta_loss, loss_dict = self._loss(yq_pred, yq)
            metric_vals = self._train_metrics(yq_pred, yq)
        # metrics & logging
        losses_inner_eval = dict()
        losses_inner_eval.update(convert_dict_to_python_types(loss_dict))
        # put train metrics into log dict
        losses_inner_eval.update(convert_dict_to_python_types(metric_vals))
        return losses_inner_eval, yq_pred.detach().cpu()

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:
        LOGGER.debug(f'--Val epoch: {epoch}')
        # setup logging
        losses_inner_learning, losses_inner_eval = {}, {}
        preds_plot_log = {}

        # get eval tasks
        eval_tasks = self._datasets['val'].get_tasks()
        # pbar = tqdm(eval_tasks, desc=f'Val epoch {epoch}', file=sys.stdout)
        for task_idx, task in enumerate(eval_tasks):
            LOGGER.debug(f'----Task idx: {task_idx}')
            # sample support and query set
            support_set, query_set = task.support_set, task.query_set

            # copy model parameters and do inner-loop learning
            inner_model = copy.deepcopy(trained_model)

            # fine-tune model for `n_inner_iter steps` and evaluate during finetuning after some gradient steps
            inner_model, log_losses_inner_learning, log_losses_inner_eval, eval_predictions = self._inner_loop_learning(
                'val', inner_model, support_set, query_set, eval_after_steps=self._inner_eval_after_steps)

            # track all logs
            losses_inner_eval[task.name] = log_losses_inner_eval
            losses_inner_learning[task.name] = log_losses_inner_learning

            # make plot of predictions
            if task_idx < self._val_pred_plots_for_tasks:
                fig, fname = task.plot_query_predictions(epoch, eval_predictions)
                # save fig & log to wandb
                save_path = self._experiment_dir / SAVEDIR_PRED_PLOT
                save_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path / fname, bbox_inches='tight', dpi=DPI)
                preds_plot_log[task.name] = fig
                # plt.close(fig)

        #! log val epoch
        val_score = self.__log_val_epoch(epoch, losses_inner_learning, losses_inner_eval, preds_plot_log)
        return val_score

    ###### LOGGING

    def __log_train_epoch(self, epoch: int, losses_inner_learning: Dict[str, List[Dict[str, torch.Tensor]]],
                          losses_inner_eval: Dict[str, Dict[str, float]], epoch_stats: Dict[str, float] = {}) -> None:
        """Log results of a training iteration (epoch).

        Args:
            epoch (int): epoch
            losses_inner_learning (Dict[str, Dict[str, List[float]]]): Dict levels: Task->metric/loss->Values at inner steps
            losses_inner_eval (Dict[str, Dict[str, float]]): Dict levels: Task->metric/loss->Value
            epoch_stats (Dict[str, float]): Any statistics on epoch level. Dict levels: log_name->Value
        """
        # prefix 'query{LOG_SEP_SYMBOL}': losses_inner_eval
        losses_inner_eval = pd.DataFrame(losses_inner_eval).transpose().mean().add_prefix(
            f'query{LOG_SEP_SYMBOL}').add_suffix(f'{LOG_SEP_SYMBOL}taskmean').to_dict()
        # prefix 'support{LOG_SEP_SYMBOL}': losses_inner_learning
        losses_inner_learning = self.__process_log_inner_learning(losses_inner_learning)
        losses_inner_learning = pd.DataFrame(losses_inner_learning).transpose().mean().add_prefix(
            f'support{LOG_SEP_SYMBOL}').add_suffix(f'{LOG_SEP_SYMBOL}taskmean').to_dict()
        losses_epoch = dict(inner_train_step=self._inner_train_step, **losses_inner_eval, **losses_inner_learning, **epoch_stats)
        self._finish_train_epoch(epoch, losses_epoch)

    def __process_log_inner_learning(
            self,
            log_dicts_losses_inner_learning: Dict[str, List[Dict[str, torch.Tensor]]],
            exclude_loss_keys: List[str] = [LOG_LOSS_TOTAL_KEY]) -> Dict[str, Dict[str, float]]:
        """We get a list of log_dicts for every task (outer Dict[task.name, List[log_dict]]). The list contains the log_dict
        for each update step.
        Each log_dict contains all losses and metrics as keys and a list of values corresponding to a single 
        inner update step. 
        This method extracts meaningful information from these data that can be logged to a wandb panel.
        For now, for example: the metrics mean accross all inner step and the metrics after the last inner step."""
        task_summary_log_dicts = {}
        for task_name, task_log_dicts in log_dicts_losses_inner_learning.items():
            log_dict = {}
            task_log_dict = convert_listofdicts_to_dictoflists(task_log_dicts, convert_vals_to_python_types=True)
            task_log_df = pd.DataFrame(task_log_dict).drop(columns=exclude_loss_keys)
            # extract loss of the first step
            log_dict.update(task_log_df.iloc[0].add_suffix(f'{LOG_SEP_SYMBOL}stepfirst').to_dict())
            # extract mean loss across steps
            log_dict.update(task_log_df.mean().add_suffix(f'{LOG_SEP_SYMBOL}stepmean').to_dict())
            # extract loss of the last step
            log_dict.update(task_log_df.iloc[-1].add_suffix(f'{LOG_SEP_SYMBOL}steplast').to_dict())
            task_summary_log_dicts[task_name] = log_dict
        return task_summary_log_dicts

    def __log_val_epoch(self, epoch: int, losses_inner_learning: Dict[str, List[Dict[str, torch.Tensor]]],
                        losses_inner_eval: Dict[int, Dict[str, Dict[str, float]]],
                        preds_plot_log: Dict[str, Figure]) -> float:
        """Log results of a validation iteration (epoch).

        Args:
            epoch (int): epoch
            losses_inner_learning (Dict[str, List[Dict[str, torch.Tensor]]]): Dict levels: Task->metric/loss->Values at inner steps
            losses_inner_eval (Dict[int, Dict[str, Dict[str, float]]]): Dict levels: inner_steps->Task->metric/loss->Value
            preds_plot_log (Dict[str, Figure]): Dict levels: Task->Plot

        Returns:
            float: The validation score used for early stopping.
        """
        # extract eval results before and after fine-tuning for each task
        losses_inner_eval_before = dict()
        losses_inner_eval_after = dict()
        for task_name, task_log in losses_inner_eval.items():
            losses_inner_eval_before[task_name] = task_log[0]  # after 0 steps
            losses_inner_eval_after[task_name] = task_log[self._n_inner_iter]

        #! EVAL: prefix 'query{LOG_SEP_SYMBOL}': losses_inner_eval
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

        #! LEARNING: prefix 'support{LOG_SEP_SYMBOL}': losses_inner_learning
        processed_losses_inner_learning = self.__process_log_inner_learning(losses_inner_learning)
        log_dict_losses_inner_learning: Dict[str, Dict[str, float]] = pd.DataFrame(
            processed_losses_inner_learning).transpose().mean().add_prefix(f'support{LOG_SEP_SYMBOL}').add_suffix(
                f'{LOG_SEP_SYMBOL}taskmean').to_dict()

        # make plot of losses_inner_learning for each task / a subset of each task
        losses_inner_plot_log = {}
        if self._log_plot_inner_learning_curves:
            losses_inner_plot_log = self.__plot_inner_learning_curves(epoch, losses_inner_learning)

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
        val_score_metric_name = list(self._val_metrics.keys())[0]  # the first metric in val_metrics
        val_score = losses_eval_after_df.mean()[f'{val_score_metric_name}{LOG_SEP_SYMBOL}after']
        self._reset_metrics()
        return val_score

    def __plot_inner_learning_curves(self,
                                     epoch: int,
                                     log_dicts_losses_inner_learning: Dict[str, List[Dict[str, torch.Tensor]]],
                                     loss_key_to_plot: str = LOG_LOSS_TOTAL_KEY) -> Dict[str, Figure]:
        """Create plots of inner loss curves and return the plots in a dictionary with their description/name as key."""
        # save path for plots
        save_path = self._experiment_dir / SAVEDIR_LOSSES_INNER
        save_path.mkdir(parents=True, exist_ok=True)
        # labels for plots
        y_label = f'inner-{loss_key_to_plot}'
        x_label = 'inner-steps'

        inner_loss_plots: Dict[str, Figure] = {}
        inner_loss_values: Dict[str, List[float]] = {}

        ## Single task losses
        fig0, ax0 = plt.subplots(1, 1)
        for task_name, task_log_dicts in log_dicts_losses_inner_learning.items():
            task_log_dict = convert_listofdicts_to_dictoflists(task_log_dicts, convert_vals_to_python_types=True)
            inner_loss_values[task_name] = task_log_dict[loss_key_to_plot]
            ax0.plot(task_log_dict[loss_key_to_plot], label=task_name)

        #
        loss_taskmean = pd.DataFrame(inner_loss_values).mean(axis=1).to_numpy()
        loss_taskstd = pd.DataFrame(inner_loss_values).std(axis=1).to_numpy()
        if self.__inner_learning_curves_ylim_upper is None:
            # ylim is very first loss + 2*std and assume loss is decreasing
            self.__inner_learning_curves_ylim_upper = loss_taskmean[0] + 2 * loss_taskstd[0]
        #
        ax0.set_ylim(bottom=0., top=self.__inner_learning_curves_ylim_upper)
        ax0.set_ylabel(y_label)
        ax0.set_xlabel(x_label)
        ax0.set_title(f'Epoch {epoch}: {loss_key_to_plot} curves inner-loop, all tasks')
        # ax.legend() # wandb displays label, when hovering

        fname = SAVEFNAME_LOSSES_INNER.format(epoch=epoch)
        fig0.savefig(save_path / fname, bbox_inches='tight', dpi=DPI)
        inner_loss_plots[f'inner-{loss_key_to_plot}'] = fig0

        ## Mean task loss
        fig1, ax1 = plt.subplots(1, 1)
        ax1.plot(loss_taskmean, color='black', label='task avg')
        # this raises an error with wandb
        # ax1.fill_between(x=np.arange(len(loss_taskmean)),
        #                 y1=loss_taskmean + loss_taskstd,
        #                 y2=loss_taskmean - loss_taskstd, alpha=0.4)
        # workaround:
        ax1.plot(loss_taskmean + loss_taskstd, color='blue', label='+std')
        ax1.plot(loss_taskmean - loss_taskstd, color='blue', label='-std')
        ax1.set_ylabel(y_label)
        ax1.set_xlabel(x_label)
        ax1.set_title(
            f'Epoch {epoch}: {loss_key_to_plot} inner-loop, avg and std across {len(inner_loss_values)} tasks')
        ax1.set_ylim(bottom=0., top=self.__inner_learning_curves_ylim_upper)

        fname = SAVEFNAME_LOSSES_INNER_AVG.format(epoch=epoch)
        fig1.savefig(save_path / fname, bbox_inches='tight', dpi=DPI)
        inner_loss_plots[f'inner-{loss_key_to_plot}-taskavg'] = fig1

        return inner_loss_plots
