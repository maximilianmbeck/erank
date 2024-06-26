import logging
import sys
from typing import Any, Dict, List
import torch
from tqdm import tqdm
from torch import nn
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from erank.data.basemetadataset import support_query_as_minibatch
from ml_utilities.data.datasetgenerator import DatasetGenerator
from erank.data.supervised_metadataset_wrapper import SupervisedMetaDatasetWrapper
from erank.trainer.subspacebasetrainer import SubspaceBaseTrainer

LOGGER = logging.getLogger(__name__)

SAVEDIR_PRED_PLOT = 'pred_plots/'
DPI = 75


class SupervisedTrainer(SubspaceBaseTrainer):
    """Class for training in a supervised setting.

    Args:
        config (DictConfig): Configuration.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        LOGGER.info('Using Supervised Trainer.')

        self._plot_predictions_every_val_multiplier = self.config.trainer.get('plot_predictions_every_val_multiplier',
                                                                              0)

    def _create_datasets(self) -> None:
        LOGGER.info('Loading train/val dataset.')
        data_cfg = self.config.data
        supervised_metads_kwargs = data_cfg.get('supervised_metadataset_wrapper_kwargs', None)
        if supervised_metads_kwargs:
            self._dataset_generator = SupervisedMetaDatasetWrapper(**supervised_metads_kwargs)
        else:
            self._dataset_generator = DatasetGenerator(dataset=data_cfg.dataset,
                                                       dataset_kwargs=data_cfg.dataset_kwargs,
                                                       dataset_split=data_cfg.dataset_split,
                                                       train_split_transforms=data_cfg.get('train_split_transforms', {}), 
                                                       val_split_transforms=data_cfg.get('val_split_transforms', {}))
        self._dataset_generator.generate_dataset()
        train_set, val_set = self._dataset_generator.train_split, self._dataset_generator.val_split
        LOGGER.info(f'Size of training/validation set: ({len(train_set)}/{len(val_set)})')
        self._datasets = dict(train=train_set, val=val_set)

    def _get_additional_train_step_log(self, step: int) -> Dict[str, Any]:
        log_dict = {}
        if self._log_additional_logs and step % int(
                self._log_additional_train_step_every_multiplier * self._log_train_step_every) == 0:
            # norm of model parameter vector
            model_param_vec = nn.utils.parameters_to_vector(self._model.parameters())
            model_param_norm = torch.linalg.norm(model_param_vec, ord=2).item()
            log_dict.update({'weight_norm': model_param_norm})

            # subspace regularizer logs
            if self._subspace_regularizer:
                additional_logs = self._subspace_regularizer.get_additional_logs()
                log_dict.update(additional_logs)

        return log_dict

    def _hook_on_val_epoch_end(self, progress_idx: int, trained_model: nn.Module) -> None:
        self._plot_predictions(epoch=progress_idx, model=trained_model)

    def _plot_predictions(self, epoch: int, model: nn.Module) -> None:
        if self._plot_predictions_every_val_multiplier > 0 and epoch % int(
                self._val_every * self._plot_predictions_every_val_multiplier) == 0:
            if isinstance(self._dataset_generator, SupervisedMetaDatasetWrapper):
                task = self._dataset_generator.get_meta_task()
                query_set = support_query_as_minibatch(task.query_set, device=self.device)
                with torch.no_grad():
                    query_preds = model(query_set[0])
                fig, fname = task.plot_query_predictions(epoch, preds={0: query_preds})
                # save fig & log to wandb
                save_path = self._experiment_dir / SAVEDIR_PRED_PLOT
                save_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path / fname, bbox_inches='tight', dpi=DPI)
                plt.close(fig)