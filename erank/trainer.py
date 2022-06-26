import logging
import wandb
import torch
import torchmetrics
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm
from torch import nn
from omegaconf import OmegaConf
from pathlib import Path
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler
from ml_utilities.torch_utils import get_optim
from ml_utilities.torch_models import get_model_class
from ml_utilities.torch_models.fc import FC
from ml_utilities.trainers.basetrainer import BaseTrainer
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from erank.data import random_split_train_tasks
from erank.regularization import EffectiveRankRegularization

LOGGER = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def __init__(self, config: OmegaConf):
        self.config = config
        super().__init__(experiment_dir=config.experiment_data.experiment_dir,
                         seed=config.experiment_data.seed,
                         gpu_id=config.experiment_data.gpu_id,
                         n_epochs=config.trainer.n_epochs,
                         val_every=config.trainer.val_every,
                         save_every=config.trainer.save_every,
                         early_stopping_patience=config.trainer.early_stopping_patience)
        #
        self._erank_regularizer: EffectiveRankRegularization = None

    def _setup(self):
        LOGGER.info('Starting wandb.')
        exp_data = self.config.experiment_data
        # add wandb config: need to convert to python native dict
        wandb.init(project=exp_data.project_name, name=HydraConfig.get().job.name, dir=Path.cwd(),
                   config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
                   tags=self.config.tags, notes=self.config.notes)

    def _create_datasets(self) -> None:
        # create fashion mnist datasets
        LOGGER.info('Loading train/val dataset.')
        data_cfg = self.config.data
        data_dir = Path(get_original_cwd()) / data_cfg.dataset_dir
        normalizer = data_cfg.normalizer
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(normalizer.mean, normalizer.std)])

        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)

        train_set, val_set = random_split_train_tasks(
            train_dataset, num_train_tasks=data_cfg.num_train_tasks, train_task_idx=data_cfg.train_task_idx,
            train_val_split=data_cfg.train_val_split)
        self._datasets = dict(train=train_set, val=val_set)

    def _create_dataloaders(self) -> None:
        train_loader = data.DataLoader(
            dataset=self._datasets['train'],
            batch_size=self.config.trainer.batch_size, shuffle=True, drop_last=False,
            num_workers=self.config.trainer.num_workers)
        val_loader = data.DataLoader(dataset=self._datasets['val'], batch_size=self.config.trainer.batch_size,
                                     shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)
        self._loaders = dict(train=train_loader, val=val_loader)

    def _create_model(self) -> None:
        LOGGER.info('Creating model.')
        model_class = get_model_class(self.config.model.name)
        if self.config.trainer.init_model:
            LOGGER.info(f'Loading model {self.config.trainer.init_model} to device {self.device}.')
            self._model = model_class.load(self.config.trainer.init_model, device=self.device)
        else:
            self._model = model_class(**self.config.model.model_kwargs)

    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        LOGGER.info('Create optimizer and scheduler.')
        self._optimizer, self._lr_scheduler = create_optimizer_and_scheduler(
            model.parameters(), **self.config.trainer.optimizer_scheduler)

    def _create_loss(self) -> None:
        LOGGER.info('Creating loss.')
        self._loss = nn.CrossEntropyLoss(reduction='mean')

        erank_cfg = self.config.trainer.erank
        erank_reg = None
        if erank_cfg.type == 'none':
            LOGGER.info('No erank regularizer.')
        elif erank_cfg.type in ['random', 'pretraindiff']:
            LOGGER.info(f'Erank regularization of type {erank_cfg.type}.')
            erank_reg = EffectiveRankRegularization(
                buffer_size=erank_cfg.buffer_size, init_model=self._model, loss_weight=erank_cfg.loss_weight,
                normalize_directions=erank_cfg.get('norm_directions', False),
                use_abs_model_params=erank_cfg.get('use_abs_model_params', False))
            if erank_cfg.type == 'random':
                erank_reg.init_directions_buffer(random_buffer=True)
            elif erank_cfg.type == 'pretraindiff':
                erank_reg.init_directions_buffer(path_to_buffer_or_runs=erank_cfg.dir_buffer)
        else:
            raise ValueError('Unknown erank type.')

        self._erank_regularizer = erank_reg

    def _create_metrics(self) -> None:
        LOGGER.info('Creating metrics.')
        self._metrics = [torchmetrics.Accuracy()]

    def _train_epoch(self, epoch: int) -> None:
        loss_vals = dict(loss_total=[], loss_ce=[], loss_erank=[])
        metric_vals = dict()

        pbar = tqdm(self._loaders['train'], desc=f'Train epoch {epoch}')
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)
            # forward pass
            y_pred = self._model(xs)

            loss = self._loss(y_pred, ys)

            # add erank regularizer
            loss_reg = torch.tensor(0.0).to(loss)
            if self._erank_regularizer is not None:
                loss_reg = self._erank_regularizer.forward(self._model)

            loss_total = loss + loss_reg

            # backward pass
            self._optimizer.zero_grad()
            loss_total.backward()
            self._optimizer.step()
            self._train_step += 1

            # update regularizer
            if self._erank_regularizer is not None:
                self._erank_regularizer.update_delta_start_params(self._model)

            # metrics & logging
            loss_log = dict(loss_total=loss_total.item(), loss_ce=loss.item(), loss_erank=loss_reg.item())

            for loss_name in loss_vals:
                loss_vals[loss_name] = loss_log[loss_name]
            with torch.no_grad():
                for metric in self._metrics:
                    metric_vals[metric._get_name()] = metric(y_pred, ys).item()

            # log step
            wandb.log({'train_step/': {'epoch': epoch, 'train_step': self._train_step,
                      **loss_vals, **metric_vals}})

        # log epoch
        for metric in self._metrics:
            metric_vals[metric._get_name()] = metric.compute().item()

        for loss_name, loss_val_list in loss_vals.items():
            loss_vals[loss_name] = torch.tensor(loss_val_list).mean().item()

        log_dict = {'epoch': epoch, 'train_step': self._train_step,
                    **loss_vals, **metric_vals}
        wandb.log({'train_epoch/': log_dict})

        LOGGER.info(f'Train epoch \n{pd.Series(log_dict)}')

        self._reset_metrics()

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:

        metric_vals = dict()
        val_losses = []

        pbar = tqdm(self._loaders['val'], desc=f'Val epoch {epoch}')
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)

            with torch.no_grad():
                y_pred = trained_model(xs)

                loss = self._loss(y_pred, ys)
                val_losses.append(loss.item())
                for metric in self._metrics:
                    m_val = metric(y_pred, ys)

        # compute mean metrics over dataset
        for metric in self._metrics:
            metric_vals[metric._get_name()] = metric.compute().item()

        # log epoch
        log_dict = {'epoch': epoch, 'loss': torch.tensor(val_losses).mean().item(), **metric_vals}
        wandb.log({'val/': log_dict})

        LOGGER.info(f'Val epoch \n{pd.Series(log_dict)}')

        # val_score is first metric in self._metrics
        val_score = metric_vals[self._metrics[0]._get_name()]

        self._reset_metrics()
        return val_score

    def _final_hook(self, *args, **kwargs):
        pass
