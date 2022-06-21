import logging
import wandb
import torch
import torchmetrics
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import nn
from omegaconf import OmegaConf
from pathlib import Path
from ml_utilities.torch_utils.factory import create_optimizer_and_scheduler
from ml_utilities.torch_utils import get_optim
from ml_utilities.torch_models import get_model_class
from ml_utilities.trainers.basetrainer import BaseTrainer
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

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
    def _setup(self):
        exp_data = self.config.experiment_data
        wandb.init(project=exp_data.project_name, name=HydraConfig.get().job.name, dir=Path.cwd(),
                   config=self.config, tags=self.config.tags, notes=self.config.notes)

    def _create_datasets(self) -> None:
        # create fashion mnist datasets
        LOGGER.info('Loading train/val dataset.')
        data_conf = self.config.data
        data_dir = Path(get_original_cwd()) / data_conf.dataset_dir
        normalizer = data_conf.normalizer
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(normalizer.mean, normalizer.std)])

        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
        train_set, val_set = data.random_split(train_dataset, [50000, 10000])
        self._datasets = dict(train=train_set, val=val_set)

    def _create_dataloaders(self) -> None:
        train_loader = data.DataLoader(
            dataset=self._datasets['train'], batch_size=self.config.trainer.batch_size, shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)
        val_loader = data.DataLoader(dataset=self._datasets['val'], batch_size=self.config.trainer.batch_size,
                                     shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)
        self._loaders = dict(train=train_loader, val=val_loader)

    def _create_model(self) -> None:
        LOGGER.info('Creating model.')
        model_class = get_model_class(self.config.model.name)
        self._model = model_class(**self.config.model.model_kwargs)

    def _create_optimizer_and_scheduler(self, model: nn.Module) -> None:
        LOGGER.info('Create optimizer and scheduler.')
        self._optimizer, self._lr_scheduler = create_optimizer_and_scheduler(model.parameters(), **self.config.trainer.optimizer_scheduler)

    def _create_loss(self) -> None:
        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def _create_metrics(self) -> None:
        self._metrics = [torchmetrics.Accuracy()]

    def _train_epoch(self, epoch: int) -> None:
        losses = []
        metric_vals = dict()

        pbar = tqdm(self._loaders['train'], desc=f'Train epoch {epoch}')
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)
            # forward pass
            y_pred = self._model(xs)

            loss = self._loss(y_pred, ys)
            # backward pass
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._train_step += 1

            # metrics & logging
            losses.append(loss.item())
            with torch.no_grad():
                for metric in self._metrics:
                    metric_vals[metric._get_name()] = metric(y_pred, ys).item()

            wandb.log({'train': {'epoch': epoch, 'train_step': self._train_step, 'loss': loss.item(), **metric_vals}})

        # wandb.log({'epoch': epoch, 'train_step': self._train_step, 'loss': torch.tensor(losses).mean()})

        self._reset_metrics()

    def _val_epoch(self, epoch: int, trained_model: nn.Module) -> float:

        metric_vals = dict()
        for metric in self._metrics:
            metric_vals[metric._get_name()] = []
        val_losses = []

        pbar = tqdm(self._loaders['val'], desc=f'Val epoch {epoch}')
        for xs, ys in pbar:
            xs, ys = xs.to(self.device), ys.to(self.device)

            with torch.no_grad():
                y_pred = trained_model(xs)

                loss = self._loss(y_pred, ys)
                val_losses.append(loss.item())
                for metric in self._metrics:
                    metric_vals[metric._get_name()].append(metric(y_pred, ys).item())

        # compute mean metrics over dataset
        for metric in self._metrics:
            metric_vals[metric._get_name()] = torch.tensor(metric_vals[metric._get_name()]).mean(dim=0)

        wandb.log({'val': {'epoch': epoch, 'loss': torch.tensor(val_losses).mean(), **metric_vals}})

        # val_score is first metric in self._metrics
        val_score = metric_vals[self._metrics[0]._get_name()].item()
        
        self._reset_metrics()
        return val_score

    def _final_hook(self, *args, **kwargs):
        pass