import logging
import wandb
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from pathlib import Path
from ml_utilities.utils import set_seed
from ml_utilities.torch_utils import get_optim
from ml_utilities.torch_models import get_model_class
from hydra.utils import get_original_cwd

LOGGER = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, config: OmegaConf):
        self.config = config
        self._setup()
        self._initialize()

    def _setup(self):
        set_seed(self.config.experiment_data.seed)
        exp_data = self.config.experiment_data
        wandb.init(project=exp_data.project_name, name=exp_data.experiment_name, dir=Path.cwd(),
                   config=self.config, tags=self.config.tags, notes=self.config.notes)

    def _initialize(self):
        # create fashion mnist datasets
        LOGGER.info('Loading train/val dataset.')
        data_conf = self.config.data
        data_dir = Path(get_original_cwd()) / data_conf.dataset_dir
        normalizer = data_conf.normalizer
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(normalizer.mean, normalizer.std)])

        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
        train_set, val_set = data.random_split(train_dataset, [50000, 10000])

        self.train_loader = data.DataLoader(dataset=train_set, batchsize=self.config.trainer.batchsize, shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)
        self.val_loader = data.DataLoader(dataset=val_set, batchsize=self.config.trainer.batchsize, shuffle=True, drop_last=False, num_workers=self.config.trainer.num_workers)

        # create model
        LOGGER.info('Creating model.')
        model_class = get_model_class(self.config.model.name)
        self.model = model_class(**self.config.model.model_kwargs)
        
        # create optimizer
        optim_class = get_optim(self.config.trainer.optimizer)
        self.optimizer = optim_class(self.model.parameters(), self.config.trainer.optimizer_kwargs)

    def train(self):
        LOGGER.info('Starting training..')
        
        for epoch in range(self.config.trainer.epochs):
            self.train_epoch()

    def train_epoch(self):
        pass

    def test(self):
        raise NotImplementedError()
