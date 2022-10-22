from pathlib import Path
from typing import List, Union
from omegaconf import DictConfig
import torch
from torch.utils import data
from torch import nn 
from tqdm import tqdm

from ml_utilities.utils import get_device
from ml_utilities.torch_models import get_model_class
from ml_utilities.torch_utils import get_loss, gradients_to_vector
from erank.data.datasetgenerator import DatasetGenerator

from erank.utils import load_best_model, load_model_from_idx


class GradientCalculator:

    def __init__(self,
                 dataset_generator_kwargs: DictConfig,
                 model_name: str = '',
                 model_path: Union[str, Path] = None,
                 run_path: Union[str, Path] = None,
                 model_idx: int = -1,
                 default_loss: Union[str, nn.Module] = None,
                 device: Union[str, int] = 'auto'):
        self.device = get_device(device)
        # load model
        if model_name and model_path:
            model_class = get_model_class(model_name)
            model = model_class.load(model_path, device=self._device)
        elif run_path:
            if model_idx == -1:
                model = load_best_model(run_path, device=self.device)
            else:
                model = load_model_from_idx(run_path, idx=model_idx, device=self.device)
        else:
            raise ValueError('No model provided!')
        self.model = model

        # generate dataset
        self.data_cfg = dataset_generator_kwargs
        dataset_generator = DatasetGenerator(**self.data_cfg)
        dataset_generator.generate_dataset()
        self.dataset = dataset_generator.train_split

        # create loss
        if default_loss:
            default_loss = self.__init_loss(default_loss)
        self.default_loss_fn = default_loss

    def __init_loss(self, loss: Union[str, nn.Module]):
        if isinstance(loss, str):
            loss_cls = get_loss(loss)  
            loss_fn = loss_cls(reduction='mean')
        else:
            loss_fn = loss
        return loss_fn

    def compute_gradients(self, batch_size: int, num_gradients: int = -1, loss: Union[str, nn.Module] = None) -> List[torch.Tensor]:
        dataloader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        loss_fn = self.default_loss_fn
        if loss_fn is None:
            assert not loss is None, 'No loss function given to compute the gradients!'
            loss_fn = self.__init_loss(loss)

        gradients = []

        it = range(num_gradients) if num_gradients > 0 else range(len(dataloader))
        data_iter = iter(dataloader)

        for batch_idx in tqdm(it):
            batch = next(data_iter)
            xs, ys = batch
            xs, ys = xs.to(self.device), ys.to(self.device)
            
            ys_pred = self.model(xs)
            loss = loss_fn(ys_pred, ys)
            self.model.zero_grad()
            loss.backward()

            grad = gradients_to_vector(self.model.parameters())
            gradients.append(grad)

        return gradients








def apply_gradient_mask():
    # TODO implment this function
    # find best way to store gradient mask: option a) store torch tensor where idx to mask are 0, option b) store parameter dict, [option c) store full model]
    # probably best way is option b)
    pass
