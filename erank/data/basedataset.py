
from typing import List, Union, Dict
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BaseDataset(Dataset):

    def __init__(self):
        self._normalizer : Dict[str, Union[float, List, float]] = None

    @property
    def normalizer_values(self) -> Dict[str, Union[float, List, float]]:
        return self._normalizer

    @property
    def normalizer(self) -> nn.Module:
        return transforms.Normalize(self._normalizer['mean'], self._normalizer['std']) if self._normalizer is not None else None