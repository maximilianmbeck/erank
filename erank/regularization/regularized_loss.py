import torch
import logging
from typing import Dict, Tuple
from torch import nn

from erank.regularization.base_regularizer import Regularizer

LOGGER = logging.getLogger(__name__)

LOG_LOSS_PREFIX = 'loss'
LOG_LOSS_TOTAL_KEY = f'{LOG_LOSS_PREFIX}_total'

class RegularizedLoss(nn.Module):

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss_module = loss
        self._regularizers: nn.ModuleDict[str, Regularizer] = nn.ModuleDict({})

    def forward(self,
                y_preds: torch.Tensor,
                y_labels: torch.Tensor,
                model: nn.Module = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = {}
        loss = self.loss_module(y_preds, y_labels)
        if torch.isinf(loss) or torch.isnan(loss):
            LOGGER.warning(
                f'{LOG_LOSS_PREFIX}_{self.loss_module.__class__.__name__} is Inf or NaN! Inf: {torch.isinf(loss)} | NaN: {torch.isnan(loss)}'
            )
        loss_dict[f'{LOG_LOSS_PREFIX}_{self.loss_module.__class__.__name__}'] = loss
        loss_total = loss
        if not model is None:
            for reg_name, reg in self._regularizers.items():
                loss_reg = reg(model)
                loss_dict[f'{LOG_LOSS_PREFIX}_{reg_name}'] = loss_reg
                loss_total += reg.loss_coefficient * loss_reg
                if torch.isinf(loss_reg) or torch.isnan(loss_reg):
                    LOGGER.warning(
                        f'{LOG_LOSS_PREFIX}_{reg_name} is Inf or NaN! Inf: {torch.isinf(loss_reg)} | NaN: {torch.isnan(loss_reg)}'
                    )

        loss_dict[LOG_LOSS_TOTAL_KEY] = loss_total

        return loss_total, loss_dict

    def add_regularizer(self, regularizer: Regularizer) -> None:
        if regularizer.name in self._regularizers.keys():
            raise ValueError(f'Regularizer {regularizer.name} already exists in RegularizedLoss.')
        else:
            LOGGER.info(f'Adding regularizer `{regularizer.name}` to RegularizedLoss')
            self._regularizers[regularizer.name] = regularizer

    def get_regularizer(self, regularizer_name: str) -> Regularizer:
        return self._regularizers[regularizer_name]

    def contains_regularizer(self, regularizer_name: str) -> bool:
        return regularizer_name in self._regularizers