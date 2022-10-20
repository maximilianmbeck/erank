import logging
from torch import nn
from omegaconf import DictConfig
from ml_utilities.torch_utils import get_loss
from ml_utilities.trainers.supervisedbasetrainer import SupervisedBaseTrainer
from erank.regularization import get_regularizer_class
from erank.regularization.regularized_loss import RegularizedLoss
from erank.regularization.subspace_regularizer import SubspaceRegularizer

LOGGER = logging.getLogger(__name__)


class SubspaceBaseTrainer(SupervisedBaseTrainer):
    """Abstract trainer for this project. Collects all common functionalitites across trainers.

    Functionality is further specialized in child classes.

    Args:
        config (DictConfig): The configuration.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config=config)
        
        self._subspace_regularizer: SubspaceRegularizer = None

        self._log_additional_train_step_every_multiplier = self.config.trainer.get(
            'log_additional_train_step_every_multiplier', 1)
        self._log_additional_logs = self.config.trainer.get('log_additional_logs', False)

    def _create_loss(self) -> None:
        LOGGER.info('Creating loss.')
        loss_cls = get_loss(self.config.trainer.loss)
        loss_module = loss_cls(reduction='mean')
        self._loss = RegularizedLoss(loss_module)

        self._subspace_regularizer = self._create_subspace_regularizer(self._model)
        if self._subspace_regularizer:
            self._loss.add_regularizer(self._subspace_regularizer)

    def _create_subspace_regularizer(self, model: nn.Module) -> SubspaceRegularizer:
        subspace_reg_cfg = self.config.trainer.get('regularizer', None)
        subspace_reg = None

        if subspace_reg_cfg is None or subspace_reg_cfg.type == 'none':
            LOGGER.info('No regularizer.')
        elif subspace_reg_cfg.init_type in ['random', 'weightsdiff', 'buffer']:
            LOGGER.info(f'Subspace regularizer: {subspace_reg_cfg.type}')
            regularizer_cls = get_regularizer_class(subspace_reg_cfg.type)
            regularizer_kwargs = subspace_reg_cfg.regularizer_kwargs
            subspace_reg = regularizer_cls(init_model=model, device=self.device, **regularizer_kwargs)

            LOGGER.info(f'Subspace regularization of init_type {subspace_reg_cfg.init_type}.')
            if subspace_reg_cfg.init_type == 'random':
                subspace_reg.init_subspace_vecs(random_buffer=True)
            elif subspace_reg_cfg.init_type == 'weightsdiff':
                dir_buffer_path = subspace_reg_cfg.get('init_dir_buffer', None)
                if dir_buffer_path is None:
                    raise ValueError(f'Subspace init_type is `weightsdiff`, but no buffer path is given!')
                subspace_reg.init_subspace_vecs(path_to_buffer_or_runs=dir_buffer_path)
            elif subspace_reg_cfg.init_type == 'buffer':
                pass  # does nothing, buffer is filled during training
        else:
            raise ValueError(f'Unknown regularizer type: {subspace_reg_cfg.init_type}')
        return subspace_reg
