from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.torch_models.cnn2d import CnnBlockConfig, create_cnn_layer
from ml_utilities.torch_models.fc import create_linear_output_layer


def get_resnet(resnet_name: str):
    # Return the resnet from literature
    # TODO create configs for standard resnets (i.e. resnet18-imagenet, resnet20-cifar)
    pass


@dataclass
class ResnetResidualBlockConfig:
    in_channels: int
    out_channels: int
    num_residual_blocks: int
    # A: CIFAR10 (zeros padded for extra dimensions, no additional parameters), B: ImageNet (1x1 conv, additional paramters)
    residual_option: str = 'A'
    first_block: bool = False


@dataclass
class ResnetConfig:
    in_channels: int
    input_layer_config: Union[CnnBlockConfig, Dict[str, Any]]
    resnet_blocks_config: List[Union[ResnetResidualBlockConfig, Dict[str, Any]]]
    residual_option: str = 'A'
    linear_output_units: List[int] = None
    act_fn: str = "relu"
    output_activation: str = None


class Resnet(BaseModel):
    """Implementation of Resnet architecture.

    References:
        .. [#] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
               arXiv:1512.03385 (2015)
        .. [#] https://github.com/akamaster/pytorch_resnet_cifar10
        .. [#] https://d2l.ai/chapter_convolutional-modern/resnet.html 
    """

    def __init__(self, in_channels: int,
                 input_layer_config: Union[CnnBlockConfig, Dict[str, Any]],
                 resnet_blocks_config: List[Union[ResnetResidualBlockConfig, Dict[str, Any]]],
                 residual_option: str = 'A',
                 linear_output_units: List[int] = None,
                 act_fn: str = "relu",
                 output_activation: str = None):
        super().__init__()

        self.config = ResnetConfig(in_channels=in_channels,
                                   input_layer_config=input_layer_config,
                                   resnet_blocks_config=resnet_blocks_config,
                                   residual_option=residual_option,
                                   linear_output_units=linear_output_units,
                                   act_fn=act_fn,
                                   output_activation=output_activation)

        self.resnet = create_resnet(**asdict(self.config))

    def forward(self, x):
        return self.resnet(x)

    def reset_parameters(self):
        return self.resnet.reset_parameters()


def create_resnet(in_channels: int,
                  input_layer_config: Union[CnnBlockConfig, Dict[str, Any]],
                  resnet_blocks_config: List[Union[ResnetResidualBlockConfig, Dict[str, Any]]],
                  residual_option: str = 'A',
                  linear_output_units: List[int] = None,
                  act_fn: str = "relu",
                  output_activation: str = None) -> nn.Module:
    if not isinstance(input_layer_config, CnnBlockConfig):
        input_layer_config = CnnBlockConfig(**input_layer_config)
    input_layer_config.in_channels = in_channels
    input_layer_config.out_channels = resnet_blocks_config[0].in_channels
    input_layer_config.act_fn = act_fn

    input_layer = create_cnn_layer(**asdict(input_layer_config))

    residual_layers = _create_resnet_residual_layers(resnet_blocks_config, residual_option)

    global_avg_pool_layer = nn.AdaptiveAvgPool2d((1, 1))

    num_output_layer_input_features = resnet_blocks_config[-1].out_channels
    output_layer = create_linear_output_layer(
        in_features=num_output_layer_input_features, out_units=linear_output_units, flatten_input=True,
        output_activation=output_activation, act_fn=act_fn)

    layers = [input_layer, residual_layers, global_avg_pool_layer, output_layer]
    return nn.Sequential(*layers)


def _create_resnet_residual_layers(
        resnet_blocks_config: List[Union[ResnetResidualBlockConfig, Dict[str, Any]]],
        residual_option: str = None) -> nn.Module:

    resnet_blocks = []
    first_block = True
    for blk_cfg in resnet_blocks_config:
        if not isinstance(blk_cfg, ResnetResidualBlockConfig):
            blk_cfg = ResnetResidualBlockConfig(**blk_cfg)
        blk_cfg.first_block = first_block
        first_block = False
        if residual_option is not None:
            blk_cfg.residual_option = residual_option
        resnet_blocks.append(_ResidualBlock.create_resdiual_block_sequence(**asdict(blk_cfg)))

    return nn.Sequential(*resnet_blocks)

class _LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]):
        super(_LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class _ResidualBlock(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, in_channels: int, num_channels: int, stride: int = 1, residual_option: str = 'A'):
        """
        Parameters
        ----------
        option : str
            either 'A' or 'B'
            refers to the paper Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
            A: pad zeros in skip connections where dimensions do not match
            B: use 1x1 convolutions in skip connections where dimensions do not match
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3,
                               padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)

        self.skip_connect = nn.Identity()
        if stride != 1 or in_channels != num_channels:
            if residual_option == 'A':
                # num_channels // 4 appends zeros at top and bottom of output channel dimension
                # ::2 takes only every second entry in the feature map
                # i.e. x.shape = [1, 16, 32, 32] gets [1, 32, 16, 16]
                self.skip_connect = _LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, num_channels // 4, num_channels // 4), "constant", 0))
            elif residual_option == 'B':
                self.skip_connect = nn.Sequential(nn.Conv2d(in_channels, num_channels,
                                                            kernel_size=1, stride=stride),
                                                  nn.BatchNorm2d(num_channels))
            else:
                raise ValueError(f'Unknown residual option: {residual_option}')
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    @staticmethod
    def create_resdiual_block_sequence(in_channels: int, out_channels: int, num_residual_blocks: int,
                                       first_block: bool = False, residual_option: str = 'A') -> nn.Module:
        blk = []
        for i in range(num_residual_blocks):
            if i == 0 and not first_block:
                blk.append(
                    _ResidualBlock(in_channels, out_channels, stride=2, residual_option=residual_option))
            else:
                blk.append(_ResidualBlock(out_channels, out_channels, residual_option=residual_option))
        return nn.Sequential(*blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.skip_connect(x)
        return F.relu(y)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if isinstance(self.skip_connect, nn.Conv2d):
            self.skip_connect.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
