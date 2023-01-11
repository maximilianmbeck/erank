from typing import Any, Dict, Type, Callable, Union
import torchvision
import inspect
from omegaconf import OmegaConf, DictConfig

available_pytorch_visiontransforms = torchvision.transforms.transforms.__all__

__pytorch_visiontransforms = inspect.getmembers(
    torchvision.transforms, lambda transform_class: inspect.isclass(transform_class) and transform_class.__name__ in
    available_pytorch_visiontransforms)

_pytorch_transforms_registry = {name: cls for name, cls in __pytorch_visiontransforms}

# for now there are no custom transforms
_transforms_registry = _pytorch_transforms_registry


def get_transform_class(transform_name: str) -> Type:
    if transform_name in _transforms_registry:
        return _transforms_registry[transform_name]
    else:
        assert False, f"Unknown transform name \"{transform_name}\". Available transforms are: {str(_transforms_registry.keys())}"


def create_transform(transform_cfg: Union[str, Dict[str, Any]]) -> Callable:
    if isinstance(transform_cfg, str):
        transform_cls = get_transform_class(transform_cfg)
        return transform_cls()
    elif isinstance(transform_cfg, (dict, DictConfig)):
        transform_name = list(transform_cfg.keys())
        assert len(transform_name) == 1, f'No or multiple transform config passed. Expect only one!'
        transform_name = transform_name[0]
        transform_cls = get_transform_class(transform_name)
        transform_args = OmegaConf.to_container(transform_cfg[transform_name])
        transform = transform_cls(*transform_args) if isinstance(transform_args, list) else transform_cls(
            **transform_args)
        return transform
    else:
        raise TypeError