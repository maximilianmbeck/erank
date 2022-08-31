from typing import Type
from erank.regularization.base_regularizer import Regularizer
from erank.regularization.dotproduct_regularizer import DotProductRegularizer
from erank.regularization.erank_regularizer import EffectiveRankRegularizer

_regularizer_registry = {'erank': EffectiveRankRegularizer, 'dotproduct': DotProductRegularizer}


def get_regularizer_class(regularizer_type: str) -> Type[Regularizer]:
    if regularizer_type in _regularizer_registry:
        return _regularizer_registry[regularizer_type]
    else:
        assert False, f'Unknown regularizer type \"{regularizer_type}\". Available regularizer types are: {str(_regularizer_registry.keys())}'