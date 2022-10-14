from typing import Dict, Callable, Tuple, Union
import sys
import torch
import pandas as pd
from tqdm import tqdm

from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.utils import set_seed, get_device
from erank.regularization.subspace_regularizer import SubspaceRegularizer


def create_model_erank_df(
    models: Union[torch.Tensor, Dict[str, torch.Tensor]],
    model_steps: int = 10,
    random_baseline: bool = True,
    random_init_models: Union[BaseModel, Dict[str, BaseModel]] = None,
    erank_fn: Callable[[torch.Tensor], torch.Tensor] = SubspaceRegularizer.erank,
    seed: int = 0,
    device: Union[torch.device, str, int] = "auto"
) -> pd.DataFrame:  # columns: different model_sequences, rows: erank with number of models
    GAUSS_RAND_KEY = 'Gaussian random'
    RAND_MODEL_INIT_KEY = 'Random model initializations'
    TRAINED_MODELS_KEY = 'Trained models'

    def _erank(matrix: torch.Tensor) -> float:
        return erank_fn(matrix).item()

    def _generate_randinit_matrix(random_init_model: BaseModel, name_suffix: str = '') -> Tuple[str, torch.Tensor]:
        random_init_model.to(device=device)
        random_init_model_vecs = []
        descr = f'{RAND_MODEL_INIT_KEY}_{name_suffix}' if name_suffix else RAND_MODEL_INIT_KEY
        for i in tqdm(range(num_vectors), file=sys.stdout, desc='Generate random initializations'):
            random_init_model.reset_parameters()
            random_init_model_vecs.append(torch.nn.utils.parameters_to_vector(random_init_model.parameters()))
        random_init_model_vecs = torch.stack(random_init_model_vecs).detach().to(device=device)
        return descr, random_init_model_vecs

    device = get_device(device)

    # setup
    set_seed(seed)
    erank_data = {}
    model_matrix_shape = None  # (num_models, num_model_parameters)
    if isinstance(models, torch.Tensor):
        model_matrix_shape = models.shape
        models = {TRAINED_MODELS_KEY: models.to(device=device)}
        erank_data.update({TRAINED_MODELS_KEY: []})
    elif isinstance(models, dict):
        model_erank_data = {}
        for descr, model_matrix in models.items():
            models[descr] = model_matrix.to(device=device)
            if not model_matrix_shape:
                model_matrix_shape = model_matrix.shape
            assert model_matrix_shape == model_matrix.shape, f'Shape of `{descr}` model matrix does not match!'
            model_erank_data[descr] = []
        erank_data.update(model_erank_data)

    num_vectors = model_matrix_shape[0]
    assert isinstance(models, dict)

    random_vecs = None
    if random_baseline:
        random_vecs = torch.randn(model_matrix_shape, device=device)
        erank_data.update({GAUSS_RAND_KEY: []})

    if isinstance(random_init_models, BaseModel):
        random_init_models = {'': random_init_models}
    
    random_init_model_matrices = {}
    for descr, random_init_model in random_init_models.items():
        random_model_descr, model_matrix = _generate_randinit_matrix(random_init_model, name_suffix=descr)
        random_init_model_matrices[random_model_descr] = model_matrix
        erank_data.update({random_model_descr: []})



    # data generation loop
    model_idxes = torch.randperm(num_vectors)
    num_vec_idxes = list(range(1, len(model_idxes), model_steps))
    if not num_vectors in num_vec_idxes:
        num_vec_idxes.append(num_vectors)
    for i in tqdm(num_vec_idxes, file=sys.stdout, desc='Calculate eranks'):
        if random_baseline:
            erank_data[GAUSS_RAND_KEY].append(_erank(random_vecs[:i]))
        if random_init_models:
            for descr, model_matrix in random_init_models.items():
                erank_data[descr].append(_erank(model_matrix[:i]))

        for descr, model_matrix in models.items():
            erank_data[descr].append(_erank(model_matrix[model_idxes[:i]]))

    df_index = pd.Index(num_vec_idxes, dtype=int, name='Number of Vectors')

    return pd.DataFrame(erank_data, index=df_index)
