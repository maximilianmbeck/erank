import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Union
from erank.data.basemetaclassificationdataset import BaseMetaClassificationDataset, ClassificationTask
from erank.data.basemetadataset import BaseMetaDataset, Task, QUERY_X_KEY, QUERY_Y_KEY, SUPPORT_X_KEY, SUPPORT_Y_KEY
from PIL import Image

CLASS_NAME_TEMPLATE = '{alphabet}--{character}'
OMNIGLOT_METADATASET_TRAIN_NORMALIZER = {
    'mean': [0.9213101208773438],
    'std': [0.2628733349463854],
    'num_dataset_samples': 17660
}

class OmniglotDataset(BaseMetaClassificationDataset):
    """Omniglot dataset

    The description as well as train/val/test splits are borrowed from Meta-Dataset:

    Omniglot is organized into two high-level directories, referred to as
    the background and evaluation sets, respectively, with the former
    intended for training and the latter for testing. Each of these contains a
    number of sub-directories, corresponding to different alphabets.
    Each alphabet directory in turn has a number of sub-folders, each
    corresponding to a character, which stores 20 images of that character, each
    drawn by a different person.
    We consider each character to be a different class for our purposes.
    The following diagram illustrates this struture.

    omniglot_root
    |- images_background
       |- alphabet
          |- character
             |- images of character
          ...
    |- images_evaluation
      |- alphabet
          |- character
             |- images of character
          ...

    The Omniglot dataset has about 15MB on disk.

    We use Lake's original train/test splits as we believe this is a more
    challenging setup and because we like that it's hierarchically structured.
    We also held out a subset of that train split to act as our validation set.
    Specifically, the 5 alphabets from that set with the least number of
    characters were chosen for this purpose.

    Args:
        data_root_path (Union[str, Path]): Root directory of Omniglot dataset. Must contain a folder named `omniglot` 
                                           containing the dataset files with the correct file structure.
        support_size (int): Num. train samples per task.
        query_size (int): Num. test samples per task
        num_tasks (int, optional): Number of N-way-k-shot combinations. Defaults to -1 (as many as possible).
        split (str): Either 'train', 'val' or 'test'. 'val' is only available in 'metadataset' layout.
        dataset_layout (str, optional): The dataset layout to use. Either 'lake' or 'metadataset'.
                                        'lake': train/test split from original publication.
                                        'metadataset': train/val/test split of Meta-dataset.
                                        Defaults to 'metadataset'.
        seed (int, optional): Seed for data sampling. Defaults to 0.

    References:
        .. [#] Triantafillou, Eleni, et al. "Meta-dataset: A dataset of datasets for learning to learn from few examples." arXiv preprint arXiv:1903.03096 (2019).
        .. [#] https://github.com/google-research/meta-dataset/blob/0bf2cf45e7296b703b2eae59042f5af4839f2ddc/meta_dataset/dataset_conversion/dataset_to_records.py#L659
        .. [#] https://github.com/brendenlake/omniglot
        .. [#] Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "The Omniglot challenge: a 3-year progress report." 
               Current Opinion in Behavioral Sciences 29 (2019): 97-104.
    """
    name = 'omniglot'

    dataset_folder_name = 'omniglot'
    toplevel_folders_n_alphabets = {
        'images_background': 30,  # original train data 
        'images_evaluation': 20,  # original test data
    }
    images_per_character = 20
    dataset_layout_split_options = {'metadataset': ['train', 'val', 'test'], 'lake': ['train', 'test']}
    dataset_split_toplevel_folders = {
        'train': 'images_background',
        'val': 'images_background',
        'test': 'images_evaluation'
    }
    # We chose the 5 smallest alphabets (i.e. those with the least characters)
    # out of the 'background' set of alphabets that are intended for train/val
    # We keep the 'evaluation' set of alphabets for testing exclusively
    # The chosen alphabets have 14, 14, 16, 17, and 20 characters, respectively.
    metadataset_validation_alphabets = [
        'Blackfoot_(Canadian_Aboriginal_Syllabics)', 'Ojibwe_(Canadian_Aboriginal_Syllabics)',
        'Inuktitut_(Canadian_Aboriginal_Syllabics)', 'Tagalog', 'Alphabet_of_the_Magi'
    ]
    output_img_size = (28, 28)
    # precomputed normalizer using all data in 'images_background'
    normalizer = OMNIGLOT_METADATASET_TRAIN_NORMALIZER

    def __init__(
            self,
            data_root_path: Union[str, Path],
            n_way_classification: int,
            support_size: int,
            query_size: int,
            split: str,
            num_tasks: int = 0,
            regenerate_task_support_set: bool = True,
            regenerate_task_query_set: bool = True,
            dataset_layout: str = 'metadataset',
            seed: int = 0,
            normalizer: Dict[str, List[float]] = OMNIGLOT_METADATASET_TRAIN_NORMALIZER):
        super().__init__(data_root_path=data_root_path,
                         n_way_classification=n_way_classification,
                         support_size=support_size,
                         query_size=query_size,
                         split=split,
                         num_tasks=num_tasks,
                         regenerate_task_support_set=regenerate_task_support_set,
                         regenerate_task_query_set=regenerate_task_query_set,
                         seed=seed,
                         normalizer=normalizer)
        # check dataset configuration
        self.dataset_layout = dataset_layout
        assert self.split in OmniglotDataset.dataset_layout_split_options[
            self.
            dataset_layout], f'Split `{self.split}` not available for dataset_layout `{OmniglotDataset.dataset_layout_split_options[self.dataset_layout]}`'

        self.dataset_path = self._data_root_path / OmniglotDataset.dataset_folder_name
        assert self.dataset_path.exists(), f'No omniglot dataset/directory found at `{self._data_root_path}`!'
        # check dataset toplevel folders
        toplevel_dirs = [d.stem for d in self.dataset_path.iterdir() if d.is_dir()]
        assert set(OmniglotDataset.toplevel_folders_n_alphabets.keys()).issubset(
            toplevel_dirs), f'One or both toplevel folders of Omniglot are missing!'

        # load data into memory
        self._alphabets: Dict[str, List[str]] = None
        self._data = self._load_data(
            self.split)  # TODO: do this with https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python
        # pre-generate some tasks which are accessed via get task to ensure deterministic behavior
        self.create_pregen_tasks()

    def _load_data(self, split: str) -> Dict[str, np.ndarray]:
        self.__check_dataset()
        # load relevant alphabets
        data_folder = OmniglotDataset.dataset_split_toplevel_folders[split]
        alphabets = [a.stem for a in (self.dataset_path / data_folder).iterdir()]

        if self.dataset_layout == 'metadataset':
            if split == 'train':
                alphabets = list(set(alphabets) - set(OmniglotDataset.metadataset_validation_alphabets))
            elif split == 'val':
                alphabets = OmniglotDataset.metadataset_validation_alphabets

        # load classes and data from the alphabets
        data_dict, alphabets_dict = {}, {}
        for alphabet in tqdm(alphabets, file=sys.stdout, desc='Loading Omniglot Alphabets'):
            for character in sorted((self.dataset_path / data_folder / alphabet).iterdir()):
                # add character to alphabets_dict
                if not alphabet in alphabets_dict:
                    alphabets_dict[alphabet] = [character.stem]
                else:
                    alphabets_dict[alphabet].append(character.stem)

                class_name = CLASS_NAME_TEMPLATE.format(alphabet=alphabet, character=character.stem)
                # load characters into data dict
                data_dict[class_name] = self.__load_characters_from_alphabet(character_folder=character)

        self._alphabets = alphabets_dict
        data_dict = dict(sorted(data_dict.items()))
        return data_dict

    def __check_dataset(self) -> None:
        # load all available alphabets
        alphabets_in_folder = {
            tlf: [a.stem for a in (self.dataset_path / tlf).iterdir()
                 ] for tlf in OmniglotDataset.toplevel_folders_n_alphabets.keys()
        }
        # check number of alphabets in folders
        for folder, n_alphabets in OmniglotDataset.toplevel_folders_n_alphabets.items():
            assert len(
                alphabets_in_folder[folder]
            ) == n_alphabets, f'Folder `{folder}` contains {len(alphabets_in_folder[folder])}, but expected {n_alphabets}.'

    def __load_characters_from_alphabet(self, character_folder: Path) -> np.ndarray:
        images_for_character = []
        for character_image_file in character_folder.iterdir():
            img = Image.open(character_image_file).resize(OmniglotDataset.output_img_size)
            img = np.asarray(img, dtype=np.int8)
            # add channel dimension
            img = np.expand_dims(img, axis=0)
            images_for_character.append(img)

        assert len(
            images_for_character
        ) == OmniglotDataset.images_per_character, f'Number of images ({len(images_for_character)}) for character `{character_folder}` does not match {OmniglotDataset.images_per_character}.'

        return np.array(images_for_character)