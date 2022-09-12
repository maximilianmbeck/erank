from typing import Dict, List, Union
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from erank.data.basemetaclassificationdataset import BaseMetaClassificationDataset

MINI_IMAGENET_TRAIN_CL_NORMALIZER = {
    'mean': [120.1979412172855, 114.82953431329128, 102.99532529569989],
    'std': [55.8573964877539, 54.94042821584164, 55.151333856338944],
    'num_dataset_samples': 38400
}


class MiniImagenetDataset(BaseMetaClassificationDataset):
    """Mini-Imagenet Dataset"""

    name = 'miniImagenet'

    dataset_folder_name = 'miniImagenet'
    splits_num_classes = {'train': 64, 'val': 16, 'test': 20}

    output_img_size = (84, 84)
    normalizer = None

    def __init__(self,
                 data_root_path: Union[str, Path],
                 n_way_classification: int,
                 support_size: int,
                 query_size: int,
                 split: str,
                 num_tasks: int = 0,
                 regenerate_task_support_set: bool = True,
                 regenerate_task_query_set: bool = True,
                 seed: int = 0,
                 normalizer: Dict[str, List[float]] = MINI_IMAGENET_TRAIN_CL_NORMALIZER):
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

        self.dataset_path = self._data_root_path / MiniImagenetDataset.dataset_folder_name
        assert self.dataset_path.exists(), f'No miniImagenet dataset/directory found at `{self._data_root_path}`!'
        # check dataset toplevel folders
        toplevel_dirs = [d.stem for d in self.dataset_path.iterdir() if d.is_dir()]
        assert set(MiniImagenetDataset.splits_num_classes.keys()).issubset(
            toplevel_dirs), f'One or more toplevel folders for miniImagenet are missing!'

        # load data into memory
        self._data = self._load_data(self.split)

        # pre-generate some tasks which are accessed via get_task() to ensure deterministic behavior
        self.pregen_tasks, self.pregen_task_name_to_index = self.create_pregen_tasks()

    def _load_data(self, split: str) -> Dict[str, np.ndarray]:
        # TODO use .csv files to for indexing whole dataset at once (all classes)
        split_dir = self.dataset_path / split

        data = {}
        for class_folder in tqdm(sorted(split_dir.iterdir()), file=sys.stdout, desc='Loading MiniImagenet classes'):
            data[class_folder.stem] = self.__load_images_for_class(class_folder)
        assert len(list(data.keys())) == MiniImagenetDataset.splits_num_classes[
            split], f'Number of classes ({len(list(data.keys()))}) for split `{split}` does not match predefined number of classes ({MiniImagenetDataset.splits_num_classes[split]})!'
        return data

    def __load_images_for_class(self, class_folder: Path) -> np.ndarray:
        images_for_class = []

        for image_file in class_folder.iterdir():
            img = Image.open(image_file).resize(MiniImagenetDataset.output_img_size)
            img = np.asarray(img, dtype=np.uint8)
            # convert to shape CxHxW (channel first)
            img = img.transpose(2, 0, 1)
            images_for_class.append(img)

        return np.array(images_for_class)