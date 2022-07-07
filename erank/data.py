from typing import Tuple
import torch
import torch.utils.data as data


def random_split_train_tasks(dataset: data.Dataset, num_train_tasks: int = 1, train_task_idx: int = 0,
                             train_val_split: float = 0.8, seed: int = 0, num_subsplit_tasks: int = 0,
                             subsplit_first_n_train_tasks: int = 0) -> Tuple[data.Dataset, data.Dataset]:
    """Splits a dataset into different (sample-wise) training tasks. 
    Each training task has different set of data samples. Validation set is same for every task.

    Args:
        dataset (data.Dataset): The dataset to split. 
        num_train_tasks (int, optional): Number of training tasks to split. Defaults to 1.
        train_task_idx (int, optional): The current training task. Defaults to 0.
        train_val_split (float, optional): Fraction of train/val samples. Defaults to 0.8.
        seed (int, optional): The seed. Defaults to 0.

    Returns:
        Tuple[data.Dataset, data.Dataset]: train dataset, val dataset
    """
    assert train_task_idx >= 0 and train_task_idx < (
        num_train_tasks - subsplit_first_n_train_tasks) + num_subsplit_tasks, 'Invalid train_task_idx given.'

    n_train_samples = int(train_val_split * len(dataset))

    n_samples_per_task = int(n_train_samples / num_train_tasks)

    train_split_lengths = num_train_tasks * [n_samples_per_task]

    # make sure that sum of all splits equal total number of samples in dataset
    # n_val_samples can be greater than specified by train_val_split
    n_val_samples = len(dataset) - torch.tensor(train_split_lengths).sum().item()

    split_lengths = num_train_tasks * [n_samples_per_task] + [n_val_samples]
    data_splits = data.random_split(dataset, split_lengths, generator=torch.Generator().manual_seed(seed))

    if num_subsplit_tasks > 0:
        # further split first ´subsplit_first_n_train_tasks´ into `num_subsplit_tasks`
        subsplit_dataset = data.ConcatDataset(data_splits[:subsplit_first_n_train_tasks])
        # remove first n train tasks idxs from data split list
        data_splits = data_splits[subsplit_first_n_train_tasks:]
        n_samples_per_subsplit = int(len(subsplit_dataset) / num_subsplit_tasks)

        subsplit_lengths = num_subsplit_tasks * [n_samples_per_subsplit]
        # distribute remaining samples (due to rounding) from beginning
        samples_remaining = len(subsplit_dataset) - sum(subsplit_lengths)
        for i in range(len(subsplit_lengths)):
            if samples_remaining <= 0:
                break
            subsplit_lengths[i] += 1
            samples_remaining -= 1

        assert sum(subsplit_lengths) == len(subsplit_dataset)

        data_subsplits = data.random_split(subsplit_dataset, subsplit_lengths,
                                           generator=torch.Generator().manual_seed(seed + 1))

        # concat data_splits: [subsplit sets] + train sets + val set
        data_splits = data_subsplits + data_splits

    # select train task split + val split
    return data_splits[train_task_idx], data_splits[-1]
