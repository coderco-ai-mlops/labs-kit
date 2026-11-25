"""torch.utils.data for the practice version of PyTorch: datasets and the loader that hands them out in batches."""

import numpy as np

from .._tensor import Tensor, _rng, float64, from_numpy, missing, stack, tensor


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if any(len(t) != len(tensors[0]) for t in tensors):
            raise AssertionError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    order, parts = _rng[0].permutation(len(dataset)).tolist(), []
    for length in lengths:
        parts.append(Subset(dataset, order[:length]))
        order = order[length:]
    return parts


class SequentialSampler:
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(SequentialSampler):
    def __iter__(self):
        return iter(_rng[0].permutation(len(self.data_source)).tolist())


def default_collate(items):
    """Join single items into one batch, the way PyTorch does: a pair of tensors becomes a pair of stacked tensors."""
    first = items[0]
    if isinstance(first, Tensor):
        return stack(items)
    if isinstance(first, np.ndarray):
        return stack([from_numpy(item) for item in items])
    if isinstance(first, float):
        return tensor(items, dtype=float64)
    if isinstance(first, int):
        return tensor(items)
    if isinstance(first, dict):
        return {key: default_collate([item[key] for item in items]) for key in first}
    if isinstance(first, (tuple, list)):
        return [default_collate(list(column)) for column in zip(*items)]
    return items


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=None, **options):
        self.dataset, self.batch_size, self.drop_last = dataset, batch_size, drop_last
        self.collate_fn = collate_fn or default_collate
        self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def __len__(self):
        whole, rest = divmod(len(self.dataset), self.batch_size)
        return whole + (1 if rest and not self.drop_last else 0)

    def __iter__(self):
        order = list(self.sampler)
        for start in range(0, len(order), self.batch_size):
            chosen = order[start:start + self.batch_size]
            if len(chosen) == self.batch_size or not self.drop_last:
                yield self.collate_fn([self.dataset[i] for i in chosen])

__path__ = []   # so that `from ... import` of a name that is not here reaches the plain sentence too


def __getattr__(name):
    raise missing("torch.utils.data", name, globals())
