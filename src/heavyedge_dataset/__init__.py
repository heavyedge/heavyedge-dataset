"""Package to load edge profile data using PyTorch dataset.

Refer to `PyTorch tutorial <tutorial>`_ for information about custom dataset.

.. _tutorial: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import numbers
from collections.abc import Sequence

import numpy as np
from torch.utils.data import Dataset

__all__ = [
    "ProfileDataset",
]


class ProfileDataset(Dataset):
    """Edge profile dataset.

    Loads data as a tuple of two numpy arrays:

    1. Profile data, shape: (N, m, L).
    2. Length of each profile, shape: (N,).

    N is the number of loaded data, m is dimension of coordinates, and
    L is the maximum length of profiles.

    Parameters
    ----------
    file : heavyedge.ProfileData
        Open hdf5 file.
    m : {1, 2}
        Profile data dimension.
        1 means only y coordinates, and 2 means both x and y coordinates.
    transform : callable, optional
        Optional transformation to be applied on samples.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import ProfileDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     profiles, lengths = ProfileDataset(file, m=2)[:]
    >>> profiles.shape
    (22, 2, 3200)
    >>> lengths.shape
    (22,)

    Should this dataset be used for :class:`torch.utils.data.DataLoader`,
    ``collate_fn`` argument should be passed to the data loader.

    >>> from torch.utils.data import DataLoader
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     dataset = ProfileDataset(file, m=2)
    ...     loader = DataLoader(dataset, collate_fn=lambda x: x)
    ...     profiles, lengths = next(iter(loader))
    >>> profiles.shape
    (1, 2, 3200)
    >>> lengths.shape
    (1,)

    If data should be loaded as :class:`torch.Tensor`, pass ``transform`` argument.

    >>> import torch
    >>> def to_tensor(sample):
    ...     return (torch.from_numpy(sample[0]), torch.from_numpy(sample[1]))
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     dataset = ProfileDataset(file, m=2, transform=to_tensor)
    ...     loader = DataLoader(dataset, collate_fn=lambda x: x)
    ...     profiles, lengths = next(iter(loader))
    >>> type(profiles)
    <class 'torch.Tensor'>
    """

    def __init__(self, file, m=1, transform=None):
        self.file = file
        self.m = m
        self.transform = transform
        self.x = file.x()

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            Y, L, _ = self.file[idx]
            Y = Y[np.newaxis, :]
        else:
            # Support multi-indexing
            idxs = idx
            needs_sort = isinstance(idx, (Sequence, np.ndarray))
            if needs_sort:
                # idxs must be sorted for h5py
                idxs = np.array(idxs)
                sort_idx = np.argsort(idxs)
                idxs = idxs[sort_idx]
            Y, L, _ = self.file[idxs]
            if needs_sort:
                reverse_idx = np.argsort(sort_idx)
                Y = Y[reverse_idx]
                L = L[reverse_idx]
            Y = Y[:, np.newaxis, :]
        if self.m == 1:
            pass
        elif self.m == 2:
            x = np.tile(self.x, Y.shape[:-1] + (1,))
            Y = np.concatenate([x, Y], axis=-2)
        else:
            raise ValueError(f"Unsupported dimension: {self.m} (Must be 1 or 2).")
        ret = (Y, L)
        if self.transform is not None:
            ret = self.transform(ret)
        return ret

    def __getitems__(self, idxs):
        # PyTorch API
        return self.__getitem__(idxs)
