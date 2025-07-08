"""
Dataset classes
---------------

PyTorch dataset classes for edge profiles.
"""

import abc
import numbers
from collections.abc import Sequence

import numpy as np
from heavyedge.api import landmarks_type3
from torch.utils.data import Dataset

from .landmarks import math_landmarks_1d, pseudo_landmarks_1d, pseudo_landmarks_2d

__all__ = [
    "ProfileDataset",
    "PseudoLmDataset",
    "MathLm1dDataset",
    "MathLm2dDataset",
]


class ProfileDatasetBase(abc.ABC):
    """Abstract base class for profile dataset."""

    @property
    @abc.abstractmethod
    def file(self):
        """Profile data file.

        Returns
        -------
        heavyedge.ProfileData
        """

    @abc.abstractmethod
    def default_transform(self, profiles, lengths):
        """Default transform by the dataset class.

        Parameters
        ----------
        profiles : (N, M) array
            Profile data.
        lengths : (N,) array
            Length of each profile.
        """
        pass

    @property
    @abc.abstractmethod
    def transform(self):
        """Optional transformation passed to the dataset instance.

        Returns
        -------
        Callable
        """

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            Y, L, _ = self.file[idx]
            ret = self.default_transform([Y], [L])
            if self.transform:
                ret = self.transform(ret)
            ret = ret[0]
        else:
            ret = self.__getitems__(idx)
        return ret

    def __getitems__(self, idxs):
        # PyTorch API
        needs_sort = isinstance(idxs, (Sequence, np.ndarray))
        if needs_sort:
            # idxs must be sorted for h5py
            idxs = np.array(idxs)
            sort_idx = np.argsort(idxs)
            idxs = idxs[sort_idx]
        else:
            pass
        Ys, Ls, _ = self.file[idxs]
        if needs_sort:
            reverse_idx = np.argsort(sort_idx)
            Ys = Ys[reverse_idx]
            Ls = Ls[reverse_idx]
        ret = self.default_transform(Ys, Ls)
        if self.transform:
            ret = self.transform(ret)
        return ret


class ProfileDataset(ProfileDatasetBase, Dataset):
    """Full profile dataset.

    Parameters
    ----------
    file : heavyedge.ProfileData
        Open hdf5 file.
    m : {1, 2}
        Profile data dimension.
        1 means only y coordinates, and 2 means both x and y coordinates.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import ProfileDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     data = ProfileDataset(file, 2)[:]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... for coords in data:
    ...     plt.plot(*coords, color="gray")
    """

    def __init__(self, file, m, transform=None):
        self._file = file
        self.m = m
        self._transform = transform

        self.x = file.x()

    @property
    def file(self):
        return self._file

    def default_transform(self, profiles, lengths):
        if self.m == 1:
            ret = [Y[:L].reshape(1, -1) for Y, L in zip(profiles, lengths)]
        elif self.m == 2:
            ret = [np.stack([self.x[:L], Y[:L]]) for Y, L in zip(profiles, lengths)]
        else:
            raise ValueError(f"Invalid dimension: {self.m}")
        return ret

    @property
    def transform(self):
        return self._transform


class PseudoLmDataset(ProfileDatasetBase, Dataset):
    """Pseudo-landmark dataset.

    Parameters
    ----------
    file : heavyedge.ProfileData
        Open hdf5 file.
    k : int
        Number of landmarks to sample.
    m : {1, 2}
        Profile data dimension.
        1 means only y coordinates, and 2 means both x and y coordinates.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import PseudoLmDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     data = PseudoLmDataset(file, 10, 2)[:]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*data.transpose(1, 2, 0), color="gray")
    """

    def __init__(self, file, k, m, transform=None):
        self._file = file
        self.k = k
        self.m = m
        self._transform = transform

        self.x = file.x()

    @property
    def file(self):
        return self._file

    def default_transform(self, profiles, lengths):
        if self.m == 1:
            ret = pseudo_landmarks_1d(profiles, lengths, self.k)
        elif self.m == 2:
            ret = pseudo_landmarks_2d(self.x, profiles, lengths, self.k)
        else:
            raise ValueError(f"Invalid dimension: {self.m}")
        return ret

    @property
    def transform(self):
        return self._transform


class MathLm1dDataset(ProfileDatasetBase, Dataset):
    """1-D mathematical landmarks dataset.

    Parameters
    ----------
    file : heavyedge.ProfileData
        Open hdf5 file.
    sigma : scalar
        Standard deviation of Gaussian kernel for landmark detection.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import MathLm1dDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     data = MathLm1dDataset(file, 32)[:]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*data.transpose(1, 2, 0), color="gray")
    """

    def __init__(self, file, sigma, transform=None):
        self._file = file
        self.sigma = sigma
        self._transform = transform

    @property
    def file(self):
        return self._file

    def default_transform(self, profiles, lengths):
        return math_landmarks_1d(profiles, lengths, self.sigma)

    @property
    def transform(self):
        return self._transform


class MathLm2dDataset(ProfileDatasetBase, Dataset):
    """2-D mathematical landmarks dataset.

    Parameters
    ----------
    file : heavyedge.ProfileData
        Open hdf5 file.
    sigma : scalar
        Standard deviation of Gaussian kernel for landmark detection.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import MathLm2dDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     lm, _ = MathLm2dDataset(file, 32)[:]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*lm.transpose(1, 2, 0), color="gray")
    """

    def __init__(self, file, sigma, transform=None):
        self._file = file
        self.sigma = sigma
        self._transform = transform

        self.x = file.x()

    @property
    def file(self):
        return self._file

    def default_transform(self, profiles, lengths):
        # Todo: cythonize this method to avoid python loop.
        # This will require cythonizing landmarks_type3().
        lm, center_height = [], []
        for Y, L in zip(profiles, lengths):
            Y = Y[:L]
            indices = np.flip(landmarks_type3(Y, self.sigma))
            lm.append(np.stack([self.x[indices], Y[indices]]))
            center_height.append(np.mean(Y[: indices[0]]))
        return np.array(lm), np.array(center_height)

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            Y, L, _ = self.file[idx]
            lm, ch = self.default_transform([Y], [L])
            if self.transform:
                lm, ch = self.transform(lm, ch)
            lm, ch = lm[0], ch[0]
        else:
            lm, ch = self.__getitems__(idx)
        return (lm, ch)

    @property
    def transform(self):
        return self._transform
