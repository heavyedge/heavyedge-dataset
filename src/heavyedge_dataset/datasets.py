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

__all__ = [
    "ProfileDataset",
    "PseudoLandmarkDataset",
    "MathematicalLandmarkDataset",
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

    @property
    @abc.abstractmethod
    def transform(self):
        """Optional transformation to be applied on samples.

        Returns
        -------
        Callable
        """

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            Y, L, _ = self.file[idx]
            Ys, Ls = [Y], [L]
        else:
            # Support multi-indexing
            idxs = idx
            needs_sort = isinstance(idx, (Sequence, np.ndarray))
            if needs_sort:
                # idxs must be sorted for h5py
                idxs = np.array(idxs)
                sort_idx = np.argsort(idxs)
                idxs = idxs[sort_idx]
            Ys, Ls, _ = self.file[idxs]
            if needs_sort:
                reverse_idx = np.argsort(sort_idx)
                Ys = Ys[reverse_idx]
                Ls = Ls[reverse_idx]

        ret = self.default_transform(Ys, Ls)
        if self.transform:
            ret = self.transform(ret)
        return ret

    def __getitems__(self, idxs):
        # PyTorch API
        return self.__getitem__(idxs)

    @abc.abstractmethod
    def default_transform(self, profiles, lengths):
        """Default data transformation.

        Subclass must implement this method to transform profile data into target data.

        Parameters
        ----------
        profiles : (N, M) array
            Profile data.
        lengths : (N,) array
            Length of each profile in *profiles*.
        """
        pass


class ProfileDataset(ProfileDatasetBase, Dataset):
    """Full profile dataset in 1-D or 2-D.

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

    @property
    def transform(self):
        return self._transform

    def default_transform(self, profiles, lengths):
        if self.m == 1:
            ret = [Y[:L].reshape(1, -1) for Y, L in zip(profiles, lengths)]
        elif self.m == 2:
            ret = [np.stack([self.x[:L], Y[:L]]) for Y, L in zip(profiles, lengths)]
        else:
            raise ValueError(f"Invalid dimension: {self.m}")
        return ret


class PseudoLandmarkDataset(ProfileDatasetBase, Dataset):
    """Pseudo-landmark dataset in 1-D or 2-D.

    Pseudo-landmarks are points that are equidistantly sampled.

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
        Optional transformation to be applied on samples.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import PseudoLandmarkDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     data = PseudoLandmarkDataset(file, 10, 2)[:]
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

    @property
    def transform(self):
        return self._transform

    def default_transform(self, profiles, lengths):
        ret = []
        if self.m == 1:
            for Y, L in zip(profiles, lengths):
                idxs = np.linspace(0, L - 1, self.k, dtype=int)
                ret.append(Y[idxs].reshape(1, -1))
        elif self.m == 2:
            for Y, L in zip(profiles, lengths):
                idxs = np.linspace(0, L - 1, self.k, dtype=int)
                ret.append(np.stack([self.x[idxs], Y[idxs]]))
        else:
            raise ValueError(f"Invalid dimension: {self.m}")
        return np.array(ret)


class MathematicalLandmarkDataset(ProfileDatasetBase, Dataset):
    """Mathematical landmark dataset in 1-D.

    Mathematical landmarks are points which are choosed by their
    mathematical properties, i.e., slope or curvature.

    Parameters
    ----------
    file : heavyedge.ProfileData
        Open hdf5 file.
    sigma : scalar
        Standard deviation of Gaussian kernel for landmark detection.
    transform : callable, optional
        Optional transformation to be applied on samples.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import MathematicalLandmarkDataset
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    ...     data = MathematicalLandmarkDataset(file, 32)[:]
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

    @property
    def transform(self):
        return self._transform

    def default_transform(self, profiles, lengths):
        ret = []
        for Y, L in zip(profiles, lengths):
            Y = Y[:L]
            indices = np.flip(landmarks_type3(Y, self.sigma))
            y = np.concat([[np.mean(Y[: indices[0]])], Y[indices]])
            ret.append(y.reshape(1, -1))
        return np.array(ret)
