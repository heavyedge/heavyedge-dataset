.. HeavyEdge-Dataset documentation master file, created by
   sphinx-quickstart on Tue Jul  8 16:03:04 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*******************************
HeavyEdge-Dataset documentation
*******************************

.. module:: heavyedge_dataset

HeavyEdge-Dataset is a Python package providing custom PyTorch datasets for loading edge profiles in various ways.

Refer to `PyTorch tutorial <tutorial>`_ for information about custom dataset, and `HeavyEdge-Landmarks document <landmarks>`_ for details on landmarks.

.. _tutorial: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
.. _landmarks: https://heavyedge-landmarks.readthedocs.io/en/latest/

.. note::

    To run examples in this document, install the package with ``doc`` optional dependency::

        pip install heavyedge-dataset[doc]

=========
Tutorials
=========

This section provides basic tutorials for beginners.

Constructing dataset
====================

Datasets access profile data from hdf5 file, wrapped by :class:`heavyedge.ProfileData` class.
It is recommended to use the context manager for file handling.
In this tutorial, we use preprocessed data distributed by the :mod:`heavyedge` package.

In the example below, we get the entire data using :class:`ProfileDataset`.
The dataset gives edge profiles and the length of coated regions.
Passing `m=1` gets only the y coordinates.

.. plot::
    :context: reset

    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_dataset import ProfileDataset
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
    ...     profiles, lengths = ProfileDataset(file, m=1)[:]
    >>> profiles.shape
    (35, 1, 3200)
    >>> lengths.shape
    (35,)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*profiles.transpose(1, 2, 0))

Passing `m=2` gets x and y coordinates.

.. plot::
    :context: close-figs

    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
    ...     profiles, lengths = ProfileDataset(file, m=2)[:]
    >>> profiles.shape
    (35, 2, 3200)
    >>> lengths.shape
    (35,)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*profiles.transpose(1, 2, 0))

:class:`PseudoLandmarkDataset` locates pseudo-landmarks from profiles.
Use `k` parameter to control the number of landmarks to sample.

.. plot::
    :context: close-figs

    >>> from heavyedge_dataset import PseudoLandmarkDataset
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
    ...     landmarks = PseudoLandmarkDataset(file, m=2, k=10)[:]
    >>> landmarks.shape
    (35, 2, 10)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*profiles.transpose(1, 2, 0), color="gray")
    ... plt.plot(*landmarks.transpose(1, 2, 0))

:class:`MathematicalLandmarkDataset` detects mathematical landmarks and average plateau heights from profiles.
Landmark detection requires `sigma` parameter for the level of smoothing of profiles.

.. plot::
    :context: close-figs

    >>> from heavyedge_dataset import MathematicalLandmarkDataset
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
    ...     landmarks, heights = MathematicalLandmarkDataset(file, m=2, sigma=32)[:]
    >>> landmarks.shape
    (35, 2, 5)
    >>> heights.shape
    (35,)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*profiles.transpose(1, 2, 0), color="gray")
    ... plt.plot(*landmarks.transpose(1, 2, 0))
    ... for h in heights:
    ...     plt.axhline(h, ls="--", alpha=0.1)

Data indexing
=============

In the previous examples, full data were accessed by slicing.
Indices and steps can also be specified.

>>> from heavyedge import get_sample_path, ProfileData
>>> from heavyedge_dataset import ProfileDataset
>>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
...     profiles, lengths = ProfileDataset(file, m=2)[2:9:3]
>>> profiles.shape
(3, 2, 3200)

You can also use list indexing in any arbitrary order.

>>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
...     profiles, lengths = ProfileDataset(file, m=2)[[2, 1, 0]]
>>> profiles.shape
(3, 2, 3200)

When a single index is specified, the result is squeezed.

>>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
...     profile, length = ProfileDataset(file, m=2)[0]
>>> profile.shape
(2, 3200)
>>> length.shape
()

Data transformation
===================

Pass a callable to ``transform`` parameter to modify the data.
In this example, :func:`heavyedge_landmarks.minmax` is used for within-sample minmax scaling of each profile.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import minmax
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
    ...     landmarks = PseudoLandmarkDataset(file, m=2, k=10, transform=minmax)[:]
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*landmarks.transpose(1, 2, 0))

Data loading
============

You can use :class:`torch.utils.data.DataLoader` for batched loading.
Note that because the default `collate_fn` transforms the loaded data to :class:`torch.Tensor`, datasets which return tuple require `collate_fn` parameter to be set.

>>> import torch
>>> from torch.utils.data import DataLoader
>>> with ProfileData(get_sample_path("Prep-Type3.h5")) as file:
...     dataset = ProfileDataset(file, m=2)
...     loader = DataLoader(dataset, batch_size=5, collate_fn=lambda x: tuple(map(torch.from_numpy, x)))
...     profiles, lengths = next(iter(loader))
>>> profiles.shape
torch.Size([5, 2, 3200])
>>> lengths.shape
torch.Size([5])

==========
Module API
==========

Profile data
============

Loads full profile data.

.. autoclass:: heavyedge_dataset.ProfileDataset
    :members:

Landmark data
=============

Loads landmark data representing profiles.

.. autoclass:: heavyedge_dataset.PseudoLandmarkDataset
    :members:

.. autoclass:: heavyedge_dataset.MathematicalLandmarkDataset
    :members:
