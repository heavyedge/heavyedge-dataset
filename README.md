# HeavyEdge-Dataset

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/heavyedge-dataset.svg)](https://pypi.python.org/pypi/heavyedge-dataset/)
[![PyPI Version](https://img.shields.io/pypi/v/heavyedge-dataset.svg)](https://pypi.python.org/pypi/heavyedge-dataset/)
[![License](https://img.shields.io/github/license/heavyedge/heavyedge-dataset)](https://github.com/heavyedge/heavyedge-dataset/blob/master/LICENSE)
[![CI](https://github.com/heavyedge/heavyedge-dataset/actions/workflows/ci.yml/badge.svg)](https://github.com/heavyedge/heavyedge-dataset/actions/workflows/ci.yml)
[![CD](https://github.com/heavyedge/heavyedge-dataset/actions/workflows/cd.yml/badge.svg)](https://github.com/heavyedge/heavyedge-dataset/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/heavyedge-dataset/badge/?version=latest)](https://heavyedge-dataset.readthedocs.io/en/latest/?badge=latest)

Package to load edge profile data as PyTorch dataset.

## Usage

HeavyEdge-Dataset provides dataset classes profile data file.

A simple use case to load two-dimensional coordinates of profiles and their lengths:

```python
from heavyedge import get_sample_path, ProfileData
from heavyedge_dataset import ProfileDataset
with ProfileData(get_sample_path("Prep-Type2.h5")) as file:
    data = ProfileDataset(file, 2)[:]
```

Refer to the package documentation for more information.

## Documentation

The manual can be found online:

> https://heavyedge-dataset.readthedocs.io

If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Developing

### Installation

For development features, you must install the package by `pip install -e .[dev]`.

### Testing

Run `pytest` command to perform unit test.
