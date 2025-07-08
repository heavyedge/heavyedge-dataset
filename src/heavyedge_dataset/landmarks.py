"""
Landmark locators
-----------------

Functions to locate landmarks from edge profiles.
"""

import numpy as np
from heavyedge.api import landmarks_type3

__all__ = [
    "pseudo_landmarks_1d",
    "pseudo_landmarks_2d",
    "math_landmarks_1d",
]


def pseudo_landmarks_1d(Ys, Ls, k):
    ret = []
    for Y, L in zip(Ys, Ls):
        idxs = np.linspace(0, L - 1, k, dtype=int)
        ret.append(Y[idxs].reshape(1, -1))
    return np.array(ret)


def pseudo_landmarks_2d(x, Ys, Ls, k):
    ret = []
    for Y, L in zip(Ys, Ls):
        idxs = np.linspace(0, L - 1, k, dtype=int)
        ret.append(np.stack([x[idxs], Y[idxs]]))
    return np.array(ret)


def math_landmarks_1d(Ys, Ls, sigma):
    ret = []
    for Y, L in zip(Ys, Ls):
        Y = Y[:L]
        indices = np.flip(landmarks_type3(Y, sigma))
        y = np.concat([[np.mean(Y[: indices[0]])], Y[indices]])
        ret.append(y.reshape(1, -1))
    return np.array(ret)
