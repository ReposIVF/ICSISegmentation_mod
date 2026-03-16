import numpy as np


def yeo_johnson_transform(x, lmbda):
    pos = x >= 0
    if lmbda != 2:
        x[pos] = ((x[pos] + 1) ** lmbda - 1) / lmbda if lmbda != 0 else np.log(x[pos] + 1)
    if lmbda != 0:
        x[~pos] = -((-x[~pos] + 1) ** (2 - lmbda) - 1) / (2 - lmbda) if lmbda != 2 else -np.log(-x[~pos] + 1)
    return x


def standardize(x, mean, scale):
    return (x - mean) / scale
