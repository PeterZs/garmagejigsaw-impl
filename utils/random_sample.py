import numpy as np
from scipy.stats.qmc import LatinHypercube


def LatinHypercubeSample(sample_size, n_samples):
    """
    Latin Hypercube sample.
    :param sample_size:
    :param n_samples:
    :return:
    """
    sampler = LatinHypercube(d=1)  # 创建一个 Latin Hypercube 采样器，维度为1
    sample = sampler.random(n=n_samples)
    sample_idx = (sample * sample_size).astype(int).flatten()
    return sample_idx


def balancedSample(sample_size, n_samples):
    samples = np.linspace(0, sample_size - 1, n_samples)
    samples = np.round(samples).astype(int)
    return samples
