from opensimplex import OpenSimplex
from scipy import stats
import numpy as np
from time import time


def smooth_gaussian_noise_function(means=np.array([0]), variances=np.array([[1]]), velocity=1, seed=3):
    """
    Creates a function that generates a smoothly varying random variable with the given mean and standard deviation.
    The velocity is the rate at which the resulting function will move through the OpenSimplex space.
    """
    if seed is None:
        seed = int(time() * 1e6)

    noise_generators = np.array([OpenSimplex(seed * (i + 1)) for i in range(len(means))])

    def noise(t):
        """
        Calculates a noise value for the given time.
        """
        noise_values = np.array([noise_generator.noise2d(t * velocity, 0) for noise_generator in noise_generators])
        cdf_values = (noise_values + 1) / 2  # map (-1, 1) to (0, 1)
        # TODO: properly use covariance matrix instead of just diagonal
        gaussian_values = np.array([stats.norm.ppf(cdf_value, loc=mean, scale=np.sqrt(variance))
                                    for cdf_value, mean, variance in zip(cdf_values, means, np.diagonal(variances))])
        return gaussian_values

    return noise


def gaussian_noise_function(means=np.array([0]), variances=np.array([[1]])):
    """
    Creates a function that generates a random variable with the given mean and standard deviation.
    """
    def noise(_):
        # TODO: properly use covariance matrix instead of just diagonal
        return np.array([np.random.normal(mean, np.sqrt(variance))
                         for mean, variance in zip(means, np.diagonal(variances))])

    return noise
