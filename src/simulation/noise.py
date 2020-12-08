from opensimplex import OpenSimplex
from scipy import stats
from time import time


def noise_function(mean=0, std=1, velocity=1, seed=None):
    """
    Creates a function that generates a smoothly varying random variable with the given mean and standard deviation.
    The velocity is the rate at which the resulting function will move through the OpenSimplex space.
    """
    if seed is None:
        seed = int(time() * 1e6)

    noise_generator = OpenSimplex(seed)

    def noise(t):
        """
        Calculates a noise value for the given time.
        """
        noise_value = noise_generator.noise2d(t * velocity, 0)
        cdf_value = (noise_value + 1) / 2  # map (-1, 1) to (0, 1)
        gaussian_value = stats.norm.ppf(cdf_value, loc=mean, scale=std)
        return gaussian_value

    return noise
