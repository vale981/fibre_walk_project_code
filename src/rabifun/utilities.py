import numpy as np


def add_noise(signal: np.ndarray, noise_magnitude: float = 0.1):
    """
    Add Gaussian noise to ``signal``.  The standard deviation is
    ``noise_magnitude`` relative to the mean absolute signal.

    :returns: The noisy signal.
    """

    signal_magnitude = np.abs(signal).mean()
    return signal + np.random.normal(0, signal_magnitude * noise_magnitude, len(signal))
