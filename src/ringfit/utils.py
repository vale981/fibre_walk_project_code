###############################################################################
#                                  Utilities                                  #
###############################################################################

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


def find_frequency_steps(laser: np.ndarray, window_fraction: int = 60) -> np.ndarray:
    """Find the indices of the laser signal ``laser`` where the
    frequency of the laser changes. The parameter ``window_fraction``
    is the fraction of the signal length that is used as the window
    size for the sliding average.
    """

    window = len(laser) // window_fraction
    if window % 2 != 0:
        window += 1

    sliding_average = uniform_filter1d(laser, window)
    left_averages = np.pad(
        sliding_average[:-window], (window // 2, window // 2), mode="edge"
    )
    right_averages = np.pad(
        sliding_average[window:], (window // 2, window // 2), mode="edge"
    )

    step_diffs = np.abs(left_averages - right_averages)
    step_diffs /= step_diffs.max()

    peaks = find_peaks(step_diffs, height=0.5, distance=window)[0]
    return peaks


def shift_and_normalize(array: np.ndarray) -> np.ndarray:
    shifted = array - array.min()
    return shifted / abs(shifted).max()


def smoothe_signal(
    signal: np.ndarray, window_size: float = 0.01, time_step: float = 1
) -> np.ndarray:
    """Smoothe the signal ``signal`` using a uniform filter with a window
    size of ``window_size / time_step``."""

    window = int(window_size / time_step)
    return uniform_filter1d(signal, window)


class WelfordAggregator:
    """A class to aggregate values using the Welford algorithm.

    The Welford algorithm is an online algorithm to calculate the mean
    and variance of a series of values.

    The aggregator keeps track of the number of samples the mean and
    the variance.  Aggregation of identical values is prevented by
    checking the sample index.  Tracking can be disabled by setting
    the initial index to ``None``.

    See also the `Wikipedia article
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm>`_.

    :param first_value: The first value to aggregate.
    """

    __slots__ = ["n", "mean", "_m_2"]

    def __init__(self, first_value: np.ndarray):
        self.n: int = 1
        self.mean: np.ndarray = first_value
        self._m_2 = np.zeros_like(first_value)

    def update(self, new_value: np.ndarray):
        """Updates the aggregator with a new value."""

        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self._m_2 += np.abs(delta) * np.abs(delta2)

    @property
    def sample_variance(self) -> np.ndarray:
        """
        The empirical sample variance.  (:math:`\sqrt{N-1}`
        normalization.)
        """

        if self.n == 1:
            return np.zeros_like(self.mean)

        return self._m_2 / (self.n - 1)

    @property
    def ensemble_variance(self) -> np.ndarray:
        """The ensemble variance."""
        return self.sample_variance / self.n

    @property
    def ensemble_std(self) -> np.ndarray:
        """The ensemble standard deviation."""
        return np.sqrt(self.ensemble_variance)
