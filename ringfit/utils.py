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
