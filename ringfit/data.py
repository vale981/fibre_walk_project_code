###############################################################################
#                               Loading the Data                              #
###############################################################################

import numpy as np
from pathlib import Path
import functools
from . import utils


class ScanData:
    def __init__(self, laser: np.ndarray, output: np.ndarray, time: np.ndarray):
        """
        A class to hold the data from an oscilloscope scan where the
        laser frequency is stepped as per the modulation ``laser``.
        The output intensity ``intensity`` is proportional to the
        actual intensity measured by the oscilloscope.

        :param laser: The laser modulation signal.
        :param output: The output intensity signal.
        :param time: The time axis for the signals.
        """

        self._laser = laser
        self._output = output
        self._time = time

    @property
    def laser(self):
        """The laser modulation signal."""

        return self._laser

    @property
    def output(self):
        """The output intensity signal."""

        return self._output

    @property
    def time(self):
        """The time axis for the signals."""

        return self._time

    @functools.cache
    def laser_steps(self, *args, **kwargs):
        """Find the indices of the laser signal ``laser`` where the
        frequency of the laser changes.  For the parameters, see
        :func:`utils.find_frequency_steps`.

        The result is cached for future calls.
        """

        return utils.find_frequency_steps(self._laser, *args, **kwargs)

    @functools.cache
    def laser_step_times(self, *args, **kwargs):
        """
        The times at which the laser frequency changes.  See
        :any:`laser_steps`.
        """

        return self._time[self.laser_steps()]

    @functools.cache
    def output_end_averages(self, end_fraction: float = 0.1, *args, **kwargs):
        """
        The average output intensity at the end of each laser
        frequency modulation step.  **If** the system is in its
        steady, then this will be proportional to the steady state
        transmission at that frequency.
        """

        steps = self.laser_steps()
        step_size = np.mean(np.diff(steps))

        if end_fraction > 1:
            raise ValueError("end_fraction must be between 0 and 1.")

        window = int(step_size * end_fraction)

        return np.array([self._output[step - window : step].mean() for step in steps])


def load_scan(directory: str | Path):
    """Load and parse the oscilloscope data from the ``directory``.

    The directory should contain ``signal_laser.npy``, ``signal_outp.npy``, and ``time.npy``.
    """

    directory = Path(directory)
    laser = np.load(directory / "signal_laser.npy")
    output = np.load(directory / "signal_outp.npy")
    time = np.load(directory / "time.npy")

    return ScanData(laser, output, time)
