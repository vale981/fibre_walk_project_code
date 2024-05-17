###############################################################################
#                               Loading the Data                              #
###############################################################################

import numpy as np
from pathlib import Path
import functools
from . import utils


class ScanData:
    def __init__(
        self,
        laser: np.ndarray,
        output: np.ndarray,
        time: np.ndarray,
        truncation: [float, float] = [0, 100],
        sparcity: float = 1,
    ):
        """
        A class to hold the data from an oscilloscope scan where the
        laser frequency is stepped as per the modulation ``laser``.
        The output intensity ``intensity`` is proportional to the
        actual intensity measured by the oscilloscope.

        :param laser: The laser modulation signal.
        :param output: The output intensity signal.
        :param time: The time axis for the signals.
        :param truncation: The fraction of the signals to truncate
            from the beginning and end.
        :param sparcity: The fraction of the signals to keep.
        """

        if len(laser) != len(output) or len(laser) != len(time):
            raise ValueError("The signals must all be the same length.")

        length = len(laser)
        begin = int(truncation[0] * length / 100)
        end = int(truncation[1] * length / 100)

        every = int(1 / sparcity)

        self._laser = laser[begin:end:every]
        self._output = output[begin:end:every]
        self._time = time[begin:end:every]

    @classmethod
    def from_dir(cls, directory: str | Path, **kwargs):
        """Load and parse the oscilloscope data from the
        ``directory``.  The ``**kwargs`` are passed to the
        constructor.

        The directory should contain ``signal_laser.npy``,
        ``signal_outp.npy``, and ``time.npy``.
        """

        directory = Path(directory)
        laser = np.load(directory / "signal_laser.npy")
        output = np.load(directory / "signal_outp.npy")
        time = np.load(directory / "time.npy")

        return cls(laser, output, time, **kwargs)

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
    def laser_steps(self, *args, **kwargs) -> np.ndarray:
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
    def output_end_averages(
        self, end_fraction: float = 0.1, steps: np.ndarray | None = None
    ) -> np.ndarray:
        """
        The average output intensity at the end of each laser
        frequency modulation step.  **If** the system is in its
        steady, then this will be proportional to the steady state
        transmission at that frequency.
        """

        if steps is None:
            steps = self.laser_steps()

        step_size = np.mean(np.diff(steps))

        if end_fraction > 1:
            raise ValueError("end_fraction must be between 0 and 1.")

        window = int(step_size * end_fraction)

        return np.array([self._output[step - window : step].mean() for step in steps])

    def for_step(self, step: int, steps: np.ndarray | None = None):
        """Return time and output for the ``step``.  If ``steps`` is
        not provided, then they are retrieved using the default
        settings.  See :any:`laser_steps`.
        """
        time_steps: np.ndarray = self.laser_steps() if steps is None else steps

        if step < 0 or step >= len(time_steps):
            raise ValueError("The step must be between 0 and the number of steps.")

        padded_steps = [0, *time_steps, len(self._output) - 1]
        return (
            self._time[padded_steps[step] : padded_steps[step + 1]],
            self._output[padded_steps[step] : padded_steps[step + 1]],
            self._laser[padded_steps[step] : padded_steps[step + 1]],
        )
