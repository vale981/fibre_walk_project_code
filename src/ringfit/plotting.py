from . import data
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import make_interp_spline
import numpy as np

from plot_utils import *


def fancy_error(x, y, err, ax, **kwargs):
    line = ax.plot(
        x,
        y,
        **kwargs,
    )

    err = ax.fill_between(
        x,
        y + err,
        y - err,
        color=lighten_color(line[0].get_color(), 0.5),
        alpha=0.5,
    )

    return line, err


def plot_scan(
    data: data.ScanData,
    laser=False,
    output=True,
    steps: bool | int = False,
    normalize=False,
    smoothe_output: bool | int = False,
    ax=None,
    **kwargs,
):
    if not (laser or output):
        raise ValueError("At least one of 'laser' or 'output' must be True.")

    time, output_data, laser_data = (
        (data.time, data.output, data.laser)
        if isinstance(steps, bool)
        else data.for_step(steps)
    )

    if laser:
        if normalize:
            laser_data = (laser_data - laser_data.min()) / (
                laser_data.max() - laser_data.min()
            )

        lines = ax.plot(time, laser_data, **kwargs)

    if output:
        if normalize:
            output_data = (output_data - output_data.min()) / (
                output_data.max() - output_data.min()
            )

        if smoothe_output:
            if not isinstance(smoothe_output, int):
                smoothe_output = 60

            window = len(output_data) // smoothe_output
            output_data = uniform_filter1d(output_data, window)

        lines = ax.plot(time, output_data, **kwargs)

    if isinstance(steps, bool) and steps:
        peaks = data.laser_steps()
        peaks = [0, *peaks, len(data.time) - 1]

        vertical = output_data.min()
        for peak in peaks[1:-1]:
            ax.axvline(
                data.time[peak],
                color=lighten_color(lines[0].get_color()),
                linestyle="--",
                zorder=-10,
            )

        for i, (begin, end) in enumerate(zip(peaks[:-1], peaks[1:])):
            ax.text(
                (data.time[begin] + data.time[end]) / 2,
                vertical,
                f"{i}",
                ha="center",
                va="bottom",
            )


@wrap_plot
def plot_transmission(data: data.ScanData, timepoints=1000, ax=None, **kwargs):
    amplitude = data.output_end_averages()
    times = data.laser_step_times()

    smoothed = make_interp_spline(times, amplitude, k=3)
    smooth_times = np.linspace(times.min(), times.max(), timepoints)

    lines = ax.plot(smooth_times, smoothed(smooth_times), **kwargs)
    plt.plot(
        times, amplitude, "o", color=lighten_color(lines[0].get_color()), alpha=0.5
    )
