from . import data
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np


def wrap_plot(f):
    def wrapped(*args, ax=None, setup_function=plt.subplots, **kwargs):
        fig = None
        if not ax:
            fig, ax = setup_function()

        ret_val = f(*args, ax=ax, **kwargs)
        return (fig, ax, ret_val) if ret_val else (fig, ax)

    return wrapped


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


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


@wrap_plot
def plot_scan(
    data: data.ScanData,
    laser=False,
    output=True,
    steps=False,
    normalize=False,
    smoothe_output=False,
    ax=None,
    **kwargs,
):
    if not (laser or output):
        raise ValueError("At least one of 'laser' or 'output' must be True.")

    if laser:
        laser = data.laser
        if normalize:
            laser = (laser - laser.min()) / (laser.max() - laser.min())

        lines = ax.plot(data.time, laser, **kwargs)

    if output:
        output = data.output
        if normalize:
            output = (output - output.min()) / (output.max() - output.min())

        if smoothe_output:
            if not isinstance(smoothe_output, int):
                smoothe_output = 60

            window = len(output) // smoothe_output
            output = uniform_filter1d(output, window)

        lines = ax.plot(data.time, output, **kwargs)

    if steps:
        peaks = data.laser_steps()
        for peak in peaks:
            ax.axvline(
                data.time[peak],
                color=lighten_color(lines[0].get_color()),
                linestyle="--",
                zorder=-10,
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
