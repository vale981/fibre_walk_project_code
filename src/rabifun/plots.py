from plot_utils import *
from .system import Params
from .analysis import fourier_transform
import matplotlib.pyplot as plt
import numpy as np


@wrap_plot
def plot_simulation_result(
    fig,
    t: np.ndarray,
    signal: np.ndarray,
    params: Params,
    window=None,
):
    """Plot the simulation result. The signal is plotted in the first axis
    and the Fourier transform is plotted in the second axis.

    :param t: time axis
    :param signal: output signal
    :param params: system parameters
    :param window: time window for the Fourier transform

    :returns: figure and axes
    """

    (ax1, ax2) = fig.subplots(2, 1)

    ax1.plot(t, signal)
    ax1.set_title(f"Output signal\n {params}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Intensity")

    freq, fft = fourier_transform(t, signal, window)
    fft = fft / np.max(np.abs(fft))

    ax2.set_xlim(0, params.Ω * (params.N))
    ax3 = ax2.twinx()

    ax2.plot(freq, np.abs(fft))
    # ax2.set_yscale("log")
    ax2.set_title("FFT")
    ax2.set_xlabel("ω [linear]")
    ax2.set_ylabel("Power")
    ax2.legend()

    ax3.plot(
        freq,
        np.angle(fft),
        linestyle="--",
        color="C2",
        alpha=0.5,
        zorder=-10,
    )
    ax3.set_ylabel("Phase")

    return (ax1, ax2)


def plot_sidebands(ax, params: Params):
    """Visualize the frequency of the sidebands.

    :param ax: axis to plot on
    :param params: system parameters
    """
    energy = params.rabi_splitting

    first_sidebands = np.abs(
        -(params.laser_detuning + params.measurement_detuning)
        + np.array([1, -1]) * energy / 2
        + params.Δ / 2
    )
    second_sidebands = (
        params.Ω * (1 - params.δ)
        - (params.laser_detuning + params.measurement_detuning)
        + np.array([1, -1]) * energy / 2
        - params.Δ / 2
    )

    ax.axvline(
        params.ω_eom / (2 * np.pi) - params.measurement_detuning,
        color="black",
        label="steady state",
    )

    for n, sideband in enumerate(first_sidebands):
        ax.axvline(
            sideband,
            color=f"C1",
        )

    for n, sideband in enumerate(second_sidebands):
        ax.axvline(
            sideband,
            color=f"C2",
        )

    ax.legend()
