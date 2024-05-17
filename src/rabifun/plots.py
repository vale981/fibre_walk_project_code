from plot_utils import *
from .system import Params, output_signal
from .analysis import fourier_transform
import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_result(
    t: np.ndarray, signal: np.ndarray, params: Params, window=None
):
    """Plot the simulation result. The signal is plotted in the first axis
    and the Fourier transform is plotted in the second axis.

    :param t: time axis
    :param signal: output signal
    :param params: system parameters
    :param window: time window for the Fourier transform

    :returns: figure and axes
    """

    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, signal)
    ax1.set_title(f"Output signal\n {params}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Intensity")

    freq, fft = fourier_transform(t, signal, window)
    fft = fft / np.max(np.abs(fft))

    freq *= 2 * np.pi
    ax2.set_xlim(0, params.Ω * (params.N))
    ax3 = ax2.twinx()

    ax2.plot(freq, np.abs(fft) ** 2)
    ax2.set_yscale("log")
    ax2.set_title("FFT")
    ax2.set_xlabel("ω [angular]")
    ax2.set_ylabel("Power")
    ax2.legend()

    ax3.plot(freq, np.angle(fft), linestyle="--", color="C2", alpha=0.5, zorder=-10)
    ax3.set_ylabel("Phase")

    return f, (ax1, ax2)


def plot_sidebands(ax, params: Params):
    """Visualize the frequency of the sidebands.

    :param ax: axis to plot on
    :param params: system parameters
    """
    energy = params.rabi_splitting
    sidebands = (
        params.Ω - params.laser_detuning + np.array([1, -1]) * energy / 2 - params.Δ / 2
    )

    ax.axvline(params.Ω - params.Δ, color="black", label="steady state")

    for n, sideband in enumerate(sidebands):
        ax.axvline(
            sideband,
            color=f"C{n}",
            label=f"rabi-sideband {n}",
        )

    ax.legend()
