from networkx import is_aperiodic
from plot_utils import *
from .system import (
    Params,
    RuntimeParams,
    solve,
    coupled_mode_indices,
    mode_name,
    uncoupled_mode_indices,
    correct_for_decay,
)
from .analysis import fourier_transform, RingdownPeakData, RingdownParams, lorentzian
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

    ax2.plot(freq, np.abs(fft) ** 2)
    # ax2.set_yscale("log")
    ax2.set_title("FFT")
    ax2.set_xlabel("ω [linear]")
    ax2.set_ylabel("Power")
    ax2.legend()

    ax3.plot(
        freq[:-1],
        np.cumsum(np.angle(fft[1:] / fft[:-1])),
        linestyle="--",
        color="C2",
        alpha=0.5,
        zorder=-10,
    )
    ax3.set_ylabel("Phase")

    return (ax1, ax2)


def plot_rabi_sidebands(ax, params: Params):
    """Visualize the frequency of the sidebands.

    :param ax: axis to plot on
    :param params: system parameters
    """
    energy = params.rabi_splitting

    first_sidebands = np.abs(
        -(params.laser_detuning + params.measurement_detuning)
        + np.array([1, -1]) * energy / 2
        + params.ω_c / 2
    )
    second_sidebands = (
        params.Ω * (1 - params.δ)
        - (params.laser_detuning + params.measurement_detuning)
        + np.array([1, -1]) * energy / 2
        - params.ω_c / 2
    )

    for ω in RuntimeParams(params).drive_frequencies:
        ax.axvline(
            ω - params.measurement_detuning,
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


def clone_color_or_default(lines: dict, mode: int):
    """
    Get the color of a mode or a default color if it doesn't exist.

    :param lines: A dictionary of lines indexed by mode index.
    :param mode: The mode to get the color of.
    """

    line = lines.get(mode, None)
    if line is None:
        return f"C{mode}"

    return line.get_color()


def plot_rotating_modes(
    ax,
    solution,
    params,
    plot_uncoupled=False,
    clone_colors=None,
    only_A_site=False,
    correct_for_decay=False,
):
    """Plot the amplitude of the modes in the rotating frame.

    :param ax: The axis to plot on.
    :param solution: The solution to plot.
    :param params: The system parameters.
    :param plot_uncoupled: Whether to plot the uncoupled modes.
    :param clone_colors: A dictionary of lines indexed by mode index
        from which to clone colors from.
    :param only_A_site: Whether to plot only the A site modes.
    :param correct_for_decay: Whether to correct for decay by
        multiplying with an exponential.
    """

    lines = dict()
    if clone_colors is None:
        clone_colors = dict()

    h = solution.y
    if correct_for_decay:
        h = correct_for_decay(solution, params)

    for mode in [1] if only_A_site else coupled_mode_indices(params):
        lines[mode] = ax.plot(
            solution.t,
            np.abs(h[mode]),
            label=mode_name(mode) + (" (rwa)" if params.rwa else ""),
            color=clone_color_or_default(clone_colors, mode),
            linestyle="dashdot" if params.rwa else "-",
        )[0]

    if plot_uncoupled and not only_A_site:
        for mode in uncoupled_mode_indices(params):
            lines[mode] = ax.plot(
                solution.t,
                np.abs(h[mode]),
                label=mode_name(mode) + (" (rwa)" if params.rwa else ""),
                color=clone_color_or_default(clone_colors, mode),
                linestyle="dotted" if params.rwa else "--",
            )[0]

    # ax.legend()
    ax.set_xlabel("Time (1/Ω)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        "Mode Amplitudes in the Rotating Frame"
        + (" (corrected)" if correct_for_decay else "")
    )

    return lines


def plot_rwa_vs_real_amplitudes(ax, solution_no_rwa, solution_rwa, params, **kwargs):
    """Plot the amplitudes of the modes of the system ``params`` in
    the rotating frame with and without the RWA onto ``ax``.

    The keyword arguments are passed to :any:`plot_rotating_modes`.

    :param ax: The axis to plot on.
    :param non_rwa: The solution without the rwa.
    :param rwa: The solution with the rwa.
    :param params: The system parameters.
    :param kwargs: Additional keyword arguments to pass to
        :any:`plot_rotating_modes`.

    :returns: A tuple of the line dictionaries for the non-rwa and rwa solutions.
    """

    original_rwa = params.rwa
    params.rwa = False
    no_rwa_lines = plot_rotating_modes(ax, solution_no_rwa, params, **kwargs)

    params.rwa = True
    rwa_lines = plot_rotating_modes(
        ax, solution_rwa, params, **kwargs, clone_colors=no_rwa_lines
    )

    params.rwa = original_rwa

    return no_rwa_lines, rwa_lines


def plot_spectrum_and_peak_info(ax, peaks: RingdownPeakData, annotate=False):
    """Plot the fft spectrum with peaks.

    :param ax: The axis to plot on.
    :param peaks: The peak data.
    :param annotate: Whether to annotate the peaks.

        If ``true`` each peak will be annotated with it's estimated frequency.
    """

    ax.clear()
    ax.plot(
        peaks.freq,
        peaks.power - peaks.noise_floor,
        label="FFT Power",
        color="C0",
        linewidth=0.1,
        marker="o",
        markersize=3,
    )

    fine_freq = np.linspace(peaks.freq[0], peaks.freq[-1], 1000)
    if annotate:
        for i, (freq, Δfreq, lorentz) in enumerate(
            zip(peaks.peak_freqs, peaks.Δpeak_freqs, peaks.lorentz_params)
        ):
            roundfreq, rounderr = scientific_round(freq, Δfreq)

            t = ax.text(
                freq,
                lorentz[0] * 1.1,
                rf"{i} (${roundfreq}\pm {rounderr}$)",
                fontsize=10,
                clip_on=True,
            )
            t.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="white"))

            ax.plot(
                fine_freq,
                lorentzian(fine_freq, *lorentz),
                "--",
                color="C2",
                alpha=0.5,
            )

    ax.set_title("FFT Spectrum")
    ax.set_xlabel("ω [linear]")
    ax.set_ylabel("Power")
    ax.set_xlim(peaks.freq[0], peaks.freq[-1])
    ax.legend()


def annotate_ringodown_mode_positions(params: Params, ax):
    """
    Add y ticks and vertical lines indicating the theoretical
    frequencies and widths of the modes in that can appear in the
    ringdown spectrum.

    :param params: The system parameters.
    :param ax: The pyplot axis to annotate.
    """

    runtime = RuntimeParams(params)
    ax_ticks = ax.twiny()
    ax_ticks.sharey(ax)
    ax_ticks.set_xticks(runtime.ringdown_frequencies)
    ax_ticks.set_xticklabels(
        [
            f"{mode_name(i, params.N)} = {freq:.2f}"
            for i, freq in enumerate(runtime.ringdown_frequencies)
        ]
    )
    ax_ticks.set_xlim(ax.get_xlim())

    for pos, peak_freq in zip(runtime.ringdown_frequencies, runtime.Ωs):
        ax.axvline(
            pos,
            color="black",
            alpha=0.5,
            linestyle="--",
            zorder=-100,
        )

        ax.axvspan(
            pos - peak_freq.imag / (2 * np.pi),
            pos + peak_freq.imag / (2 * np.pi),
            color="black",
            alpha=0.05,
            linestyle="--",
            zorder=-100,
        )

    return ax
