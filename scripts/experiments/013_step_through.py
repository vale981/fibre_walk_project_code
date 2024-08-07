from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
import functools
import multiprocessing
import copy
from scipy.ndimage import rotate

# %% interactive


def make_shots(params, total_lifetimes, eom_range, eom_steps):
    solutions = []
    t = time_axis(params, lifetimes=total_lifetimes, resolution=0.01)

    pool = multiprocessing.Pool()

    shot_params = []
    for step in range(eom_steps):
        current_params = copy.deepcopy(params)
        current_params.drive_override = (
            np.array(
                [
                    params.Ω
                    * (eom_range[0] + (eom_range[1] - eom_range[0]) * step / eom_steps)
                ]
            ),
            np.array([1.0]),
        )

        shot_params.append((t, current_params))

    solutions = pool.starmap(solve, shot_params)
    return t, solutions


def process_shots(t, solutions, noise_amplitude, params):
    average_power_spectrum = None
    window = (float(params.laser_off_time) or 0, t[-1])

    rng = np.random.default_rng(seed=0)

    # let us get a measure calibrate the noise strength
    signal_amp = 0
    signals = []
    for solution in solutions:
        signal = output_signal(t, solution.y, params)
        signals.append(signal)
        signal_amp += abs(signal).max()

    signal_amp /= len(solutions)

    for signal in signals:
        signal += rng.normal(scale=noise_amplitude * signal_amp, size=len(signal))

        freq, fft = fourier_transform(
            t,
            signal,
            low_cutoff=0.0 * params.Ω,
            high_cutoff=params.Ω * (params.N + 1),
            window=window,
        )

        power = np.abs(fft) ** 2
        if average_power_spectrum is None:
            average_power_spectrum = power

        else:
            average_power_spectrum += power

    return freq, (average_power_spectrum / len(solutions))


def plot_power_spectrum(ax_spectrum, freq, average_power_spectrum, params):
    offset = np.median(average_power_spectrum)
    ax_spectrum.plot(freq, average_power_spectrum)

    runtime = RuntimeParams(params)
    lorentz_freqs = np.linspace(freq.min(), freq.max(), 10000)
    fake_spectrum = np.zeros_like(lorentz_freqs)

    for i, peak_freq in enumerate(runtime.Ωs):
        pos = np.abs(
            params.measurement_detuning
            - peak_freq.real / (2 * np.pi)
            + params.δ * params.Ω
            + params.laser_detuning,
        )

        ax_spectrum.axvline(
            pos,
            color="red",
            linestyle="--",
            zorder=-100,
        )

        amplitude = average_power_spectrum[np.argmin(abs(freq - pos))] - offset
        ax_spectrum.annotate(
            mode_name(i),
            (
                pos + 0.1,
                (amplitude + offset) * (1 if peak_freq.real < 0 else 1.1),
            ),
        )

        fake_spectrum += lorentzian(
            lorentz_freqs, amplitude, pos, peak_freq.imag / np.pi
        )

    ax_spectrum.plot(lorentz_freqs, fake_spectrum + offset)


def generate_data(
    g_0=0.3,
    η_factor=5,
    noise_amplitude=0.3,
    laser_detuning=0,
    N=10,
    eom_ranges=(0.5, 2.5),
    eom_steps=20,
    small_loop_detuning=0,
    excitation_lifetimes=2,
    measurement_lifetimes=4,
):
    η = 0.2
    Ω = 13

    params = Params(
        η=η,
        η_hybrid=η_factor * η,
        Ω=Ω,
        δ=1 / 4,
        ω_c=0.1,
        g_0=g_0,
        laser_detuning=13 * (-1 - 1 / 4) + laser_detuning,
        N=N,
        N_couplings=N,
        measurement_detuning=0,
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=0,
        laser_off_time=0,
        small_loop_detuning=small_loop_detuning,
        drive_override=(np.array([1.0]), np.array([1.0])),
    )

    params.laser_off_time = params.lifetimes(excitation_lifetimes)
    params.drive_off_time = params.lifetimes(excitation_lifetimes)

    t, solutions = make_shots(
        params, excitation_lifetimes + measurement_lifetimes, eom_ranges, eom_steps
    )

    _, (sol_on_res) = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        ((1 + params.δ), (1 + params.δ)),
        1,
    )

    freq, average_power_spectrum = process_shots(t, solutions, noise_amplitude, params)
    _, spectrum_on_resonance = process_shots(t, sol_on_res, noise_amplitude, params)

    fig = make_figure()
    fig.clear()

    ax_multi, ax_single = fig.subplot_mosaic("AA\nBB").values()
    ax_multi.set_title("Averaged Power Spectrum")
    ax_single.set_title("Single-shot Power-Spectrum with EOM on resonance")

    for ax in [ax_multi, ax_single]:
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power")
        ax.set_yscale("log")

    plot_power_spectrum(ax_multi, freq, average_power_spectrum, params)
    plot_power_spectrum(ax_single, freq, spectrum_on_resonance, params)

    runtime = RuntimeParams(params)
    # ax_spectrum.set_yscale(yscale)

    return fig


# %% save
if __name__ == "__main__":
    fig = generate_data(
        g_0=1,
        η_factor=5,
        noise_amplitude=0.25,
        N=2,
        eom_ranges=(1.1, 1.3),
        eom_steps=100,
        small_loop_detuning=0,
        laser_detuning=0,
        excitation_lifetimes=2,
        measurement_lifetimes=20,
    )
