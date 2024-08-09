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


def solve_shot(t, params, t_before, t_after):
    solution = solve(t, params)
    amps = solution.y[::, len(t_before) - 1 :]

    return t_after, amps


def make_shots(params, total_lifetimes, eom_range, eom_steps, σ_modulation_time):
    solutions = []

    analyze_time = params.lifetimes(total_lifetimes) - params.laser_off_time
    t_after = time_axis(params, total_time=analyze_time, resolution=0.01)

    pool = multiprocessing.Pool()

    shot_params = []
    rng = np.random.default_rng(seed=0)

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

        off_time = rng.normal(
            params.laser_off_time, σ_modulation_time * params.lifetimes(1)
        )
        current_params.laser_off_time = off_time
        current_params.drive_off_time = off_time
        current_params.total_lifetimes = (off_time + analyze_time) / params.lifetimes(1)

        t_before = time_axis(params, total_time=off_time, resolution=0.01)
        t = np.concatenate([t_before[:-1], t_after + t_before[-1]])

        shot_params.append((t, current_params, t_before, t_after))

    solutions = pool.starmap(solve_shot, shot_params)
    return solutions


def process_shots(solutions, noise_amplitude, params):
    average_power_spectrum = None

    rng = np.random.default_rng(seed=0)

    # let us get a measure calibrate the noise strength
    signals = []
    for t, amps in solutions:
        signal = output_signal(t, amps, params)
        signals.append((t, signal))

    noise_amplitude *= 2 * 2 * np.pi / params.η
    for t, signal in signals:
        signal += rng.normal(scale=noise_amplitude, size=len(signal))
        window = (0, t[-1])

        freq, fft = fourier_transform(
            t,
            signal,
            low_cutoff=0.5 * params.Ω,
            high_cutoff=params.Ω * 2.5,
            window=window,
        )

        power = np.abs(fft) ** 2
        power = power / power.max()
        if average_power_spectrum is None:
            average_power_spectrum = power

        else:
            average_power_spectrum += power

    power = average_power_spectrum / len(solutions)
    power -= np.median(power)
    power /= power.max()

    return (freq, power)


def plot_power_spectrum(
    ax_spectrum, freq, average_power_spectrum, params, annotate=True
):
    # ax_spectrum.plot(freq, average_power_spectrum)
    runtime = RuntimeParams(params)

    ringdown_params = RingdownParams(
        fω_shift=params.measurement_detuning,
        mode_window=(3, 3),
        fΩ_guess=params.Ω,
        fδ_guess=params.Ω * params.δ,
        η_guess=0.5,
        absolute_low_cutoff=8,
    )

    peak_info = find_peaks(
        freq, average_power_spectrum, ringdown_params, prominence=0.1
    )
    peak_info, lm_result = refine_peaks(peak_info, ringdown_params, height_cutoff=0.05)
    peak_info.power = average_power_spectrum
    plot_spectrum_and_peak_info(
        ax_spectrum, peak_info, ringdown_params, annotate=annotate
    )
    if lm_result is not None:
        ax_spectrum.plot(freq, lm_result.best_fit, color="red")

    for i, peak_freq in enumerate(runtime.Ωs):
        pos = np.abs(
            params.measurement_detuning
            - peak_freq.real / (2 * np.pi)
            + params.δ * params.Ω
            + params.laser_detuning,
        )

        ax_spectrum.axvline(
            pos,
            color="black",
            alpha=0.5,
            linestyle="--",
            zorder=-100,
        )

        ax_spectrum.axvspan(
            pos - peak_freq.imag / (2 * np.pi),
            pos + peak_freq.imag / (2 * np.pi),
            color="black",
            alpha=0.05,
            linestyle="--",
            zorder=-100,
        )


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
    σ_modulation_time=0.01,
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

    solutions = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        eom_ranges,
        eom_steps,
        σ_modulation_time,
    )

    (sol_on_res) = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        ((1 + params.δ), (1 + params.δ)),
        1,
        0,
    )

    (sol_on_res_bath) = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        ((1), (1)),
        1,
        0,
    )

    freq, average_power_spectrum = process_shots(solutions, noise_amplitude, params)
    _, spectrum_on_resonance = process_shots(sol_on_res, noise_amplitude, params)
    _, spectrum_on_resonance_bath = process_shots(
        sol_on_res_bath, noise_amplitude, params
    )

    fig = make_figure()
    fig.clear()

    fig.suptitle(f"""
    Spectroscopy Protocol V2

    Ω/2π = {params.Ω}MHz, η/2π = {params.η}MHz, g_0 = {params.g_0}Ω, N = {params.N}
    noise amplitude = {noise_amplitude} * 2/η, η_A = {η_factor} x η, EOM stepped from {eom_ranges[0]:.2f}Ω to {eom_ranges[1]:.2f}Ω in {eom_steps} steps
    """)
    ax_multi, ax_single, ax_single_bath = fig.subplot_mosaic("AA\nBC").values()

    plot_power_spectrum(ax_multi, freq, average_power_spectrum, params)
    plot_power_spectrum(ax_single, freq, spectrum_on_resonance, params, annotate=False)
    plot_power_spectrum(
        ax_single_bath, freq, spectrum_on_resonance_bath, params, annotate=False
    )

    runtime = RuntimeParams(params)
    for ax in [ax_multi, ax_single, ax_single_bath]:
        ax.set_xlabel("Frequency (MHz)")
        ax.sharex(ax_multi)
        ax.sharey(ax_multi)

        ax_ticks = ax.twiny()
        ax_ticks.sharey(ax)
        ax_ticks.set_xticks(runtime.ringdown_frequencies)
        ax_ticks.set_xticklabels(
            [mode_name(i, params.N) for i in range(2 * params.N + 2)]
        )
        ax_ticks.plot(freq, np.zeros_like(freq), alpha=0)
        ax_ticks.set_xlim(ax.get_xlim())

    ax_multi.set_title("Averaged Power Spectrum (sans noise offset)")
    ax_single.set_title("Single-shot, EOM on A site")
    ax_single_bath.set_title("Single-shot, EOM on bath mode")

    # ax_spectrum.set_yscale(yscale)

    fig.tight_layout()
    return fig


# %% save
if __name__ == "__main__":
    fig = generate_data(
        g_0=1,
        η_factor=5,
        noise_amplitude=2e-3,
        N=2,
        eom_ranges=(1.1, 1.35),
        eom_steps=100,
        small_loop_detuning=0,
        laser_detuning=0,
        excitation_lifetimes=2,
        measurement_lifetimes=30,
        σ_modulation_time=0.2,
    )
