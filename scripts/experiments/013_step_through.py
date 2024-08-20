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


class WelfordAggregator:
    """A class to aggregate values using the Welford algorithm.

    The Welford algorithm is an online algorithm to calculate the mean
    and variance of a series of values.

    The aggregator keeps track of the number of samples the mean and
    the variance.  Aggregation of identical values is prevented by
    checking the sample index.  Tracking can be disabled by setting
    the initial index to ``None``.

    See also the `Wikipedia article
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm>`_.

    :param first_value: The first value to aggregate.
    """

    __slots__ = ["n", "mean", "_m_2"]

    def __init__(self, first_value: np.ndarray):
        self.n = 1
        self.mean = first_value
        self._m_2 = np.zeros_like(first_value)

    def update(self, new_value: np.ndarray):
        """Updates the aggregator with a new value."""

        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self._m_2 += np.abs(delta) * np.abs(delta2)

    @property
    def sample_variance(self) -> np.ndarray:
        """
        The empirical sample variance.  (:math:`\sqrt{N-1}`
        normalization.)
        """

        if self.n == 1:
            return np.zeros_like(self.mean)

        return self._m_2 / (self.n - 1)

    @property
    def ensemble_variance(self) -> np.ndarray:
        """The ensemble variance."""
        return self.sample_variance / self.n

    @property
    def ensemble_std(self) -> np.ndarray:
        """The ensemble standard deviation."""
        return np.sqrt(self.ensemble_variance)


def solve_shot(t, params, t_before, t_after):
    solution = solve(t, params)
    amps = solution.y[::, len(t_before) - 1 :]

    return t_after, amps


def make_shots(params, total_lifetimes, eom_range, eom_steps, num_freq):
    solutions = []

    analyze_time = params.lifetimes(total_lifetimes) - params.laser_off_time
    t_after = time_axis(params, total_time=analyze_time, resolution=0.01)

    pool = multiprocessing.Pool()

    shot_params = []
    rng = np.random.default_rng(seed=0)

    for step in range(eom_steps):
        base = params.Ω * (
            eom_range[0] + (eom_range[1] - eom_range[0]) * step / eom_steps
        )

        current_params = copy.deepcopy(params)
        current_params.drive_override = (
            base + params.Ω * np.arange(num_freq),
            np.ones(num_freq),
        )
        current_params.drive_phases = rng.uniform(0, 2 * np.pi, size=num_freq)

        off_time = rng.normal(params.laser_off_time, 0.1 * params.laser_off_time)

        params.laser_off_time
        current_params.laser_off_time = off_time
        current_params.drive_off_time = off_time
        current_params.total_lifetimes = (off_time + analyze_time) / params.lifetimes(1)

        t_before = time_axis(params, total_time=off_time, resolution=0.01)
        t = np.concatenate([t_before[:-1], t_after + t_before[-1]])

        shot_params.append((t, current_params, t_before, t_after))

    solutions = pool.starmap(solve_shot, shot_params)
    return solutions


def process_shots(solutions, noise_amplitude, params):
    rng = np.random.default_rng(seed=0)

    # let us get a measure calibrate the noise strength
    signals = []
    for t, amps in solutions:
        signal = output_signal(t, amps, params)
        signals.append((t, signal))

    noise_amplitude *= 2 * 2 * np.pi / params.η

    aggregate = None
    for t, signal in signals:
        signal += rng.normal(scale=noise_amplitude, size=len(signal))
        window = (0, t[-1])

        freq, fft = fourier_transform(
            t,
            signal,
            low_cutoff=0.1 * params.Ω,
            high_cutoff=params.Ω * 5,
            window=window,
        )

        power = np.abs(fft) ** 2

        # ugly hack because shape is hard to predict
        if aggregate is None:
            aggregate = WelfordAggregator(power)
        else:
            aggregate.update(power)

    max_power = np.max(aggregate.mean)
    return (freq, aggregate.mean / max_power, aggregate.ensemble_std / max_power)


def plot_power_spectrum(
    ax_spectrum, freq, average_power_spectrum, σ_power_spectrum, params, annotate=True
):
    # ax_spectrum.plot(freq, average_power_spectrum)
    runtime = RuntimeParams(params)

    ringdown_params = RingdownParams(
        fω_shift=params.measurement_detuning,
        mode_window=(4, 4),
        fΩ_guess=params.Ω,
        fδ_guess=params.Ω * params.δ,
        η_guess=0.5,
        absolute_low_cutoff=0.1 * params.Ω,
    )

    peak_info = find_peaks(
        freq, average_power_spectrum, ringdown_params, prominence=0.05, height=0.1
    )

    peak_info, lm_result = refine_peaks(
        peak_info, ringdown_params, height_cutoff=0.05, σ=σ_power_spectrum
    )

    peak_info.power = average_power_spectrum
    plot_spectrum_and_peak_info(
        ax_spectrum, peak_info, ringdown_params, annotate=annotate
    )
    if lm_result is not None:
        # print(lm_result.fit_report())
        fine_freq = np.linspace(freq.min(), freq.max(), 5000)
        fine_fit = lm_result.eval(ω=fine_freq)
        ax_spectrum.plot(fine_freq, fine_fit, color="red")
        ax_spectrum.set_ylim(-0.1, max(1, fine_fit.max() * 1.1))

    print(runtime.Ωs.real / (2 * np.pi))

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
    g_0=0.5,
    η_factor=5,
    noise_amplitude=0.3,
    laser_detuning=0,
    N=10,
    eom_ranges=(0.5, 2.5),
    eom_steps=20,
    small_loop_detuning=0,
    excitation_lifetimes=2,
    measurement_lifetimes=4,
    num_freq=3,
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
        laser_detuning=laser_detuning,
        N=N,
        N_couplings=N,
        measurement_detuning=0,
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=0,
        laser_off_time=0,
        small_loop_detuning=small_loop_detuning,
        drive_override=(np.array([]), np.array([])),
    )

    params.laser_off_time = params.lifetimes(excitation_lifetimes)
    params.drive_off_time = params.lifetimes(excitation_lifetimes)

    solutions = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        eom_ranges,
        eom_steps,
        num_freq,
    )

    (sol_on_res) = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        ((1 + params.δ), (1 + params.δ)),
        1,
        num_freq,
    )

    (sol_on_res_bath) = make_shots(
        params,
        excitation_lifetimes + measurement_lifetimes,
        ((1 + params.δ * 1.1), (1 + params.δ * 1.1)),
        1,
        num_freq,
    )

    freq, average_power_spectrum, σ_power_spectrum = process_shots(
        solutions, noise_amplitude, params
    )
    _, spectrum_on_resonance, σ_power_spectrum_on_resonance = process_shots(
        sol_on_res, noise_amplitude, params
    )
    _, spectrum_on_resonance_bath, σ_power_spectrum_on_resonance_bath = process_shots(
        sol_on_res_bath, noise_amplitude, params
    )

    fig = make_figure()
    fig.clear()

    fig.suptitle(f"""
    Spectroscopy Protocol V2

    Ω/2π = {params.Ω}MHz, η/2π = {params.η}MHz, g_0 = {params.g_0}Ω, N = {params.N}
    noise amplitude = {noise_amplitude} * 2/η, η_A = {η_factor} x η, EOM stepped from {eom_ranges[0]:.2f}Ω to {eom_ranges[1]:.2f}Ω in {eom_steps} steps
    total time = {(excitation_lifetimes + measurement_lifetimes) * eom_steps / (params.η * 1e6)}s
    """)
    ax_multi, ax_single, ax_single_bath = fig.subplot_mosaic("AA\nBC").values()

    plot_power_spectrum(
        ax_multi, freq, average_power_spectrum, σ_power_spectrum, params
    )
    plot_power_spectrum(
        ax_single,
        freq,
        spectrum_on_resonance,
        σ_power_spectrum_on_resonance,
        params,
        annotate=False,
    )
    plot_power_spectrum(
        ax_single_bath,
        freq,
        spectrum_on_resonance_bath,
        σ_power_spectrum_on_resonance_bath,
        params,
        annotate=False,
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

    ax_multi.set_title("Averaged Power Spectrum")
    ax_single.set_title("Single-shot, No detuning")
    ax_single_bath.set_title("Single-shot, EOM 10% detuned")

    # ax_spectrum.set_yscale(yscale)

    fig.tight_layout()
    return fig


# %% save
if __name__ == "__main__":
    fig = generate_data(
        g_0=0.5,
        η_factor=5,
        noise_amplitude=5e-3,
        N=5,
        eom_ranges=(0.7, 0.9),  # (1.9, 2.1),
        eom_steps=100,
        small_loop_detuning=0,
        laser_detuning=0,
        excitation_lifetimes=1,
        measurement_lifetimes=3,
        num_freq=4,
    )
