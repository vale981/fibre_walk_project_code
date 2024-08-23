"""A demonstration of the ringdown spectroscopy protocol."""

from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from ringfit.utils import WelfordAggregator
from rabifun.analysis import *
import multiprocessing
import copy


def solve_shot(
    params: Params, t: np.ndarray, t_before: np.ndarray, t_after: np.ndarray
):
    """A worker function to solve for the time evolution in separate processes.

    :param params: The parameters of the system.
    :param t: The time axis.
    :param t_before: The time axis before the EOM is switched off.
    :param t_after: The time axis after the EOM is switched off.
    """

    solution = solve(t, params)
    amps = solution.y[::, len(t_before) - 1 :]

    return t_after, amps


def make_shots(
    params: Params,
    total_lifetimes: float,
    eom_range: tuple[float, float],
    eom_steps: int,
    num_freq: int = 1,
    randomize_off_time: bool = True,
):
    """Generate a series of shots with varying EOM frequencies.

    The implementation here slightly varies the off time of the laser
    so as to introduce some random relative phases of the modes.

    :param params: The parameters of the system.
    :param total_lifetimes: The total time of the experiment in
        lifetimes.
    :param eom_range: The range of EOM frequencies in units of
        :any:`params.Ω`.
    :param eom_steps: The number of steps in the EOM frequency range.
    :param num_freq: The number of frequencies to drive.  If a number
        greater than 1 is given, the EOM will be driven at multiple
        frequencies where the highest frequency is the base frequency
        plus an consecutive integer multiples of :any:`params.Ω`.
    :param randomize_off_time: Whether to randomize the off time of the
        laser.  This is useful to introduce some random relative phases
        for the modes.
    """

    solutions = []
    shot_params = []

    rng = np.random.default_rng(seed=0)
    off_time = params.laser_off_time or 0
    analyze_time = params.lifetimes(total_lifetimes) - off_time
    t_after = time_axis(params, total_time=analyze_time, resolution=0.01)

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
        off_time += rng.normal(0, 0.1 * off_time) if randomize_off_time else 0

        current_params.laser_off_time = None  # off_time
        current_params.drive_off_time = off_time

        t_before = time_axis(params, total_time=off_time, resolution=0.01)
        t = np.concatenate([t_before[:-1], t_after + t_before[-1]])
        shot_params.append((current_params, t, t_before, t_after))

    pool = multiprocessing.Pool()
    solutions = pool.starmap(solve_shot, shot_params)
    return solutions


def process_shots(
    params: Params,
    solutions: list[tuple[np.ndarray, np.ndarray]],
    noise_amplitude: float,
    num_freq: int,
):
    """
    Calculates the normalized average Fourier power spectrum of a
    series of experimental (simulated) shots.

    :param params: The parameters of the system.
    :param solutions: A list of solutions to process returned by
        :any:`solve_shot`.
    :param noise_amplitude: The amplitude of the noise to add to the
        signal.

        The amplitude is normalized by 2/η which is roughly the steady
        state signal amplitude if a bath mode is excited resonantly by
        a unit-strength input.

    :param num_freq: The number of frequencies to drive.  See
        :any:`make_shots` for details.
    """

    rng = np.random.default_rng(seed=0)

    noise_amplitude /= params.η * np.pi

    aggregate = None
    for t, amps in solutions:
        signal = output_signal(t, amps, params)
        signal += rng.normal(scale=noise_amplitude, size=len(signal))
        window = (0, t[-1])

        freq, fft = fourier_transform(
            t,
            signal,
            low_cutoff=0.3 * params.Ω,
            high_cutoff=params.Ω * (1 + num_freq),
            window=window,
        )

        power = np.abs(fft) ** 2

        # ugly hack because shape is hard to predict
        if aggregate is None:
            aggregate = WelfordAggregator(power)
        else:
            aggregate.update(power)

    assert aggregate is not None  # appease pyright

    max_power = np.max(aggregate.mean)
    return (freq, aggregate.mean / max_power, aggregate.ensemble_std / max_power)


def process_and_plot_results(
    params: Params,
    ax: plt.Axes,
    freq: np.ndarray,
    average_power_spectrum: np.ndarray,
    σ_power_spectrum: np.ndarray,
    annotate: bool = True,
):
    """
    Fits the ringdown spectrum and plots the results.

    :param params: The parameters of the system.
    :param ax: The axis to plot on.
    :param freq: The frequency array.
    :param average_power_spectrum: The average power spectrum obtained from :any:`process_shots`.
    :param σ_power_spectrum: The standard deviation of the power
        spectrum.
    :param annotate: Whether to annotate the plot with peak and mode positions.
    """

    ringdown_params = RingdownParams(
        fω_shift=params.measurement_detuning,
        mode_window=(params.N, params.N),
        fΩ_guess=params.Ω,
        fδ_guess=params.Ω * params.δ,
        η_guess=0.5,
        absolute_low_cutoff=0.3 * params.Ω,
    )

    peak_info = find_peaks(
        freq,
        average_power_spectrum,
        ringdown_params,
        prominence=0.05,
        height=0.1,
        σ_power=σ_power_spectrum,
    )

    peak_info = refine_peaks(peak_info, ringdown_params, height_cutoff=0.05)

    plot_spectrum_and_peak_info(ax, peak_info, annotate=annotate)

    if peak_info.lm_result is not None:
        fine_freq = np.linspace(freq.min(), freq.max(), 5000)
        fine_fit = peak_info.lm_result.eval(ω=fine_freq)
        ax.plot(
            fine_freq,
            fine_fit - peak_info.noise_floor,
            color="C3",
            zorder=-100,
            label="Fit",
        )
        ax.set_ylim(-0.1, max(1, fine_fit.max() * 1.1))

    ax.set_xlabel("Frequency (MHz)")
    ax.legend()

    if annotate:
        annotate_ringodown_mode_positions(params, ax)


def generate_data(
    Ω=13,
    η=0.2,
    g_0=0.5,
    η_factor=5,
    noise_amplitude=0.3,
    laser_detuning=0,
    laser_on_mode=0,
    N=10,
    eom_ranges=(0.5, 2.5),
    eom_steps=20,
    excitation_lifetimes=2,
    measurement_lifetimes=4,
    num_freq=3,
    extra_title="",
    save: str | None = None,
    randomize_off_time: bool = True,
):
    """Simulate and plot the ringdown spectroscopy protocol.

    The idea is to have the laser on ``laser_on_mode`` and to sweep
    the EOM frequency over a range of values given in ``eom_ranges``
    in ``eom_steps`` steps.  For each step, the laser and EOM are
    inputting into the system for a time given by
    ``excitation_lifetimes``.  Then, the ringdown signal is collected
    for a time given by ``measurement_lifetimes``.  (Lifetime units
    are given by ``η``.) The resulting power spectra are averaged and
    then fitted.

    :param Ω: The FSR of the system.
    :param η: The decay rate of the system.
    :param g_0: The coupling strength of the system in units of
        :any:`Ω`.  Note that the effective coupling strength between
        the ``A`` site and the bath modes is reduced by a factor of
        :math:`\sqrt{2}`.

    :param η_factor: The factor by which the decay rate of the A site
        is greater.
    :param noise_amplitude: The amplitude of the noise to add to the
        signal.  See :any:`process_shots` for details.
    :param laser_detuning: The detuning of the laser from the the mode
        it is exciting.
    :param laser_on_mode: The mode that the laser is exciting.
    :param N: The number of bath modes.
    :param eom_ranges: The range of EOM frequencies in units of
        :any:`Ω`.
    :param eom_steps: The number of steps in the EOM frequency range.
    :param excitation_lifetimes: The time the EOM is driving the
        system.
    :param measurement_lifetimes: The time the system is left to ring
        down.

        Note that the laser is not turned off during the ringdown.

    :param num_freq: The number of frequencies to drive.  See
        :any:`make_shots` for details.
    :param extra_title: A string to add to the title of the plot.
    :param save: The filename to save the figure to.  If None,
        the figure is not saved.
    :param randomize_off_time: Whether to randomize the off time of the
        laser.  See :any:`make_shots` for details.

    :returns: The figure containing the plot.
    """

    final_laser_detuning = laser_detuning + (
        0 if laser_on_mode == 0 else (laser_on_mode - 1 / 4) * Ω
    )

    params = Params(
        η=η,
        η_hybrid=η_factor * η,
        Ω=Ω,
        δ=1 / 4,
        ω_c=0.1,
        g_0=g_0 * num_freq,  # as it would be normalized otherwise
        laser_detuning=final_laser_detuning,
        N=N,
        N_couplings=N,
        measurement_detuning=0,
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=0,
        laser_off_time=None,
        small_loop_detuning=0,
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
        randomize_off_time=randomize_off_time,
    )

    freq, average_power_spectrum, σ_power_spectrum = process_shots(
        params,
        solutions,
        noise_amplitude,
        num_freq,
    )

    fig = make_figure(extra_title, figsize=(12, 6))
    fig.clear()
    ax = fig.subplots()

    process_and_plot_results(params, ax, freq, average_power_spectrum, σ_power_spectrum)
    ax.text(
        0.01,
        0.95,
        f"""$Ω/2π = {params.Ω}$MHz
$η/2π = {params.η}MHz$
$g_0 = {g_0}Ω$
$N = {params.N}$
noise = ${noise_amplitude * 2}$
$η_A = {η_factor}η$
EOM range = {eom_ranges[0]:.2f}Ω to {eom_ranges[1]:.2f}Ω
EOM steps = {eom_steps}
excitation time = {excitation_lifetimes} lifetimes
measurement time = {measurement_lifetimes} lifetimes
on mode = {laser_on_mode}
laser detuning = {laser_detuning}
num freq = {num_freq}
total time = {(excitation_lifetimes + measurement_lifetimes) * eom_steps / (params.η * 1e6)}s
randomize off time = {'yes' if randomize_off_time else 'no'} """,
        transform=ax.transAxes,
        ha="left",
        va="top",
        size=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )
    ax.set_title(extra_title)

    fig.tight_layout()

    save_figure(fig, f"002_{save}")
    quick_save_pickle(
        locals(),
        f"002_{save}",
    )

    return fig


# %% save
if __name__ == "__main__":
    generate_data(
        g_0=0.4,
        η_factor=5,
        noise_amplitude=0.3,
        N=5,
        eom_ranges=(0.7, 0.9),
        eom_steps=200,
        laser_detuning=0,
        laser_on_mode=0,
        excitation_lifetimes=2,
        measurement_lifetimes=5,
        num_freq=4,
        randomize_off_time=True,
        extra_title="Laser on A site",
        save="001_laser_on_A",
    )

    fig = generate_data(
        g_0=0.5,
        η_factor=5,
        noise_amplitude=0.3,
        N=5,
        eom_ranges=(1.2, 1.3),
        eom_steps=200,
        laser_detuning=0,
        laser_on_mode=-1,
        excitation_lifetimes=2,
        measurement_lifetimes=4,
        num_freq=1,
        extra_title="Laser on Bath Mode",
        randomize_off_time=True,
        save="002_laser_on_bath",
    )

    fig = generate_data(
        g_0=0.3,
        η_factor=5,
        noise_amplitude=0.0,
        N=5,
        eom_ranges=(1.2, 1.3),
        eom_steps=100,
        laser_detuning=0,
        laser_on_mode=-1,
        excitation_lifetimes=2,
        measurement_lifetimes=4,
        num_freq=1,
        randomize_off_time=False,
        extra_title="Laser on Bath Mode, No randomized off-time.",
        save="003_laser_on_bath_no_random",
    )

    fig = generate_data(
        g_0=0.3,
        η_factor=5,
        noise_amplitude=0.0,
        N=5,
        eom_ranges=(1.2, 1.3),
        eom_steps=100,
        laser_detuning=0,
        laser_on_mode=-1,
        excitation_lifetimes=2,
        measurement_lifetimes=4,
        num_freq=1,
        randomize_off_time=True,
        extra_title="Laser on Bath Mode, Randomized off-time.",
        save="004_laser_on_bath_random",
    )
