from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from plot_utils import wrap_plot

# %% interactive


def transient_rabi():
    """A transient rabi oscillation without noise."""

    params = Params(
        η=0.7,
        Ω=13,
        δ=1 / 4,
        d=0.02,
        laser_detuning=0,
        Δ=0,
        N=3,
        measurement_detuning=1,
        rwa=False,
    )

    params.laser_off_time = params.lifetimes(15)
    t = time_axis(params, 50, 0.1)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    f, (_, ax) = plot_simulation_result(
        make_figure(), t, signal, params, window=(params.lifetimes(15), t[-1])
    )
    plot_sidebands(ax, params)
    f.suptitle("Transient Rabi oscillation")


def steady_rabi():
    """A steady state rabi oscillation without noise."""

    params = Params(
        η=0.7,
        Ω=13,
        δ=1 / 4,
        d=0.02,
        laser_detuning=0,
        Δ=0,
        N=3,
        measurement_detuning=1,
        rwa=False,
    )
    t = time_axis(params, lifetimes=20, resolution=0.01)

    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    f, (_, ax) = plot_simulation_result(
        make_figure(), t, signal, params, window=(params.lifetimes(8), t[-1])
    )
    plot_sidebands(ax, params)

    f.suptitle("Steady State Rabi oscillation. No Rabi Sidebands.")


def noisy_transient_rabi():
    """A transient rabi oscillation with noise."""

    params = Params(
        η=0.7,
        Ω=13,
        δ=1 / 4,
        d=0.02,
        laser_detuning=0,
        Δ=0,
        N=3,
        measurement_detuning=1,
        rwa=False,
    )

    t = time_axis(params, 20, 0.01)

    params.laser_off_time = params.lifetimes(5)

    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    noise_strength = 5
    signal = add_noise(signal, noise_strength)

    f, (_, ax) = plot_simulation_result(
        make_figure(), t, signal, params, window=(params.laser_off_time, t[-1])
    )
    plot_sidebands(ax, params)

    f.suptitle(f"Transient Rabi oscillation with noise strength {noise_strength}.")


def ringdown_after_rabi():
    """Demonstrates the nonstationary ringdown of the resonator after turning off the EOM and laser drive."""
    off_lifetime = 5
    laser_detuning = 0.1

    params = Params(
        η=0.7,
        Ω=13,
        δ=1 / 4,
        d=0.02,
        laser_detuning=laser_detuning,
        Δ=0,
        N=3,
        measurement_detuning=2,
        rwa=False,
    )

    params.laser_off_time = params.lifetimes(off_lifetime)
    params.drive_off_time = params.lifetimes(off_lifetime)

    t = time_axis(params, lifetimes=20, resolution=0.01)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    # noise_strength = 0.1
    # signal = add_noise(signal, noise_strength)

    f, (_, fftax) = plot_simulation_result(
        make_figure(), t, signal, params, window=(params.lifetimes(off_lifetime), t[-1])
    )

    fftax.axvline(params.Ω - params.δ - params.laser_detuning, color="black")
    fftax.axvline(params.laser_detuning, color="black")

    f.suptitle(f"Ringdown after rabi osci EOM after {off_lifetime} lifetimes.")


def sweep():
    """A sweep of the laser over the spectrum."""

    params = Params(
        η=1,
        Ω=1,
        δ=1 / 4,
        d=0.0,
        laser_detuning=-2,
        Δ=0,
        N=3,
        measurement_detuning=0,
        rwa=False,
    )
    t = time_axis(params, params.lifetimes(2000), 0.1)
    params.dynamic_detunting = (2 * params.δ + params.N) * params.Ω, t[-1]
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    f, (_, ax) = plot_simulation_result(make_figure(), t, signal, params)
    plot_sidebands(ax, params)
    # ax.set_xlim(0.73, 0.77)
    f.suptitle("Transient Rabi oscillation")
