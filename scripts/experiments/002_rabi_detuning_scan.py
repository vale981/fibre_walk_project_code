from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from plot_utils import wrap_plot

# %% interactive


def transient_rabi():
    """A transient rabi oscillation without noise."""

    params = Params(
        η=0.7,
        Ω=13.07,
        δ=13.07 / 4,
        d=13.07 / 4 * 1,
        laser_detuning=0.1,
        Δ=13.07 / 4 * 0.01,
        N=3,
        measurement_detuning=1,
        rwa=False,
    )
    params.laser_off_time = params.lifetimes(10)
    t = time_axis(params, 50, 0.1)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    f, (_, ax) = plot_simulation_result(
        make_figure(), t, signal, params, window=(params.lifetimes(10), t[-1])
    )
    plot_sidebands(ax, params)
    # ax.set_xlim(0.73, 0.77)
    f.suptitle("Transient Rabi oscillation")


def steady_rabi():
    """A steady state rabi oscillation without noise."""

    params = Params(η=0.001, d=0.1, laser_detuning=0, Δ=0.05)
    t = time_axis(params, lifetimes=10, resolution=1)

    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    f, (_, ax) = plot_simulation_result(
        make_figure(), t, signal, params, window=(params.lifetimes(8), t[-1])
    )
    plot_sidebands(ax, params)

    f.suptitle("Steady State Rabi oscillation. No Rabi Sidebands.")


def noisy_transient_rabi():
    """A transient rabi oscillation with noise."""

    params = Params(η=0.001, d=0.1, laser_detuning=0, Δ=0.05)
    t = time_axis(params, 2, 1)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    noise_strength = 0.1
    signal = add_noise(signal, noise_strength)

    f, (_, ax) = plot_simulation_result(make_figure(), t, signal, params)
    plot_sidebands(ax, params)

    f.suptitle(f"Transient Rabi oscillation with noise strength {noise_strength}.")


def ringdown_after_rabi():
    """Demonstrates the nonstationary ringdown of the resonator after turning off the EOM and laser drive."""
    off_lifetime = 4
    laser_detuning = 0.1
    params = Params(η=0.0001, d=0.01, laser_detuning=laser_detuning, Δ=0.00, N=2)

    params.laser_off_time = params.lifetimes(off_lifetime)
    params.drive_off_time = params.lifetimes(off_lifetime)

    t = time_axis(params, lifetimes=5, resolution=1)
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
    """A transient rabi oscillation without noise."""

    params = Params(η=1, δ=1 / 4, d=0.0, laser_detuning=100, N=1)
    t = time_axis(params, 1000, 0.001)
    params.dynamic_detunting = -100, t[-1]
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params)

    f, (_, ax) = plot_simulation_result(make_figure(), t, signal, params)
    plot_sidebands(ax, params)
    # ax.set_xlim(0.73, 0.77)
    f.suptitle("Transient Rabi oscillation")
