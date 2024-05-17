from rabifun.plots import plot_simulation_result
from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from plot_utils import autoclose

# %% interactive


@autoclose
def transient_rabi():
    """A transient rabi oscillation without noise."""

    params = Params(η=0.001, d=0.1, laser_detuning=0, Δ=0.05)
    t = time_axis(params, 2, 1)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params.laser_detuning)

    f, (_, ax) = plot_simulation_result(t, signal, params)
    plot_sidebands(ax, params)
    f.suptitle("Transient Rabi oscillation")


@autoclose
def steady_rabi():
    """A steady state rabi oscillation without noise."""

    params = Params(η=0.001, d=0.1, laser_detuning=0, Δ=0.05)
    t = time_axis(params, lifetimes=10, resolution=1)

    solution = solve(t, params)
    signal = output_signal(t, solution.y, params.laser_detuning)

    f, (_, ax) = plot_simulation_result(
        t, signal, params, window=(params.lifetimes(8), t[-1])
    )
    plot_sidebands(ax, params)

    f.suptitle("Steady State Rabi oscillation. No Rabi Sidebands.")


@autoclose
def noisy_transient_rabi():
    """A transient rabi oscillation with noise."""

    params = Params(η=0.001, d=0.1, laser_detuning=0, Δ=0.05)
    t = time_axis(params, 2, 1)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params.laser_detuning)

    noise_strength = 0.1
    signal = add_noise(signal, noise_strength)

    f, (_, ax) = plot_simulation_result(t, signal, params)
    plot_sidebands(ax, params)

    f.suptitle(f"Transient Rabi oscillation with noise strength {noise_strength}.")


@autoclose
def ringdown_after_rabi():
    """Demonstrates the nonstationary ringdown of the resonator after turning off the EOM and laser drive."""
    off_lifetime = 4
    laser_detuning = 0.1

    params = Params(η=0.001, d=0.1, laser_detuning=laser_detuning, Δ=0.5)
    params.laser_off_time = params.lifetimes(off_lifetime)
    params.drive_off_time = params.lifetimes(off_lifetime)

    t = time_axis(params, lifetimes=5, resolution=1)
    solution = solve(t, params)
    signal = output_signal(t, solution.y, params.laser_detuning)

    # noise_strength = 0.1
    # signal = add_noise(signal, noise_strength)

    f, (_, fftax) = plot_simulation_result(
        t, signal, params, window=(params.lifetimes(off_lifetime), t[-1])
    )

    fftax.axvline(params.Ω - params.laser_detuning, color="black")
    fftax.axvline(params.laser_detuning, color="black")

    f.suptitle(f"Ringdown after rabi osci EOM after {off_lifetime} lifetimes.")
