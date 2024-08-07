from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
import functools

from scipy.ndimage import rotate

# %% interactive


def make_params_and_solve(
    total_lifetimes,
    eom_off_lifetime,
    ω_c=0.1 / 2,
    N=10,
    g_0=1 / 4,
    small_loop_detuning=0,
    laser_detuning=0,
    η=1.0,
    η_hybrid=1.0,
    drive_detuning=0,
):
    """
    Make a set of parameters for the system with the current
    best-known settings.
    """

    Ω = 13

    params = Params(
        η=η,
        η_hybrid=η_hybrid,
        Ω=Ω,
        δ=1 / 4,
        ω_c=ω_c,
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
    )

    params.laser_off_time = params.lifetimes(eom_off_lifetime)
    params.drive_off_time = params.lifetimes(eom_off_lifetime)

    params.drive_override = (
        np.array(
            [
                params.Ω + params.δ * params.Ω * 1.4,
                # params.Ω,
            ]
        ),
        np.array([1.0]),
    )

    print(params.drive_override)

    t = time_axis(params, lifetimes=total_lifetimes, resolution=0.001)
    solution = solve(t, params)
    return params, t, solution


def generate_phase_one_data(
    drive_detuning=0,
    g_0=0.3,
    off_factor=0.5,
    noise=False,
    extra_title="",
    laser_detuning=0,
    yscale="linear",
):
    """
    This generates some intensity data for phase one of the
    calibration protocol, where the aim is to extract the resonator
    spectrum.
    """

    total_lifetimes = 10
    eom_off_lifetime = total_lifetimes * off_factor
    fluct_size = 0.05
    noise_amp = 0.3

    params, t, solution = make_params_and_solve(
        total_lifetimes,
        eom_off_lifetime,
        N=2,
        g_0=g_0,
        small_loop_detuning=0,
        laser_detuning=laser_detuning,
        η=0.18,
        η_hybrid=0.18 * 5,
        drive_detuning=drive_detuning,
    )

    rng = np.random.default_rng(seed=0)
    raw_signal = output_signal(t, solution.y, params)

    signal = raw_signal.copy()
    if noise:
        signal += noise_amp * rng.standard_normal(len(signal))

    fig = make_figure(f"simulation_noise_{noise}", figsize=(20, 3 * 5))

    ax_realtime, ax_rotating_bath, ax_rotating_system, ax_spectrum = (
        fig.subplot_mosaic("""
    AA
    BC
    DD
    DD
    """).values()
    )

    ax_rotating_system.sharey(ax_rotating_bath)
    ax_rotating_system.sharex(ax_rotating_bath)

    for mode in range(2):
        ax_rotating_system.plot(
            t[::50],
            abs((solution.y[mode, ::50]) * (params.η / params.η_hybrid)) ** 2 / 2,
        )

    for mode in range(2, solution.y.shape[0]):
        ax_rotating_bath.plot(
            t[::50], abs((solution.y[mode, ::50])) ** 2, label=mode_name(mode)
        )

    for ax in [ax_rotating_bath, ax_rotating_system]:
        ax.set_xlabel("Time")
        ax.set_ylabel("Intensity")
        ax.legend()

    ax_rotating_bath.set_title("Bath Modes")
    ax_rotating_system.set_title(
        "Hybridized Modes [corrected for magnitude visible in FFT]"
    )

    ax_realtime.plot(t[::50], signal[::50])
    ax_realtime.axvline(params.drive_off_time, color="black", linestyle="--")
    ax_realtime.set_xlabel("Time")
    ax_realtime.set_ylabel("Intensity")
    ax_realtime.set_title("Photo-diode AC Intensity")

    # now we plot the power spectrum
    window = (float(params.laser_off_time) or 0, t[-1])
    # window = (0, float(params.laser_off_time or 0))

    freq, fft, t = fourier_transform(
        t,
        signal,
        low_cutoff=1,
        high_cutoff=params.Ω * 5,
        window=window,
        ret_time=True,
    )

    ax_spectrum.clear()
    power = np.abs(fft) ** 2
    ax_spectrum.plot(freq, power)

    runtime = RuntimeParams(params)

    for i, freq in enumerate(runtime.Ωs.real):
        pos = np.abs(
            params.measurement_detuning
            - freq / (2 * np.pi)
            + params.δ * params.Ω
            + params.laser_detuning,
        )

        ax_spectrum.axvline(
            pos,
            color="red",
            linestyle="--",
            zorder=-100,
        )
        ax_spectrum.annotate(
            mode_name(i),
            (pos + 0.1, power.max() * (0.9 if freq < 0 else 1)),
        )

    ax_spectrum.set_yscale(yscale)

    return fig


# %% save
if __name__ == "__main__":
    fig = generate_phase_one_data(
        noise=True,
        g_0=0.8,
        off_factor=0.5,
        laser_detuning=13 * (1 - 1 / 4),
    )
