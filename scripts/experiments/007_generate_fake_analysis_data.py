from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
import functools

# %% interactive


@functools.lru_cache()
def make_params_and_solve(
    total_lifetimes,
    eom_off_lifetime,
    ω_c=0.1 / 2,
    N=10,
    g_0=1 / 4,
    small_loop_detuning=0,
    laser_detuning=0,
):
    """
    Make a set of parameters for the system with the current
    best-known settings.
    """

    Ω = 13

    params = Params(
        η=0.5,
        Ω=Ω,
        δ=1 / 4,
        ω_c=ω_c,
        g_0=g_0,
        laser_detuning=laser_detuning,
        N=N,
        N_couplings=N,
        measurement_detuning=Ω * (3),
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
        np.array([params.Ω, params.Ω * (1 - params.δ), params.δ * 2 * params.Ω]),
        np.repeat(params.Ω, 3),
    )
    params.drive_override[1][-1] *= 4
    params.drive_override[1][-1] *= 4

    t = time_axis(params, lifetimes=total_lifetimes, resolution=0.01)
    solution = solve(t, params)
    return params, t, solution


def generate_phase_one_data():
    """
    This generates some intensity data for phase one of the
    calibration protocol, where the aim is to extract the resonator
    spectrum.
    """

    total_lifetimes = 20
    eom_off_lifetime = total_lifetimes * 1 / 2
    fluct_size = 0.05
    SNR = 0.8

    params, t, solution = make_params_and_solve(
        total_lifetimes,
        eom_off_lifetime,
        N=20,
        g_0=1,
        small_loop_detuning=0,  # 0.05,
        laser_detuning=-0.01,
    )
    signal = output_signal(t, solution.y, params)

    rng = np.random.default_rng(seed=0)
    signal += (
        np.sqrt(np.mean(abs(signal) ** 2)) / SNR * rng.standard_normal(len(signal))
    )

    fig = make_figure()

    ax_realtime, ax_rotating, ax_spectrum = fig.subplot_mosaic("""
    AB
    CC
    """).values()

    for mode in range(solution.y.shape[0]):
        ax_rotating.plot(t[::50], np.abs(solution.y[mode, ::50]) ** 2)
    ax_rotating.set_xlabel("Time")
    ax_rotating.set_ylabel("Intensity")
    ax_rotating.set_title("Mode Intensities in Rotating Frame")

    ax_realtime.plot(t[::50], signal[::50])
    ax_realtime.axvline(params.laser_off_time, color="black", linestyle="--")
    ax_realtime.set_xlabel("Time")
    ax_realtime.set_ylabel("Intensity")
    ax_realtime.set_title("Measures Intensity")

    # now we plot the power spectrum
    window = (float(params.laser_off_time or 0), float(t[-1]))

    ringdown_params = RingdownParams(
        fω_shift=params.measurement_detuning,
        mode_window=(params.N + 2, params.N + 2),
        fΩ_guess=params.Ω * (1 + rng.standard_normal() * fluct_size),
        fδ_guess=params.Ω * params.δ * (1 + rng.standard_normal() * fluct_size),
        η_guess=params.η * (1 + rng.standard_normal() * fluct_size),
    )

    scan = ScanData(np.ones_like(signal), signal, t)
    peak_info = find_peaks(scan, ringdown_params, window, prominence=0.1 / 4)
    peak_info = refine_peaks(peak_info, ringdown_params)
    plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params)

    Ω, ΔΩ, δ, Δδ, ladder = extract_Ω_δ(
        peak_info, ringdown_params, Ω_threshold=0.1, ladder_threshold=0.1, start_peaks=4
    )

    for index, type in ladder:
        freq_index = peak_info.peaks[index]
        print(type, type is StepType.BATH)
        ax_spectrum.plot(
            peak_info.freq[freq_index],
            peak_info.normalized_power[freq_index],
            "o" if type.value == StepType.BATH.value else "*",
            color="C4" if type.value == StepType.BATH.value else "C5",
            label=type,
        )

    fig.suptitle(
        f"""Calibration Phase One Demonstration\n N={params.N} * 2 modes g_0={params.g_0}, SNR={SNR}, Ω (input) = {params.Ω:.2f}, δ (input) = {params.Ω*params.δ:.2f}
Ω={Ω:.2f} ± {ΔΩ:.2f}, δ={δ:.2f} ± {Δδ:.2f}
"""
    )

    return Ω, ΔΩ, δ, Δδ, ladder
