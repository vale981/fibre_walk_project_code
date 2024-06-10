from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
import functools

# %% interactive


@functools.lru_cache()
def make_params_and_solve(
    total_lifetimes, eom_off_lifetime, ω_c=0.1 / 2, N=10, g_0=1 / 4
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
        laser_detuning=0,
        N=2 * N + 2,
        N_couplings=N,
        measurement_detuning=Ω * (3),
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=0,
        laser_off_time=0,
    )

    params.laser_off_time = params.lifetimes(eom_off_lifetime)
    params.drive_off_time = params.lifetimes(eom_off_lifetime)

    params.drive_override = (
        np.array([params.Ω, params.Ω * (1 - params.δ), params.δ * 2 * params.Ω]),
        np.repeat(params.Ω, 3),
    )

    t = time_axis(params, lifetimes=total_lifetimes, resolution=0.01)
    solution = solve(t, params)
    return params, t, solution


def generate_phase_one_data():
    """
    This generates some intensity data for phase one of the
    calibration protocol, where the aim is to extract the resonator
    spectrum.
    """

    total_lifetimes = 50
    eom_off_lifetime = total_lifetimes * 2 / 3
    fluct_size = 0.05

    params, t, solution = make_params_and_solve(
        total_lifetimes, eom_off_lifetime, N=10, g_0=10
    )
    signal = output_signal(t, solution.y, params)

    rng = np.random.default_rng(seed=0)
    signal += np.mean(abs(signal)) * rng.standard_normal(len(signal)) * fluct_size * 100

    fig = make_figure()
    ax_realtime, ax_spectrum = fig.subplots(2, 1)

    ax_realtime.plot(t[::500], signal[::500])
    ax_realtime.axvline(params.laser_off_time, color="black", linestyle="--")

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
    peak_info = find_peaks(scan, ringdown_params, window, prominence=0.1)
    peak_info = refine_peaks(peak_info, ringdown_params)
    plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params)

    Ω, ΔΩ, ladder = extract_Ω_δ(peak_info, ringdown_params, threshold=0.2)

    for index, type, _ in ladder:
        freq_index = peak_info.peaks[index]
        ax_spectrum.plot(
            peak_info.freq[freq_index],
            peak_info.normalized_power[freq_index],
            "o" if type == 0 else "*",
            color="C4" if type == 0 else "C5",
            label=type,
        )
    return Ω, ΔΩ, ladder
