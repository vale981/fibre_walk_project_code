from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
from ringfit.plotting import *
import functools
import gc

# %% load data
path = "../../data/11_07_24/second_signal"
scan = ScanData.from_dir(path, extension="npz")


# %% plot scan
gc.collect()
fig = plt.figure("interactive", constrained_layout=True, figsize=(20, 3 * 5))
fig.clf()
(ax, ax2, ax_signal, ax_window, ax_spectrum) = fig.subplot_mosaic("AB;CC;DE").values()
plot_scan(scan, ax=ax_signal, smoothe_output=1e-8, linewidth=0.5)


# %% window
T_step = 0.0002
t_peak = 0.032201
N = 100 - 3
t_scan_peak = t_peak - T_step * N + 4.07e-4  # T * N_steps
win_length = 5e-05
window = t_scan_peak - win_length / 2, t_scan_peak + win_length / 2 * 0.8

ax_signal.axvline(t_peak, color="r", linestyle="--")
ax_signal.axvline(t_scan_peak, color="r", linestyle="--")
ax_signal.axvspan(
    *window,
    color="r",
    linestyle="--",
)
mask = (scan.time > window[0]) & (scan.time < window[1])

ax_window.clear()
ax_window.plot(scan.time[mask], scan.output[mask], linewidth=0.1)

freq, fft = fourier_transform(
    scan.time, scan.output, window=window, low_cutoff=2e6, high_cutoff=90e6
)
ax.clear()
ax.plot(freq, np.abs(fft) ** 2)

ringdown_params = RingdownParams(
    fω_shift=0,
    mode_window=(0, 4),
    fΩ_guess=13e6,
    fδ_guess=0.2 * 13e6,
    η_guess=0.2e6,
    absolute_low_cutoff=2e6,
)


peak_info = find_peaks(scan, ringdown_params, window, prominence=0.01)
peak_info = refine_peaks(peak_info, ringdown_params)


plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params)
print(peak_info.peak_freqs * 1e-6)


Ω, ΔΩ, δ, Δδ, ladder = extract_Ω_δ(
    peak_info,
    ringdown_params,
    Ω_threshold=0.1,
    ladder_threshold=0.1,
    start_peaks=3,
    bifurcations=2,
)


for index, type in ladder:
    freq_index = peak_info.peaks[index]
    print(
        type,
        index,
        (peak_info.peak_freqs[index] - peak_info.peak_freqs[ladder[0][0]]) / Ω,
    )
    ax_spectrum.plot(
        peak_info.freq[freq_index],
        peak_info.normalized_power[freq_index],
        "o" if type.value == StepType.BATH.value else "*",
        color="C4" if type.value == StepType.BATH.value else "C5",
        label=type,
    )
