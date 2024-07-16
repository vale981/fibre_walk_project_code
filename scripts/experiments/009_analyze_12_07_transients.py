from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
from ringfit.plotting import *
import functools
import gc

# %% load data
path = "../../data/15_07_24/finite_life_only_bath"
scan = ScanData.from_dir(path, extension="npz")


# %% plot scan
gc.collect()
fig = plt.figure("interactive", constrained_layout=True, figsize=(20, 3 * 5))
fig.clf()
(ax, ax2, ax_signal, ax_window, ax_spectrum) = fig.subplot_mosaic("AB;CC;DE").values()
plot_scan(scan, ax=ax_signal, smoothe_output=1e-8, linewidth=0.5)


# %% window
T_step = 0.00010416666666666667
t_peak = 0.055516 + 2 * T_step - 0.2 * 0.28e-6 - 200 * T_step
t_scan_peak = t_peak
win_length = 2.5e-05 * 0.3
window = t_scan_peak, t_scan_peak + win_length

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
    scan.time, scan.output, window=window, low_cutoff=3e6, high_cutoff=90e6
)
ax.clear()
ax.plot(freq, np.abs(fft) ** 2)

ringdown_params = RingdownParams(
    fω_shift=0,
    mode_window=(0, 4),
    fΩ_guess=13e6,
    fδ_guess=0.2 * 13e6,
    η_guess=4e6,
    absolute_low_cutoff=4e6,
)


peak_info = find_peaks(scan, ringdown_params, window, prominence=0.1)
peak_info = refine_peaks(peak_info, ringdown_params)


plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params)
# %%

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
