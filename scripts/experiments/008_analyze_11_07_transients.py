from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
from ringfit.plotting import *
from scipy.signal import butter, sosfilt
import functools
import gc

path = "../../data/11_07_24/second_signal"
scan = ScanData.from_dir(path, extension="npz")


# %% plot scan
gc.collect()
fig = plt.figure("interactive", constrained_layout=True, figsize=(20, 3 * 5))
fig.clf()
(ax_signal, ax_window, ax_spectrum) = fig.subplot_mosaic("AA;BC").values()
plot_scan(scan, ax=ax_signal, linewidth=0.5, every=1000)

# %% filter
# high_pass_filter = butter(
#     30, 3e6, btype="highpass", output="sos", fs=1 / scan.timestep, analog=False
# )
# scan._output = sosfilt(high_pass_filter, scan.output)


# %% window
T_step = 0.0002
N = 100
t_scan_peak = 0.0057815 + 0.1e-6  # T * N_steps
t_scan_peak = 0.0057815 - 0.09e-6 + 11 * T_step  # T * N_steps
t_peak = t_scan_peak + N * T_step
win_length = 5e-05
window = t_scan_peak, t_scan_peak + win_length * 0.3

ax_signal.axvline(t_peak, color="r", linestyle="--")
ax_signal.axvline(t_scan_peak, color="r", linestyle="--")
ax_signal.axvspan(*window, color="r", linestyle="--")
mask = (scan.time > window[0]) & (scan.time < window[1])

ax_window.clear()


ax_window.plot(scan.time[mask], scan.output[mask], linewidth=0.1)


ringdown_params = RingdownParams(
    fω_shift=0,
    mode_window=(0, 5),
    fΩ_guess=12.9e6,
    fδ_guess=0.2 * 12.9e6,
    η_guess=0.5e6,
    absolute_low_cutoff=2e6,
)

peak_info = find_peaks(scan, ringdown_params, window, prominence=0.08)
peak_info = refine_peaks(peak_info, ringdown_params)


plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params, annotate=True)
print(peak_info.peak_freqs * 1e-6)

b1, b2, b3, reflected = peak_info.peak_freqs[[0, 1, 3, 2]]
Ω = b3 - b2
δ = abs(b3 - 3 * Ω)
δ_2 = reflected - 2 * Ω
detuning = (δ + δ_2) / 2

hyb_amp = peak_info.power[peak_info.peaks[0]] / (np.sqrt(2) * 3.5) ** 2
hyb_width = peak_info.peak_widths[0] * 3.5
hyb_freq = 2 * detuning

ax_spectrum.plot(
    peak_info.freq,
    lorentzian(peak_info.freq, hyb_amp, hyb_freq, hyb_width),
    color="C3",
    label="hybridized mode?",
)

ax_spectrum.legend()
