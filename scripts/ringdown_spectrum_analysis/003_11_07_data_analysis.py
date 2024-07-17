from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
from rabifun.analysis import *
from ringfit.data import ScanData
from ringfit.plotting import *
import gc

path = "../../data/11_07_24/second_signal"
scan = ScanData.from_dir(path, extension="npz")


# %% plot scan
gc.collect()
fig = plt.figure("interactive", constrained_layout=True, figsize=(20, 3 * 5))
fig.clf()
(ax_signal, ax_window, ax_spectrum) = fig.subplot_mosaic("AA;BC").values()
plot_scan(scan, ax=ax_signal, linewidth=0.5, every=1000)


# %% window

# here we select a step that is resonant with a hybridized peak and
# then plot the full photodiode voltage trace and just the window

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
ax_signal.set_title("Full photodiode voltage trace")
ax_signal.set_xlabel("Time [s]")
ax_signal.set_ylabel("Voltage [arb]")

mask = (scan.time > window[0]) & (scan.time < window[1])
ax_window.clear()
ax_window.plot(scan.time[mask], scan.output[mask], linewidth=0.1)
ax_window.set_title("Windowed photodiode voltage trace")
ax_window.set_xlabel("Time [s]")

# %% detect peaks

## herein we detect the pertinent peaks and refine them using a lorentzian fit
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


# %% hand-crafted interpretation
b1, b2, b3, reflected = peak_info.peak_freqs[[0, 1, 3, 2]]
Ω = b3 - b2
δ = abs(b3 - 3 * Ω)
δ_2 = reflected - 2 * Ω
detuning = (δ + δ_2) / 2

alt_Ω = b2 - b1

hyb_amp = peak_info.power[peak_info.peaks[3]] / (np.sqrt(2) * 5) ** 2
hyb_width = peak_info.peak_widths[3] * 5
hyb_freq = 2 * detuning

ax_spectrum.plot(
    peak_info.freq,
    lorentzian(peak_info.freq, hyb_amp, hyb_freq, hyb_width),
    color="C3",
    label="hybridized mode?",
)

ax_spectrum.legend()
fig.suptitle(
    f"""
Analysis of the data from the 11/07.
We modulated at the FSR (13MHz) and FSR-δ and 2δ.

In the bottom right a Fourier transform of the windowed signal in the bottom left is shown.
The first large peak is the first bath mode after the hybridized mode. It has a distance from the second bath mode of {alt_Ω*1e-6:.2f}MHz.
The second and third peaks have a distance of {Ω*1e-6:.2f}MHz, hence the first bath mode also hybridizes with the small loop! In the second half of the trace
up top we can see that the first bath mode is broader the n the second and so on, corroborating this interpretation. From there we can also surmise that the decay rate
of the hybridized mode is about 3.5 times that of the bath modes. Using this, we can estimate the amplitude of the hybridized mode and its width, which we then plot as
a lorentzian. (If the bath mode was excited to the same degree as the third bath mode which is unlikely.)

The width of the thrid peak is {peak_info.peak_widths[3]*1e-6:.2f}MHz. The estimated decay rate of the hybridized mode is {hyb_width*1e-6:.2f}MHz.
"""
)


# %% save
if __name__ == "__main__":
    save_figure(fig, "003_11_07_analysis")

    quick_save_pickle(
        dict(
            window=window,
            peak_info=peak_info,
            ringdown_parms=ringdown_params,
            detuning=detuning,
        ),
        "003_results",
    )
