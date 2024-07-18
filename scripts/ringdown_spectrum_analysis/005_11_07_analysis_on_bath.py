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
t_scan_peak = 0.0057815 - 0.095e-6 + 28 * T_step  # T * N_steps
t_peak = t_scan_peak + N * T_step
win_length = 5e-05

window = t_scan_peak, t_scan_peak + win_length * 0.2


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

# %% peak analysis
## herein we detect the pertinent peaks and refine them using a lorentzian fit
ringdown_params = RingdownParams(
    fω_shift=0,
    mode_window=(0, 4),
    fΩ_guess=12.9e6,
    fδ_guess=0.2 * 12.9e6,
    η_guess=0.5e6,
    absolute_low_cutoff=5e6,
)

peak_info = find_peaks(scan, ringdown_params, window, prominence=0.1)
peak_info = refine_peaks(peak_info, ringdown_params)
plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params, annotate=True)


# %% hand-crafted interpretation
b1, b2, b3 = peak_info.peak_freqs[[0, 1, 2]]
Ω = b3 - b2
hyb_freq = b1 - 0.18 * Ω
alt_Ω = b2 - b1

hyb_amp = peak_info.power[peak_info.peaks[1]] / (np.sqrt(2) * 5) ** 2 * 10
hyb_width = peak_info.peak_widths[1] * 5

ax_spectrum.plot(
    peak_info.freq,
    lorentzian(peak_info.freq, hyb_amp, hyb_freq, hyb_width),
    color="C3",
    label="hybridized mode?",
)
# %%
ax_spectrum.legend()
fig.suptitle(
    f"""
Analysis of the data from the 11/07.
We modulated at the FSR (13MHz) and FSR-δ and 2δ.
*Laser on first bath mode*

Here the laser hits the first bathmode beside the hybidized modes. This is the reason
why the peaks toward higher frequencies are split (we're detuned from the unperturbed spectrum).

The FSR here is {Ω*1e-6:.2f}MHz as ascertained from the bath modes. According to the simulation for driving
on the bath mode, we should see at least one hybridized mode where the red lorentzian is plotted. It's amplitude
is fantasy, but there certainly there seems to be something there!
"""
)


# %% save
if __name__ == "__main__":
    save_figure(fig, "005_11_07_analysis_on_bath")

    quick_save_pickle(
        dict(
            window=window,
            peak_info=peak_info,
            ringdown_parms=ringdown_params,
            Ω=Ω,
        ),
        "005_results",
    )
