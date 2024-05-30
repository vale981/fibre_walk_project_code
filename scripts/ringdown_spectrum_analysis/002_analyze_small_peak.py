from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *
from rabifun.analysis import *
import numpy as np
import scipy
from collections import OrderedDict
import networkx as nx
from functools import reduce

# %% load data
prev_data = quick_load_pickle("001_results")


path = "../../data/22_05_24/ringdown_try_2"
scan = ScanData.from_dir(path)
window = prev_data["window"]

# %% extract peak
a_site = 1
peak_freq = (
    prev_data["freq"][prev_data["peaks"][prev_data["unmatched"][a_site]]][0] * 1e6
)
# peak_freq = prev_data["freq"][prev_data["peaks"][8]] * 1e6
freq, fft = fourier_transform(
    scan.time,
    scan.output,
    window=window,
    low_cutoff=peak_freq - 0.4e6,
    high_cutoff=peak_freq + 0.4e6,
)
power = np.abs(fft) ** 2
power = power / np.max(power)

# %% Make Plot
fig = make_figure()
(ax_real, ax_freq) = fig.subplots(2, 1)


# %% Plot Frequency Domain
ax_freq.clear()
ax_freq.plot(freq, power, label="Data")
ax_freq.set_xlabel("Frequency (Hz)")
ax_freq.set_ylabel("Power (a.u.)")


# %% Fit Lorentzian
p0 = [1, peak_freq, 0.1e6, 0]
popt, pcov = scipy.optimize.curve_fit(
    lorentzian,
    freq,
    power,
    p0=p0,
    bounds=([0, min(freq), 0, 0], [np.inf, max(freq), np.inf, 1]),
)
perr = np.sqrt(np.diag(pcov))

ω_peak, Δω_peak = popt[1], perr[1]
γ_peak, Δγ_peak = popt[2], perr[2]
# ax_freq.plot(freq, lorentzian(freq, *p0), label="Initial Guess")
ax_freq.plot(freq, lorentzian(freq, *popt), label="Fit")

ax_freq.legend()
ax_freq.set_title(
    f"γ = {γ_peak*1e-6:.2f} ± {Δγ_peak*1e-6:.2f} MHz, ω = {ω_peak*1e-6:.2f} ± {Δω_peak*1e-6:.2f} MHz"
)

# %% Plot Time Domain from filtered fft
ax_real.clear()

full_freq, full_fft, full_t = fourier_transform(
    scan.time,
    scan.output,
    window=window,
    ret_time=True,
)

widths = 10
peak_range = (
    ω_peak - widths * γ_peak / (2 * np.pi),
    ω_peak + widths * γ_peak / (2 * np.pi),
)
filtered_fft = full_fft.copy()  # full_fft.copy()
mask = (full_freq < peak_range[0]) | (full_freq > peak_range[1])
filtered_fft[mask] = 0


reconstructed = scipy.fft.irfft(filtered_fft, workers=os.cpu_count(), n=len(full_t))
ax_real.plot(
    full_t,
    reconstructed,
    label="Reconstructed",
)
ax_real.set_xlim(full_t[0], full_t[0] + 3 * 2 * np.pi / γ_peak)
ax_real.set_xlabel("Time (s)")
ax_real.set_ylabel("Amplitude (a.u.)")
ax_real.set_title("Reconstructed Signal")

ax_real.plot(
    full_t,
    max(reconstructed)
    * np.cos(2 * np.pi * ω_peak * (full_t - full_t[0]))
    * np.exp(-(full_t - full_t[0]) * γ_peak / 2),
    label="Decaying Cosine",
    alpha=0.5,
)
ax_real.legend()
fig.tight_layout()

if __name__ == "__main__":
    save_figure(fig, "002_small_peak")
