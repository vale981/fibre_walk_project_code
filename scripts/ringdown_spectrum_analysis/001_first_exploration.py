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
import pickle

# %% load data
path = "../../data/22_05_24/ringdown_try_2"
scan = ScanData.from_dir(path, extension="npz")


# %% Set Window
window = (0.027751370026589985, 0.027751370026589985 + 0.00001 / 2)
window = tuple(
    np.array([0.03075207891902308, 0.03075207891902308 + 0.00001]) + 4e-3 - 0.87e-6
)

window = tuple(
    np.array([0.016244684251065847 + 0.000002, 0.016248626903395593 + 49e-5])
    + 8e-3
    - 12e-7
)


# %% Plot Scan
gc.collect()
fig = plt.figure("interactive", constrained_layout=True, figsize=(20, 3 * 5))
fig.clf()
(ax, ax2, ax_signal, ax_stft, ax_decay) = fig.subplot_mosaic("AB;CC;DE").values()
ax.set_title("Fourier Spectrum")
ax2.set_title("Reconstructed Spectrum")
for spec_ax in [ax, ax2]:
    spec_ax.set_xlabel("Frequency (MHz)")
    spec_ax.set_ylabel("Power")
ax3 = ax.twinx()
ax3.set_ylabel("Phase (rad)")

ax_stft.set_xlabel("Time (s)")
ax_stft.set_ylabel("Frequency (Hz)")
ax_stft.set_title("Short Time Fourier Transform")
ax_decay.set_xlabel("Time (s)")
ax_decay.set_ylabel("Power")

# ax_signal.set_xlim(*window)
plot_scan(scan, ax=ax_signal, smoothe_output=1e-8, linewidth=0.5)
ax_signal.axvspan(*window, color="red", alpha=0.1)
ax_signal.set_xlabel("Time (s)")
ax_signal.set_ylabel("Signal (mV)")
ax_signal.set_title("Raw Signal (Slighly Smoothened)")

# %% Fourier
freq, fft = fourier_transform(
    scan.time, scan.output, window=window, low_cutoff=0.5e6, high_cutoff=90e6
)

freq *= 1e-6

# ax.set_yscale("log")
ax.plot(freq, np.abs(fft))
# ax.plot(freq, np.abs(fft.real), linewidth=1, color="red")
# ax.plot(freq, fft.imag, linewidth=1, color="green")

ax3.plot(
    freq[1:],
    np.cumsum(np.angle(fft[1:] / fft[:-1])),
    linestyle="--",
    alpha=0.5,
    linewidth=0.5,
    zorder=10,
)

freq_step = freq[1] - freq[0]
Ω_guess = 13
δ_guess = 2.6

peaks, peak_info = scipy.signal.find_peaks(
    np.abs(fft) ** 2, distance=δ_guess / 2 / freq_step, prominence=1e-8
)

peak_freq = freq[peaks]
anglegrad = np.gradient(np.unwrap(np.angle(fft) + np.pi, period=2 * np.pi))
neg_peaks = peaks[anglegrad[peaks] < 0]
pos_peaks = peaks[anglegrad[peaks] > 0]
phase_detuning = np.angle(fft[peaks])

ax.plot(peak_freq, np.abs(fft[peaks]), "*")


def extract_peak(index, width, sign, detuning):
    begin = max(index - width, 0)
    return sign * (freq[begin : index + width]) + detuning, np.abs(
        fft[begin : index + width]
    )


mode_freqs = freq[peaks]

all_diffs = np.abs((mode_freqs[:, None] - mode_freqs[None, :])[:, :, None] - Ω_guess)

all_diffs[all_diffs == 0] = np.inf
all_diffs[all_diffs > 1] = np.inf
matches = np.asarray(all_diffs < np.inf).nonzero()

pairs = np.array(list(zip(*matches, all_diffs[matches])), dtype=int)

relationships = nx.DiGraph()
for node, peak in enumerate(peaks):
    relationships.add_node(node, freqency=freq[peak])

for left, right, relationship, diff in pairs:
    if freq[left] > freq[right]:
        left, right = right, left

    if (
        not relationships.has_edge(left, right)
        or relationships[left][right]["weight"] > diff * 1e-6
    ):
        relationships.add_edge(
            left, right, weight=diff * 1e-6, type=relationship, freqdis=right - left
        )


UG = relationships.to_undirected()

# extract subgraphs
neg, pos, *unmatched = [
    list(sorted(i))
    for i in sorted(list(nx.connected_components(UG)), key=lambda l: -len(l))
]

ax.plot(mode_freqs[neg], np.abs(fft[peaks[neg]]), "x")
ax.plot(mode_freqs[pos], np.abs(fft[peaks[pos]]), "o")

# ax.plot(freq[pos_peaks], np.abs(fft[pos_peaks]), "o")

Ω = (np.diff(peak_freq[neg]).mean() + np.diff(peak_freq[pos]).mean()) / 2
ΔΩ = np.sqrt((np.diff(peak_freq[neg]).var() + np.diff(peak_freq[pos]).var())) / 2


Δ_L = ((mode_freqs[pos] - mode_freqs[neg] - Ω) / 2).mean()


ax2.cla()
for peak in neg:
    ax2.plot(*extract_peak(peaks[peak], 200, 1, Δ_L + Ω / 2), color="blue")
for peak in pos:
    ax2.plot(*extract_peak(peaks[peak], 200, -1, Δ_L - Ω / 2), color="blue")

hybrid = []
for peak, sign in zip(np.array(unmatched).flatten(), [1, -1]):
    hybrid.append(sign * mode_freqs[peak] + Δ_L)
    ax2.plot(*extract_peak(peaks[peak], 200, sign, Δ_L), color="green")

δ = np.abs(np.diff(hybrid)[0] / 2)

fig.suptitle(f"Ω = {Ω:.2f}MHz, ΔΩ = {ΔΩ:.2f}MHz, Δ_L = {Δ_L:.2f}MHz, δ = {δ:.2f}MHz")

# %% Windowed Fourier
windows = np.linspace(window[0], window[0] + (window[-1] - window[0]) * 0.1, 100)
fiducial = peak_freq[neg[1]]
size = int(300 * 1e-6 / fiducial / scan.timestep)
w_fun = scipy.signal.windows.gaussian(size, std=0.1 * size / 2, sym=True)
# w_fun = scipy.signal.windows.boxcar(size)
amps = []
SFT = scipy.signal.ShortTimeFFT(
    w_fun, hop=int(size * 0.1 / 5), fs=1 / scan.timestep, scale_to="magnitude"
)

t = scan.time[(window[1] > scan.time) & (scan.time > window[0])]
ft = SFT.spectrogram(scan.output[(window[1] > scan.time) & (scan.time > window[0])])
ft[ft > 1e-2] = 0
ax_stft.imshow(
    np.log((ft[:, :400])),
    aspect="auto",
    origin="lower",
    cmap="magma",
    extent=SFT.extent(len(t)),
)
ax_stft.set_ylim(0, 50 * 1e6)
ax_stft.set_xlim(
    2.8 * SFT.lower_border_end[0] * SFT.T, SFT.upper_border_begin(len(t))[0] * SFT.T
)

# %% Decay Plot
index = np.argmin(np.abs(SFT.f - 1e6 * peak_freq[unmatched[1]])) + 1
ax_decay.clear()
ax_stft.axhline(SFT.f[index], linestyle="--", alpha=0.5)

hy_mode = np.mean(ft[index - 3 : index + 3, :], axis=0)
sft_t = SFT.t(len(t))

mask = (sft_t > 1.1 * SFT.lower_border_end[0] * SFT.T) & (sft_t < np.max(sft_t) * 0.1)
hy_mode = hy_mode[mask]
sft_t = sft_t[mask]

ax_decay.plot(sft_t, hy_mode)
# ax_decay.set_xscale("lin")
# plt.plot(sft_t, 3e-6 * np.exp(-.9e6 * (sft_t - 3*SFT.lower_border_end[0] * SFT.T)))


def model(t, a, τ):
    return a * np.exp(-τ * (t - SFT.lower_border_end[0] * SFT.T))


p, cov = scipy.optimize.curve_fit(model, sft_t, hy_mode, p0=[hy_mode[0], 1e6])
ax_decay.plot(sft_t, model(sft_t, *p))
print(p[1] * 1e-6, np.sqrt(np.diag(cov))[1] * 1e-6)
ax_decay.set_title(f"A Site decay γ = {p[1] * 1e-6:.2f}MHz")
ax_decay.set_yscale("log")
# %% save
if __name__ == "__main__":
    save_figure(fig, "001_overview")

    quick_save_pickle(
        dict(
            window=window,
            freq=freq,
            fft=fft,
            peaks=peaks,
            pos=pos,
            neg=neg,
            unmatched=unmatched,
        ),
        "001_results",
    )
