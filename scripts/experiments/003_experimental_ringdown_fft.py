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
path = "../../data/22_05_24/ringdown_with_hybridized_modes"
scan = ScanData.from_dir(path)


# %% Fourier

# window = (0.027751370026589985, 0.027751370026589985 + 0.00001 / 2)
window = tuple(
    np.array([0.03075207891902308, 0.03075207891902308 + 0.00001]) - 1e-3 - 0.82e-6
)
freq, fft = fourier_transform(
    scan.time, scan.output, window=window, low_cutoff=20e5, high_cutoff=100e6
)


# %% plot
gc.collect()
fig = plt.figure("interactive")
fig.clf()
ax, ax2 = fig.subplots(1, 2)
# ax.set_yscale("log")
ax.plot(freq, np.abs(fft))
ax3 = ax.twinx()
ax.plot(freq, np.abs(fft.real), linewidth=1, color="red")

# ax.plot(freq, fft.imag, linewidth=1, color="green")
# ax3.plot(
#     freq,
#     np.gradient(np.unwrap(np.angle(fft) + np.pi, period=2 * np.pi)),
#     linestyle="--",
# )

ax2.set_xlim(*window)


# plot_scan(scan, ax=ax2, linewidth=0.5, smoothe_output=1e-8)

freq_step = freq[1] - freq[0]

peaks, peak_info = scipy.signal.find_peaks(
    np.abs(fft) ** 2, distance=2e6 / freq_step, prominence=3000
)

anglegrad = np.gradient(np.unwrap(np.angle(fft) + np.pi, period=2 * np.pi))
neg_peaks = peaks[anglegrad[peaks] < 0]
pos_peaks = peaks[anglegrad[peaks] > 0]
ax.plot(freq[neg_peaks], np.abs(fft[neg_peaks]), "x")
ax.plot(freq[pos_peaks], np.abs(fft[pos_peaks]), "o")

# phase_detuning = np.angle(fft[peaks])


def extract_peak(index, width, sign=1):
    return sign * freq[index - width : index + width], np.abs(
        fft[index - width : index + width]
    )


# for peak in neg_peaks:
#     ax2.plot(*extract_peak(peak, 10, -1))
# for peak in pos_peaks:
#     ax2.plot(*extract_peak(peak, 10, 1))


Ω_guess = 13e6
δ_guess = 2.69e6

N = np.arange(1, 3)
possibilieties = np.concatenate([[2 * δ_guess], N * Ω_guess, N * Ω_guess - δ_guess])


abs(
    np.array(
        [Ω_guess, 2 * δ_guess, Ω_guess - δ_guess, 2 * Ω_guess, 2 * Ω_guess - δ_guess]
    )
)

mode_freqs = freq[peaks]

final_freqs = [mode_freqs[0]]


all_diffs = np.abs(
    np.abs(mode_freqs[:, None] - mode_freqs[None, :])[:, :, None]
    - possibilieties[None, None, :]
)

all_diffs[all_diffs == 0] = np.inf
all_diffs[all_diffs > δ_guess / 2] = np.inf
matches = np.asarray(all_diffs < np.inf).nonzero()

pairs = np.array(list(zip(*matches, all_diffs[matches])), dtype=int)

relationships = nx.DiGraph()
for node, peak in enumerate(peaks):
    relationships.add_node(node, freq=freq[peak])

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


pos = {}
for node, peak in enumerate(peaks):
    pos[node] = [freq[peak], abs(fft[peak])]
# nx.draw(relationships, pos, with_labels=True)
# nx.draw_networkx_edge_labels(relationships, pos)

# %%
cycle = nx.find_cycle(relationships, orientation="ignore")

total = 0
for s, t, direction in cycle:
    difference = possibilieties[relationships[s][t]["type"]]
    if direction == "reverse":
        difference *= -1
    total += difference

# %%
relationships.remove_nodes_from(list(nx.isolates(relationships)))

spectrum = list(
    sorted(nx.all_simple_paths(relationships, 1, 9), key=lambda x: -len(x))
)[0]

for s, t in zip(spectrum[:-1], spectrum[1:]):
    print(s, relationships[s][t])

for node in spectrum:
    plt.plot(freq[node], abs(fft[node]), "*")
