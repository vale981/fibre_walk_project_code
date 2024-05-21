from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *
from rabifun.analysis import *
import numpy as np


# %% load data
path = "../../data/08_05_24/characterization_first"
scan = ScanData.from_dir(path)


# %% Fourier
freq, fft = fourier_transform(
    scan.time, scan.output, low_cutoff=1000, high_cutoff=10**8
)

fig = plt.figure("interactive")
fig.clf()
ax, ax2 = fig.subplots(1, 2)
ax.set_yscale("log")
ax.plot(freq * 10 ** (-6), np.abs(fft) ** 2)
plot_scan(scan, ax=ax2, linewidth=0.1)
ax2.set_xlim(0.02, 0.020001)
