import sys

from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *

path = (
    "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/08_05_24/nice_transient_2"
)
scan = ScanData.from_dir(path, truncation=[0, 50])

# %% interactive
STEPS = [2, 3, 5]
fig = plt.figure("interactive")
ax, *axs = fig.subplots(1, len(STEPS))
plot_scan(scan, smoothe_output=500, normalize=True, laser=True, steps=True, ax=ax)

for ax, STEP in zip(axs, STEPS):
    time, output, _ = scan.for_step(step=STEP)
    t, o, params, cov, scaled = fit_transient(time, output, window_size=100)

    ax.plot(t, o)
    ax.plot(t, transient_model(t, *params))
    ax.set_title(
        f"Transient 2, γ={scaled[1] / 10**3:.2f}kHz ({cov[1] / 10**3:.2f}kHz)\n ω/2π={scaled[0] / (2*np.pi * 10**3):.5f}kHz\n step={STEP}"
    )

    freq_unit = params[1] / scaled[1]
    ax.plot(t, np.sin(2 * np.pi * 4 * 10**4 * t * freq_unit), alpha=0.1)
