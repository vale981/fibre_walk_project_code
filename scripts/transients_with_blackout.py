import sys

from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *

path = "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/09_05_24/Nicely_hybridised_2 2024,05,09, 15h57min00sec/"
scan = ScanData.from_dir(path, truncation=[0, 50])

# %% interactive
fig = plt.figure("interactive")
fig.clf()
ax, ax_trans = fig.subplots(2, 1)
plot_scan(scan, smoothe_output=10**2, normalize=True, laser=False, steps=True, ax=ax)

time, output, _ = scan.for_step(10)
t, o, params, cov, scaled = fit_transient(time, output, window_size=100)

ax_trans.plot(t, o)
ax_trans.plot(t, transient_model(t, *params))
ax_trans.set_title(
    f"Transient 2, τ_γ={1/scaled[1] * 10**6}μs ω/2π={scaled[1] / (2*np.pi * 10**3)}kHz"
)
