import sys

sys.path.append("../")
from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *

path = (
    "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/08_05_24/nice_transient_2"
)
scan = ScanData.from_dir(path, truncation=[0, 50])

STEPS = [2, 3, 5]
fig, ax = plot_scan(scan, smoothe_output=50, normalize=True, laser=False, steps=True)

for STEP in STEPS:
    time, output, _ = scan.for_step(step=STEP)
    t, o, params, cov, scaled = fit_transient(time, output, window_size=100)

    plt.figure()
    plt.plot(t, o)
    plt.plot(t, transient_model(t, *params))
    plt.title(
        f"Transient 2, γ={scaled[1] / 10**3:.2f}kHz ({cov[1] / 10**3:.2f}kHz)\n ω/2π={scaled[0] / (2*np.pi * 10**3):.5f}kHz\n step={STEP}"
    )

    freq_unit = params[1] / scaled[1]
    plt.plot(t, np.sin(2 * np.pi * 4 * 10**4 * t * freq_unit), alpha=0.1)
