import sys

sys.path.append("../")
from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *

path = "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/09_05_24/Nicely_hybridised_2 2024,05,09, 15h57min00sec/"
scan = ScanData.from_dir(path, truncation=[0, 50])

fig, ax = plot_scan(scan, smoothe_output=50, normalize=True, laser=False, steps=True)
time, output, _ = scan.for_step(10)
t, o, params, cov, scaled = fit_transient(time, output, window_size=100)

plt.figure()
plt.plot(t, o)
plt.plot(t, transient_model(t, *params))
plt.title(
    f"Transient 2, γ={scaled[1] / 10**3}kHz ω/2π={scaled[1] / (2*np.pi * 10**3)}kHz"
)
