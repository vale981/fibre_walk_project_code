from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *

path = "../data/08_05_24/characterization_first"
scan = ScanData.from_dir(path)

# %% interactive
STEPS = [25, 27, 28]
fig = plt.figure("interactive")
fig.clf()
ax, *axs = fig.subplots(1, len(STEPS) + 1)

plot_scan(
    scan, smoothe_output=10 ** (-6), normalize=True, laser=True, steps=True, ax=ax
)

# %% plot steps
for ax, STEP in zip(axs, STEPS):
    data = scan.for_step(step=STEP).smoothed(10**-6)

    t, o, params, cov, scaled = fit_transient(data.time, data.output)

    ax.cla()
    ax.plot(t, o)
    ax.plot(t, transient_model(t, *params))
    ax.set_title(
        f"Transient {STEP}, γ={1/scaled[1] * 10**6:.2f}μs ({cov[1]/scaled[1]**2 * 10**6:.2f}μs)\n ω/2π={scaled[0] / (2*np.pi * 10**3):.5f}kHz\n step={STEP}"
    )

    freq_unit = params[1] / scaled[1]
    ax.plot(t, np.sin(2 * np.pi * 4 * 10**4 * t * freq_unit), alpha=0.1)
