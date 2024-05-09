import sys

sys.path.append("../")
from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *
from ringfit.utils import *

path = "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/09_05_24/Nicely_hybridised_2 2024,05,09, 15h57min00sec/"
scan = ScanData.from_dir(path, truncation=[0, 50])
STEPS = [2, 33, 12]
fig, (ax1, *axs) = plt.subplots(nrows=1, ncols=len(STEPS) + 1)

plot_scan(scan, smoothe_output=100, normalize=True, laser=False, steps=True, ax=ax1)


def fit_frequency(step, ax):
    time, output, _ = scan.for_step(step)
    l = len(time)
    begin = int(0.5 * l)
    end = int(0.8 * l)
    time = time[begin:end]
    output = output[begin:end]
    output = smoothe_signal(output, 0.05)
    output = shift_and_normalize(output)
    output -= output.mean()

    ax.plot(time, output)
    ff = np.fft.rfftfreq(output.size, d=time[1] - time[0])
    ft = np.fft.rfft(output)

    freq_index = np.argmax(ft)

    ax.plot(
        time,
        0.5 * np.sin(time * 2 * np.pi * ff[freq_index] + np.angle(ft[freq_index])),
    )
    ax.set_title(f"f={ff[freq_index]*10**(-3):.2f}kHz\n step={step}")


for step, ax in zip(STEPS, axs):
    fit_frequency(step, ax)
