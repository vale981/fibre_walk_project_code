import sys

sys.path.append("../")
from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *

path = (
    "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/24_05_24/nice_transient_2"
)
scan = ScanData.from_dir(path, truncation=[0, 50])

fig, ax = plot_scan(scan, steps=True, smoothe_output=1000)

plt.savefig("../figures/non_steady.png")
