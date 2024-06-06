from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *
from ringfit.fit import *
from rabifun.analysis import *
from rabifun.plots import *

# %% setup figure
fig = make_figure("calibration")
ax_spectrum = fig.subplots(1, 1)


# %% load data
path = "../../data/22_05_24/ringdown_try_2"
scan = ScanData.from_dir(path)


# %% Set Window
window = tuple(
    np.array([0.016244684251065847 + 0.000002, 0.016248626903395593 + 49e-5])
    + 8e-3
    - 12e-7
)

ringdown_params = RingdownParams(fω_shift=10e4, mode_window=(0, 50))
peak_info = find_peaks(scan, ringdown_params, window, prominence=0.008)


peak_info = refine_peaks(peak_info, ringdown_params)
plot_spectrum_and_peak_info(ax_spectrum, peak_info, ringdown_params)

# extract
extract_Ω_δ(peak_info, ringdown_params)
