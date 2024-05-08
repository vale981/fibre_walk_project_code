import sys

sys.path.append("../")
from ringfit import data
import matplotlib.pyplot as plt
from ringfit.data import *
from ringfit.plotting import *

path = "/home/hiro/Documents/org/roam/code/fitting_ringdown/data/24_05_24/characterization_first"
scan = data.load_scan(path)

plot_scan(scan, steps=True)
