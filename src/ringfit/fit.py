###############################################################################
#                               Fitting Routine                               #
###############################################################################
import numpy as np
import math
import scipy.optimize as opt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def transient_model(t, Δω, γ, amplitude, phase):
    comp_phase = np.exp(1j * phase)
    osci = amplitude * comp_phase * (np.expm1((-1j * Δω - γ) * (t)))

    return np.imag(osci)


def fit_transient(time: np.ndarray, transient: np.ndarray):
    """
    Fit a transient signal ``transient`` over ``time`` to a damped
    oscillation model.
    """

    output_data = transient
    output_data -= output_data[0]
    output_data /= abs(output_data).max()

    scaled_time = np.linspace(0, 1, len(time))

    popt, pcov = opt.curve_fit(
        transient_model,
        scaled_time,
        output_data,
        p0=[1, 1, 1, 0],
        bounds=(
            [-np.inf, 0, 0, -np.pi],
            [np.inf, np.inf, np.pi, np.inf],
        ),
    )

    Δω, γ, amplitude, phase = popt

    # convert back to units
    Δω = Δω / (time[-1] - time[0])
    γ = γ / (time[-1] - time[0])

    return scaled_time, output_data, popt, np.sqrt(np.diag(pcov)), (Δω, γ)
