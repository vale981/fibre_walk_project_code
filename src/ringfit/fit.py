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


def fit_transient(time: np.ndarray, transient: np.ndarray, window_size: int = 100):
    """
    Fit a transient signal ``transient`` over ``time`` to a damped
    oscillation model.

    The smoothing window is calculated as the length of the transient
    divided by the ``window_size``.
    """

    # data_length = len(transient)
    # begin, end = transient.argmax(), -int(data_length * 0.01)

    # time = time[begin:end]
    # output_data = transient[begin:end]

    output_data = transient
    window = len(output_data) // window_size
    output_data = uniform_filter1d(output_data, window)
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
