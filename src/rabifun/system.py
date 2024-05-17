import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate


@dataclass
class Params:
    """Parameters for the system."""

    N: int = 2
    """Number of bath modes."""

    Ω: float = 1
    """Free spectral range of the system."""

    η: float = 0.1
    """Decay rate of the system."""

    d: float = 0.01
    """Drive amplitude."""

    Δ: float = 0.0
    """Detuning of the EOM drive."""

    laser_detuning: float = 0.0
    """Detuning of the laser relative to the _A_ mode."""

    laser_off_time: float | None = None
    """Time at which the laser is turned off."""

    drive_off_time: float | None = None
    """Time at which the drive is turned off."""

    def periods(self, n: float):
        return n * 2 * np.pi / self.Ω

    def lifetimes(self, n: float):
        return n / self.η

    @property
    def rabi_splitting(self):
        return np.sqrt(self.d**2 + self.Δ**2)


class RuntimeParams:
    """Secondary Parameters that are required to run the simulation."""

    def __init__(self, params: Params):
        self.Ωs = np.arange(0, params.N) * params.Ω - 1j * np.repeat(params.η, params.N)


def time_axis(params: Params, lifetimes: float, resolution: float = 1):
    """Generate a time axis for the simulation.

    :param params: system parameters
    :param lifetimes: number of lifetimes to simulate
    :param resolution: time resolution

        Setting this to `1` will give the time axis just enough
        resolution to capture the fastest relevant frequencies.  A
        smaller value yields more points in the time axis.
    """

    return np.arange(
        0, params.lifetimes(lifetimes), resolution * np.pi / (params.Ω * params.N)
    )


def eom_drive(t, x, d, Δ, Ω):
    """The electrooptical modulation drive.

    :param t: time
    :param x: amplitudes
    :param d: drive amplitude
    :param Δ: detuning
    :param Ω: FSR
    """
    stacked = np.repeat([x], len(x), 0)

    np.fill_diagonal(stacked, 0)

    stacked = np.sum(stacked, axis=1)
    driven_x = d * np.sin((Ω - Δ) * t) * stacked

    return driven_x


def make_righthand_side(runtime_params: RuntimeParams, params: Params):
    """The right hand side of the equation of motion."""

    def rhs(t, x):
        differential = runtime_params.Ωs * x

        if (params.drive_off_time is None) or (t < params.drive_off_time):
            differential += eom_drive(t, x, params.d, params.Δ, params.Ω)

        if (params.laser_off_time is None) or (t < params.laser_off_time):
            differential += np.exp(-1j * params.laser_detuning * t)

        return -1j * differential

    return rhs


def solve(t: np.ndarray, params: Params):
    """Integrate the equation of motion.

    :param t: time array
    :param params: system parameters
    """

    runtime = RuntimeParams(params)
    rhs = make_righthand_side(runtime, params)

    initial = np.zeros(params.N, np.complex128)

    return scipy.integrate.solve_ivp(
        rhs,
        (np.min(t), np.max(t)),
        initial,
        vectorized=False,
        max_step=0.01 * 2 * np.pi / np.max(abs(runtime.Ωs.real)),
        t_eval=t,
    )


def in_rotating_frame(
    t: np.ndarray, amplitudes: np.ndarray, params: Params
) -> np.ndarray:
    """Transform the amplitudes to the rotating frame."""
    Ωs = RuntimeParams(params).Ωs

    return amplitudes * np.exp(1j * Ωs[:, None].real * t[None, :])


def output_signal(t: np.ndarray, amplitudes: np.ndarray, laser_detuning: float):
    """
    Calculate the output signal when mixing with laser light of
    frequency `laser_detuning`.
    """
    return (np.sum(amplitudes, axis=0) * np.exp(1j * laser_detuning * t)).imag
