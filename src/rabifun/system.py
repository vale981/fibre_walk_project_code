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
    """Free spectral range of the system in *frequency units*."""

    δ: float = 1 / 4
    """Mode splitting in units of :any:`Ω`."""

    η: float = 0.1
    """Decay rate :math:`\eta/2` of the system in frequency units (no
    :math:`2 \pi`)."""

    d: float = 0.01
    """Drive amplitude in units of :any:`Ω`."""

    Δ: float = 0.0
    """Detuning of the EOM drive in *frequency units*."""

    laser_detuning: float = 0.0
    """Detuning of the laser relative to the _A_ mode."""

    measurement_detuning: float = 0.0
    """Additional detuning of the measurement laser signal relative to the _A_ mode."""

    laser_off_time: float | None = None
    """Time at which the laser is turned off."""

    drive_off_time: float | None = None
    """Time at which the drive is turned off."""

    rwa: bool = False
    """Whether to use the rotating wave approximation."""

    dynamic_detunting: tuple[float, float] = 0, 0
    """
    A tuple of the total amount and the timescale (``1/speed``) of the
    detuning of the laser.
    """

    def periods(self, n: float):
        """
        Returns the number of periods of the system that correspond to
        `n` cycles.
        """
        return n / self.Ω

    def lifetimes(self, n: float):
        """
        Returns the number of lifetimes of the system that correspond to
        `n` cycles.
        """
        return n / self.η

    @property
    def rabi_splitting(self):
        """The Rabi splitting of the system in *frequency units*."""
        return np.sqrt((self.Ω * self.d) ** 2 + self.Δ**2)

    @property
    def ω_eom(self):
        """The frequency of the EOM drive as *angular frequency*."""
        return 2 * np.pi * (self.Ω * (1 - self.δ) - self.Δ)


class RuntimeParams:
    """Secondary Parameters that are required to run the simulation."""

    def __init__(self, params: Params):
        Ωs = 2 * np.pi * params.Ω * np.concatenate(
            [[-1 * params.δ, params.δ], np.arange(1, params.N + 1)]
        ) - 1j * np.repeat(params.η / 2, params.N + 2)

        self.Ωs = Ωs


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


def eom_drive(t, x, d, ω):
    """The electrooptical modulation drive.

    :param t: time
    :param x: amplitudes
    :param d: drive amplitude
    :param ω: drive frequency
    """

    stacked = np.repeat([x], len(x), 0)

    stacked = np.sum(stacked, axis=1)
    driven_x = d * np.sin(ω * t) * stacked

    return driven_x


def laser_frequency(params: Params, t: np.ndarray):
    """The frequency of the laser light as a function of time."""
    base = 2 * np.pi * (params.laser_detuning + params.Ω * params.δ)
    if params.dynamic_detunting[1] == 0:
        return base

    return base + 2 * np.pi * (
        params.dynamic_detunting[0] * t / params.dynamic_detunting[1]
    )


def make_righthand_side(runtime_params: RuntimeParams, params: Params):
    """The right hand side of the equation of motion."""

    def rhs(t, x):
        differential = runtime_params.Ωs * x

        if params.rwa:
            x[0] = 0
            x[3:] = 0

        if (params.drive_off_time is None) or (t < params.drive_off_time):
            differential += eom_drive(
                t, x, 2 * np.pi * params.Ω * params.d, params.ω_eom
            )

        if (params.laser_off_time is None) or (t < params.laser_off_time):
            laser = np.exp(-1j * laser_frequency(params, t) * t)
            differential[0:2] += laser / np.sqrt(2)
            differential[2:] += laser

        if params.rwa:
            differential[0] = 0
            differential[3:] = 0

        return -1j * differential

    return rhs


def solve(t: np.ndarray, params: Params):
    """Integrate the equation of motion.

    :param t: time array
    :param params: system parameters
    """

    runtime = RuntimeParams(params)
    rhs = make_righthand_side(runtime, params)

    initial = np.zeros(params.N + 2, np.complex128)

    return scipy.integrate.solve_ivp(
        rhs,
        (np.min(t), np.max(t)),
        initial,
        vectorized=False,
        # max_step=0.01 * np.pi / (params.Ω * params.N),
        t_eval=t,
    )


def in_rotating_frame(
    t: np.ndarray, amplitudes: np.ndarray, params: Params
) -> np.ndarray:
    """Transform the amplitudes to the rotating frame."""
    Ωs = RuntimeParams(params).Ωs

    return amplitudes * np.exp(1j * Ωs[:, None].real * t[None, :])


def output_signal(t: np.ndarray, amplitudes: np.ndarray, params: Params):
    """
    Calculate the output signal when mixing with laser light of
    frequency `laser_detuning`.
    """
    return (
        np.sum(amplitudes, axis=0)
        * np.exp(
            1j
            * (laser_frequency(params, t) + 2 * np.pi * params.measurement_detuning)
            * t
        )
    ).imag
