import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate


@dataclass
class Params:
    """Parameters for the system."""

    N: int = 2
    """Number of bath modes."""

    N_couplings: int = 2
    """Number of bath modes to couple to.

    To test the RWA it is useful to couple less than the full
    complement of modes.
    """

    Ω: float = 1
    """Free spectral range of the system in *frequency units*."""

    δ: float = 1 / 4
    """Mode splitting in units of :any:`Ω`."""

    η: float = 0.1
    """Decay rate :math:`\eta/2` of the system in frequency units (no
    :math:`2 \pi`)."""

    g_0: float = 0.01
    """Drive amplitude in units of :any:`Ω`."""

    α: float = 0.0
    """The exponent of the spectral density of the bath."""

    ω_c: float = 0.0
    """The cutoff frequency of the bath."""

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

    flat_energies: bool = False
    """Whether to use a flat distribution of bath energies."""

    def __post_init__(self):
        if self.N_couplings > self.N:
            raise ValueError("N_couplings must be less than or equal to N.")

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
        if not self.flat_energies:
            raise ValueError("Rabi splitting is only defined for flat energies.")

        return np.sqrt((self.Ω * self.g_0) ** 2 + (self.ω_c * self.Ω) ** 2)


class RuntimeParams:
    """Secondary Parameters that are required to run the simulation."""

    __slots__ = ["Ωs", "drive_frequencies", "drive_amplitudes"]

    def __init__(self, params: Params):
        Ωs = 2 * np.pi * params.Ω * np.concatenate(
            [[-1 * params.δ, params.δ], np.arange(1, params.N + 1)]
        ) - 1j * np.repeat(params.η / 2, params.N + 2)

        self.Ωs = Ωs

        self.drive_frequencies, self.drive_amplitudes = (
            drive_frequencies_and_amplitudes(params)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(Ωs={self.Ωs}, drive_frequencies={self.drive_frequencies}, drive_amplitudes={self.drive_amplitudes})"


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


def eom_drive(t, x, ds, ωs, rwa):
    """The electrooptical modulation drive.

    :param t: time
    :param x: amplitudes
    :param ds: drive amplitudes
    :param ωs: linear drive frequencies
    """

    stacked = np.repeat([x], len(x), 0)

    if rwa:
        stacked[1, 2 : 2 + len(ωs)] *= ds * np.exp(-1j * 2 * np.pi * ωs * t)
        stacked[2 : 2 + len(ωs), 1] *= ds * np.exp(1j * 2 * np.pi * ωs * t)
        driven_x = np.sum(stacked, axis=1)
    else:
        stacked = np.sum(stacked, axis=1)
        driven_x = np.sum(ds * np.sin(2 * np.pi * ωs * t)) * stacked

    return driven_x


def laser_frequency(params: Params, t: np.ndarray):
    """The frequency of the laser light as a function of time."""
    base = 2 * np.pi * (params.laser_detuning + params.Ω * params.δ)
    if params.dynamic_detunting[1] == 0:
        if np.isscalar(t):
            return base

        return np.repeat(base, len(t))

    return base + 2 * np.pi * (
        params.dynamic_detunting[0] * t / params.dynamic_detunting[1]
    )


def make_righthand_side(runtime_params: RuntimeParams, params: Params):
    """The right hand side of the equation of motion."""

    def rhs(t, x):
        differential = runtime_params.Ωs * x

        if params.rwa:
            x[0] = 0
            x[2 + params.N_couplings :] = 0

        if (params.drive_off_time is None) or (t < params.drive_off_time):
            differential += eom_drive(
                t,
                x,
                runtime_params.drive_amplitudes,
                runtime_params.drive_frequencies,
                params.rwa,
            )

        if (params.laser_off_time is None) or (t < params.laser_off_time):
            laser = np.exp(-1j * laser_frequency(params, t) * t)
            differential[0:2] += laser / np.sqrt(2)
            differential[2:] += laser

        if params.rwa:
            differential[0] = 0
            differential[2 + params.N_couplings :] = 0

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
        max_step=0.01 * np.pi / (params.Ω * params.N),
        t_eval=t,
    )


def in_rotating_frame(
    t: np.ndarray, amplitudes: np.ndarray, params: Params
) -> np.ndarray:
    """Transform the amplitudes to the rotating frame."""
    Ωs = RuntimeParams(params).Ωs

    detunings = np.concatenate(
        [[0, 0], drive_detunings(params), np.zeros(params.N - params.N_couplings)]
    )

    return amplitudes * np.exp(
        1j
        * (Ωs[:, None].real - detunings[:, None] + laser_frequency(params, t)[None, :])
        * t[None, :]
    )


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


def bath_energies(params: Params) -> np.ndarray:
    """Return the energies (drive detunings) of the bath modes."""

    return np.arange(1, params.N_couplings + 1) * params.ω_c / params.N_couplings


def ohmic_spectral_density(
    ω: np.ndarray, g_0: float, α: float, ω_c: float
) -> np.ndarray:
    """The spectral density of an Ohmic bath."""

    mask = (ω > ω_c) | (ω < 0)

    ret = np.empty_like(ω)
    ret[mask] = 0
    ret[~mask] = g_0**2 * (α + 1) / (ω_c ** (α + 1)) * (ω**α)

    return ret


def drive_detunings(params: Params) -> np.ndarray:
    """Return the drive detunings of the bath modes in frequency units."""

    if params.flat_energies:
        Δs = np.repeat(params.ω_c, params.N_couplings)
    else:
        Δs = bath_energies(params)

    return Δs * params.Ω


def drive_frequencies_and_amplitudes(params: Params) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the linear frequencies and amplitudes of the drives based
    on the ``params``.
    """

    Δs = drive_detunings(params)
    if params.flat_energies:
        amplitudes = np.ones_like(Δs)
    else:
        amplitudes = (
            ohmic_spectral_density(
                Δs, params.g_0 * params.Ω, params.α, params.ω_c * params.Ω
            )
            * params.ω_c
            * params.Ω
            / Params.N_couplings
        )

    amplitudes /= np.sum(amplitudes)
    amplitudes = 2 * np.pi * params.Ω * params.g_0 * np.sqrt(amplitudes)

    ωs = ((np.arange(1, params.N_couplings + 1) - params.δ) * params.Ω) - Δs
    return ωs, amplitudes
