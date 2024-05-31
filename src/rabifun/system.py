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

    Ω: float = 13
    """Free spectral range of the system in *frequency units*."""

    δ: float = 1 / 4
    """Mode splitting in units of :any:`Ω`."""

    η: float = 0.5
    """Decay rate :math:`\eta/2` of the system in angular frequency units."""

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

    initial_state: np.ndarray | None = None
    """The initial state of the system."""

    correct_lamb_shift: bool = True
    """Whether to correct for the Lamb shift by tweaking the detuning."""

    def __post_init__(self):
        if self.N_couplings > self.N:
            raise ValueError("N_couplings must be less than or equal to N.")

        if self.initial_state and len(self.initial_state) != self.N + 2:
            raise ValueError("Initial state must have length N + 2.")

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
        return 2 * n / self.η

    @property
    def rabi_splitting(self):
        """The Rabi splitting of the system in *frequency units*."""
        if not self.flat_energies:
            raise ValueError("Rabi splitting is only defined for flat energies.")

        return np.sqrt((self.Ω * self.g_0) ** 2 + (self.ω_c * self.Ω) ** 2)


class RuntimeParams:
    """Secondary Parameters that are required to run the simulation."""

    def __init__(self, params: Params):
        freqs = (
            2
            * np.pi
            * params.Ω
            * np.concatenate([[-1 * params.δ, params.δ], np.arange(1, params.N + 1)])
        )

        decay_rates = -1j * np.repeat(params.η / 2, params.N + 2)
        Ωs = freqs + decay_rates

        self.drive_frequencies, self.detunings, self.drive_amplitudes = (
            drive_frequencies_and_amplitudes(params)
        )  # linear frequencies!

        self.Ωs = Ωs
        self.diag_energies = (
            2
            * np.pi
            * np.concatenate(
                [
                    [0, 0],
                    self.detunings,
                    np.zeros(params.N - params.N_couplings),
                ]
            )
            + decay_rates
        )

        self.detuned_Ωs = freqs - self.diag_energies.real

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


def eom_drive(t, x, ds, ωs, rwa, detuned_Ωs):
    """The electrooptical modulation drive.

    :param t: time
    :param x: amplitudes
    :param ds: drive amplitudes
    :param ωs: linear drive frequencies
    """

    ds = 2 * np.pi * ds
    if rwa:
        coupled_indices = 2 + len(ds)
        det_matrix = np.zeros((len(x), len(x)))
        det_matrix[1, 2:coupled_indices] = ds / 2
        det_matrix[2:coupled_indices, 1] = ds / 2
        driven_x = det_matrix @ x
    else:
        det_matrix = detuned_Ωs[:, None] - detuned_Ωs[None, :]
        # test = abs(det_matrix.copy())
        # test[test < 1e-10] = np.inf
        # print(np.min(test))
        # print(np.argmin(test, keepdims=True))

        det_matrix = np.exp(-1j * det_matrix * t)

        driven_x = np.sum(ds * np.sin(2 * np.pi * ωs * t)) * (det_matrix @ x)

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
        differential = runtime_params.diag_energies * x

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
                runtime_params.detuned_Ωs,
            )

        if (params.laser_off_time is None) or (t < params.laser_off_time):
            freqs = laser_frequency(params, t) - runtime_params.detuned_Ωs.real

            laser = np.exp(
                -1j * (laser_frequency(params, t) - runtime_params.detuned_Ωs.real) * t
            )

            if params.rwa:
                index = np.argmin(abs(freqs))
                laser[0:index] = 0
                laser[index + 1 :] = 0

            differential[0:2] += laser[:2] / np.sqrt(2)
            differential[2:] += laser[2:]

        # if params.rwa:
        #     differential[0] = 0
        #     differential[2 + params.N_couplings :] = 0

        return -1j * differential

    return rhs


def make_zero_intial_state(params: Params) -> np.ndarray:
    """Make initial state with all zeros."""
    return np.zeros(params.N + 2, np.complex128)


def solve(t: np.ndarray, params: Params, **kwargs):
    """Integrate the equation of motion.
    The keyword arguments are passed to :any:`scipy.integrate.solve_ivp`.

    :param t: time array
    :param params: system parameters
    """

    runtime = RuntimeParams(params)
    rhs = make_righthand_side(runtime, params)

    initial = (
        make_zero_intial_state(params)
        if params.initial_state is None
        else params.initial_state
    )

    return scipy.integrate.solve_ivp(
        rhs,
        (np.min(t), np.max(t)),
        initial,
        vectorized=False,
        # max_step=2 * np.pi / (params.Ω * params.N_couplings),
        t_eval=t,
        method="DOP853",
        atol=1e-7,
        rtol=1e-4,
        **kwargs,
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

    runtime = RuntimeParams(params)
    rotating = amplitudes * np.exp(-1j * runtime.detuned_Ωs * t)

    return (
        np.sum(rotating, axis=0)
        * np.exp(
            1j
            * (laser_frequency(params, t) + 2 * np.pi * params.measurement_detuning)
            * t
        )
    ).imag


def bath_energies(N_couplings: int, ω_c: float) -> np.ndarray:
    """Return the energies (drive detunings) of the bath modes."""

    return np.arange(1, N_couplings + 1) * ω_c / N_couplings


def ohmic_spectral_density(ω: np.ndarray, α: float) -> np.ndarray:
    """The unnormalized spectral density of an Ohmic bath."""

    return ω**α


def lamb_shift(amplitudes, Δs):
    return np.sum(amplitudes**2 / Δs)


def drive_frequencies_and_amplitudes(
    params: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the linear frequencies and amplitudes of the drives based
    on the ``params``.
    """

    if params.flat_energies:
        Δs = np.repeat(params.ω_c, params.N_couplings)
    else:
        Δs = bath_energies(params.N_couplings, params.ω_c)

    Δs *= params.Ω

    amplitudes = ohmic_spectral_density(
        bath_energies(params.N_couplings, 1),
        params.α,
    )

    amplitudes /= np.sum(amplitudes)
    amplitudes = params.Ω * params.g_0 * np.sqrt(amplitudes)

    if not params.flat_energies and params.correct_lamb_shift:
        Δs -= np.sum(amplitudes**2 / Δs)

    ωs = ((np.arange(1, params.N_couplings + 1) - params.δ) * params.Ω) - Δs
    return ωs, Δs, amplitudes


def mode_name(mode: int):
    """Return the name of the mode."""
    if mode == 0:
        return "anti-A"
    if mode == 1:
        return "A"

    return f"B{mode - 1}"


def a_site_indices(params: Params):
    """Return the indices of the A sites."""
    return [0, 1]


def coupled_bath_mode_indices(params: Params):
    """Return the indices of the bath modes that are coupled to the A site."""
    return np.arange(2, 2 + params.N_couplings)


def uncoupled_bath_mode_indices(params: Params):
    """Return the indices of the bath modes that not coupled to the A site."""
    return np.arange(2 + params.N_couplings, 2 + params.N)


def uncoupled_mode_indices(params: Params):
    """Return the indices of the modes that not coupled to the A site."""
    return np.concatenate(
        [[a_site_indices(params)[0]], uncoupled_bath_mode_indices(params)]
    )


def coupled_mode_indices(params: Params):
    """Return the indices of the modes that are coupled to the A site."""
    return np.concatenate(
        [[a_site_indices(params)[1]], coupled_bath_mode_indices(params)]
    )


def dimension(params: Params):
    """Return the dimension of the system."""
    return params.N + 2


def recurrence_time(params: Params):
    """Return the recurrence time of the system."""
    return params.N_couplings / (params.Ω * params.ω_c)


def solve_nonrwa_rwa(t: np.ndarray, params: Params, **kwargs):
    """
    Solve the system in the non-RWA and RWA cases and return the results.
    The keyword arguments are passed to :any:`solve`.

    :param t: time array
    :param params: system parameters

    :returns: non-RWA and RWA solutions
    """
    initial_rwa = params.rwa

    params.rwa = False
    nonrwa = solve(t, params, **kwargs)
    rwa_params = params
    rwa_params.rwa = True
    rwa = solve(t, rwa_params, **kwargs)

    params.rwa = initial_rwa
    return nonrwa, rwa


def correct_for_decay(solution, params):
    """Correct the ``solution`` for decay.

    :param solution: The solution from :any:`solve_ivp` to correct.
    :param params: The system parameters

    :returns: The corrected solution amplitudes.
    """
    return solution.y * np.exp(params.η / 2 * solution.t[None, :])
