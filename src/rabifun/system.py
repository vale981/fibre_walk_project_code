from pdb import run
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
    """Decay rate :math:`\eta/2` of the system in linear frequency units.

    Squared amplitudes decay like :math:`exp(-2π η t)`.
    """

    η_hybrid: float | None = None
    """Decay rate :math:`\eta/2` of the hybridized modes. If ``None``, the rate :any:`η` will be used."""

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

    correct_lamb_shift: float = 1
    """Whether to correct for the Lamb shift by tweaking the detuning."""

    drive_override: tuple[np.ndarray, np.ndarray] | None = None
    """
    Override the drive frequencies (first array, linear frequency) and
    amplitudes (second array, linear frequency units).

    In this case the detunings of the rotating frame will be zeroed
    and the parameters :any:`α` and :any:`ω_c` will be ignored.

    The drive strength is normalized to :any:`g_0`.
    """

    small_loop_detuning: float = 0
    """The detuning (in units of :any:`Ω`) of the small loop mode relative to the ``A`` mode."""

    def __post_init__(self):
        if self.N_couplings > self.N:
            raise ValueError("N_couplings must be less than or equal to N.")

        if self.initial_state and len(self.initial_state) != 2 * self.N + 2:
            raise ValueError("Initial state must have length 2N + 2.")

        if self.drive_override is not None:
            if len(self.drive_override) != 2:
                raise ValueError("Drive override must be a tuple of two arrays.")

            if len(self.drive_override[0]) != len(self.drive_override[1]):
                raise ValueError(
                    "Drive frequencies and amplitudes must have the same length."
                )

            if self.rwa:
                raise ValueError("Drive override is not compatible with the RWA.")

        if self.η_hybrid is None:
            self.η_hybrid = self.η

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
        return 2 * n / (self.η * np.pi * 2)

    @property
    def rabi_splitting(self):
        """The Rabi splitting of the system in *frequency units*."""
        if not self.flat_energies:
            raise ValueError("Rabi splitting is only defined for flat energies.")

        return np.sqrt((self.Ω * self.g_0 * 2) ** 2 + (self.ω_c * self.Ω) ** 2)


class RuntimeParams:
    """Secondary Parameters that are required to run the simulation."""

    def __init__(self, params: Params):
        self.params = params

        H_A = np.array(
            [
                [0, params.δ],
                [params.δ, params.small_loop_detuning],
            ]
        )

        eig = np.linalg.eigh(H_A)
        idx = np.argsort(eig.eigenvalues)
        anti_a_frequency, a_frequency = eig.eigenvalues[idx]
        self.a_weights = np.abs(eig.eigenvectors[:, idx][0, :])

        bath = np.arange(1, params.N + 1)
        freqs = (
            2
            * np.pi
            * params.Ω
            * (
                np.concatenate([[anti_a_frequency, a_frequency], bath, -bath])
                - a_frequency
                + params.δ
            )
        )

        decay_rates = -1j * np.repeat(np.pi * params.η, 2 * params.N + 2)
        if params.η_hybrid:
            decay_rates[[0, 1]] = -1j * np.pi * np.repeat(params.η_hybrid, 2)

        Ωs = freqs + decay_rates

        self.drive_frequencies, self.detunings, self.g, a_shift = (
            drive_frequencies_and_amplitudes(params)
        )  # linear frequencies!
        if params.drive_override is not None:
            self.drive_frequencies = params.drive_override[0]
            self.g = params.drive_override[1]
            a_shift = 0
            self.detunings *= 0

            norm = np.sqrt(np.sum(self.g**2))

            if norm > 0:
                self.g *= params.g_0

        self.g *= 2 * np.pi
        self.Ωs = Ωs

        self.ε = (
            2
            * np.pi
            * np.concatenate(
                [
                    [-a_shift, a_shift],
                    self.detunings,
                    np.zeros(params.N - params.N_couplings),
                    -self.detunings,
                    np.zeros(params.N - params.N_couplings),
                ]
            )
            + decay_rates
        )

        self.bath_ε = 2 * np.pi * self.detunings - 1j * 2 * np.pi * params.η / 2

        self.a_shift = 2 * np.pi * a_shift
        self.detuned_Ωs = freqs - self.ε.real

        self.RWA_H = np.zeros((2 * params.N + 2, 2 * params.N + 2), np.complex128)
        if not params.drive_override:
            self.RWA_H[1, 2 : 2 + params.N_couplings] = self.g
            self.RWA_H[2 : 2 + params.N_couplings, 1] = np.conj(self.g)

        self.detuning_matrix = self.detuned_Ωs[:, None] - self.detuned_Ωs[None, :]

    @property
    def mode_splitting(self):
        """The mode splitting of the system in *frequency units*."""
        return (self.Ωs[1] - self.Ωs[0]).real / (4 * np.pi)

    @property
    def ringdown_frequencies(self) -> np.ndarray:
        """
        The frequencies that are detectable in the ringdown spectrum.
        In essence, those are the eigenfrequencies of the system,
        shifted by laser detuning and measurement detuning.
        """

        return np.abs(
            self.params.measurement_detuning
            - self.Ωs.real / (2 * np.pi)
            + self.params.δ * self.params.Ω
            + self.params.laser_detuning
        )


def time_axis(
    params: Params,
    lifetimes: float | None = None,
    recurrences: float | None = None,
    total_time: float | None = None,
    resolution: float = 1,
):
    """Generate a time axis for the simulation.

    :param params: system parameters
    :param lifetimes: number of lifetimes to simulate
    :params total_time: the total timespan to set of the time array
    :param resolution: time resolution

        Setting this to `1` will give the time axis just enough
        resolution to capture the fastest relevant frequencies.  A
        smaller value yields more points in the time axis.
    """

    tmax = 0
    if lifetimes is not None:
        tmax = params.lifetimes(lifetimes)
    elif recurrences is not None:
        tmax = recurrence_time(params) * recurrences
    elif total_time:
        tmax = total_time
    else:
        raise ValueError("Either lifetimes or recurrences must be set.")

    return np.arange(0, tmax, resolution * np.pi / (params.Ω * params.N))


def eom_drive(t, x, ds, ωs, det_matrix, a_weights):
    """The electrooptical modulation drive.

    :param t: time
    :param x: amplitudes
    :param ds: drive amplitudes
    :param ωs: linear drive frequencies
    :param det_matrix: detuning matrix
    :param a_weights: weights of the A modes
    """

    # test = abs(det_matrix.copy())
    # test[test < 1e-10] = np.inf
    # print(np.min(test))
    # print(np.argmin(test, keepdims=True))

    rot_matrix = np.exp(-1j * det_matrix * t)
    for i, weight in enumerate(a_weights):
        rot_matrix[i, 2:] *= weight
        rot_matrix[2:, i] *= weight.conjugate()

    # # FIXME: that's not strictly right for the non symmetric damping
    prod = a_weights[0] * a_weights[1].conj()
    rot_matrix[0, 1] *= prod
    rot_matrix[1, 0] *= prod.conjugate()

    driven_x = np.sum(2 * ds * np.sin(2 * np.pi * ωs * t)) * (rot_matrix @ x)

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
        differential = runtime_params.ε * x

        if params.rwa:
            x[0] = 0
            x[2 + params.N_couplings :] = 0

        if (params.drive_off_time is None) or (t < params.drive_off_time):
            if params.rwa:
                differential += runtime_params.RWA_H @ x
            else:
                differential += eom_drive(
                    t,
                    x,
                    runtime_params.g,
                    runtime_params.drive_frequencies,
                    runtime_params.detuning_matrix,
                    runtime_params.a_weights,
                )
        if (params.laser_off_time is None) or (t < params.laser_off_time):
            freqs = laser_frequency(params, t) - runtime_params.detuned_Ωs.real

            laser = np.exp(-1j * freqs * t)

            if params.rwa:
                index = np.argmin(abs(freqs))
                laser[0:index] = 0
                laser[index + 1 :] = 0

            differential[0:2] += laser[:2] * runtime_params.a_weights
            differential[2:] += laser[2:]

        if params.rwa:
            differential[0] = 0
            differential[2 + params.N_couplings :] = 0

        return -1j * differential

    return rhs


def make_zero_intial_state(params: Params) -> np.ndarray:
    """Make initial state with all zeros."""
    return np.zeros(2 * params.N + 2, np.complex128)


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
        max_step=1 / 2 * np.pi / (params.Ω * (2 * params.N + 2)),
        t_eval=t,
        method="DOP853",
        atol=1e-7,
        rtol=1e-4,
        **kwargs,
    )


def output_signal(t: np.ndarray, amplitudes: np.ndarray, params: Params):
    """
    Calculate the output signal when mixing with laser light of
    frequency `laser_detuning`.
    """

    runtime = RuntimeParams(params)

    laser_times = (
        laser_frequency(params, t) + 2 * np.pi * params.measurement_detuning
    ) * t
    rotating = amplitudes.copy() * np.exp(
        -1j * (runtime.detuned_Ωs[:, None] * t[None, :] - laser_times[None, :])
    )

    rotating[0:2, :] *= runtime.a_weights[:, None].conjugate()

    return (np.sum(rotating, axis=0)).imag


def bath_energies(N_couplings: int, ω_c: float) -> np.ndarray:
    """Return the energies (drive detunings) of the bath modes."""

    return (np.arange(1, N_couplings + 1) - 1 / 2) * ω_c / N_couplings


def ohmic_spectral_density(ω: np.ndarray, α: float) -> np.ndarray:
    """The unnormalized spectral density of an Ohmic bath."""

    ω = np.concatenate([[0], ω])
    return np.diff(ω ** (α + 1))


def lamb_shift(amplitudes, Δs):
    return np.sum(amplitudes**2 / Δs)


def drive_frequencies_and_amplitudes(
    params: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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

    a_shift = 0
    if not params.flat_energies and params.correct_lamb_shift:
        a_shift = np.sum(amplitudes**2 / Δs) * params.correct_lamb_shift**2

    ωs = ((np.arange(1, params.N_couplings + 1) - params.δ) * params.Ω) - (Δs - a_shift)
    return ωs, Δs, amplitudes, a_shift


def mode_name(mode: int, N: int | None = None):
    """Return the name of the mode."""
    if mode == 0:
        return r"$\bar{A}$"
    if mode == 1:
        return "A"

    if N:
        bath_mode = mode - 2
        if mode >= N:
            return rf"$\bar{{B}}{bath_mode % N + 1}$"

    return f"B{mode - 1}"


def a_site_indices(params: Params):
    """Return the indices of the A sites."""
    return [0, 1]


def coupled_bath_mode_indices(params: Params):
    """Return the indices of the bath modes that are coupled to the A site."""
    return np.arange(2, 2 + params.N_couplings)


def uncoupled_bath_mode_indices(params: Params):
    """Return the indices of the bath modes that not coupled to the A site."""
    return np.arange(2 + params.N_couplings, 2 + 2 * params.N)


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
    return 2 * params.N + 2


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
