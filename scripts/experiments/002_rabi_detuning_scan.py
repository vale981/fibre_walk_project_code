import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate


@dataclass
class Params:
    N: int = 2
    """Number of bath modes."""

    Ω: float = 1
    """FSR"""

    η: float = 0.1
    """Decay rate."""

    d: float = 0.01
    """Drive amplitude."""

    Δ: float = 0.0
    """Detuning."""

    laser_detuning: float = 0.0
    """Detuning of the laser."""

    laser_off_time: float | None = None
    """Time at which the laser is turned off."""

    drive_off_time: float | None = None
    """Time at which the drive is turned off."""

    def periods(self, n: float):
        return n * 2 * np.pi / self.Ω

    def lifetimes(self, n: float):
        return n / self.η


class RuntimeParams:
    def __init__(self, params: Params):
        self.Ωs = np.arange(0, params.N) * params.Ω - 1j * np.repeat(params.η, params.N)


def drive(t, x, d, Δ, Ω):
    stacked = np.repeat([x], len(x), 0)

    np.fill_diagonal(stacked, 0)

    stacked = np.sum(stacked, axis=1)
    driven_x = d * np.sin((Ω - Δ) * t) * stacked

    return driven_x


def make_righthand_side(Ωs, params: Params):
    def rhs(t, x):
        differential = Ωs * x

        if (params.drive_off_time is None) or (t < params.drive_off_time):
            differential += drive(t, x, params.d, params.Δ, params.Ω)

        if (params.laser_off_time is None) or (t < params.laser_off_time):
            differential += np.exp(-1j * params.laser_detuning * t)

        return -1j * differential

    return rhs


def solve(t: np.ndarray, params: Params):
    runtime = RuntimeParams(params)
    rhs = make_righthand_side(runtime.Ωs, params)

    initial = np.zeros(params.N, np.complex128)

    sol = scipy.integrate.solve_ivp(
        rhs,
        (0, np.max(t)),
        initial,
        vectorized=False,
        max_step=0.01 * 2 * np.pi / np.max(abs(runtime.Ωs.real)),
        method="BDF",
        t_eval=t,
    )
    return sol


def in_rotating_frame(t, amplitudes, params: Params):
    Ωs = RuntimeParams(params).Ωs

    return amplitudes * np.exp(1j * Ωs[:, None].real * t[None, :])


def output_signal(t: np.ndarray, amplitudes: np.ndarray, laser_detuning: float):
    return (np.sum(amplitudes, axis=0) * np.exp(1j * laser_detuning * t)).imag


def reflect(center, value):
    diff = center - value
    return center - diff


# %% interactive
f, (ax1, ax2) = plt.subplots(2, 1)


def transient_analysis():
    params = Params(η=0.001, d=0.1, laser_detuning=0, Δ=0.05)
    params.laser_off_time = params.lifetimes(2)
    params.drive_off_time = params.lifetimes(2)
    t = np.arange(0, params.lifetimes(5), 0.5 / (params.Ω * params.N))
    solution = solve(t, params)

    signal = output_signal(t, solution.y, params.laser_detuning)
    # signal += np.random.normal(0, .1 * np.max(abs(signal)), len(signal))

    window = (params.lifetimes(2) > t) & (t > params.lifetimes(0))
    # window = t > params.lifetimes(2)
    t = t[window]
    signal = signal[window]

    ax1.clear()
    # ax1.plot(
    #     solution.t, np.real(in_rotating_frame(solution.t, solution.y, params)[1, :])
    # )
    ax1.plot(t, signal)
    ax1.set_title(f"Output signal\n {params}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Intensity")

    ax2.clear()
    freq = np.fft.rfftfreq(len(t), t[1] - t[0]) * 2 * np.pi
    fft = np.fft.rfft(signal)
    ax2.set_xlim(0, params.Ω * (params.N))

    energy = 1 / 2 * np.sqrt(4 * params.d**2 + 4 * params.Δ**2)
    for n in range(params.N):
        sidebands = (
            (n) * params.Ω
            - params.laser_detuning
            + np.array([1, -1]) * energy / 2
            - params.Δ / 2
        )

        right_sidebands = (
            (n) * params.Ω
            + params.laser_detuning
            + np.array([1, -1]) * energy / 2
            - params.Δ / 2
        )

        for sideband in sidebands:
            ax2.axvline(
                sideband,
                color=f"C{n}",
                label=f"rabi-sideband {n}",
            )

        for sideband in right_sidebands:
            ax2.axvline(
                sideband,
                color=f"C{n}",
                linestyle="--",
            )
    ax2.axvline(params.Ω - params.Δ, color="black", label="")
    ax2.axvline(2 * params.Ω - params.Δ, color="black", label="")
    ax2.plot(freq, np.abs(fft) ** 2)
    ax2.set_yscale("log")
    ax2.set_title("FFT")
    ax2.set_xlabel("ω")
    ax2.set_ylabel("Power")
    ax2.legend()

    return solution
