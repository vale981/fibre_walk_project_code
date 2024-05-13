import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class Params:
    N: int = 10
    Ω: float = 1
    η: float = 0.001
    d: float = 0.1


def make_ks(N: int):
    return np.arange(0, N - 1) * 2 * np.pi / N


def dirac_comb(x: float | np.ndarray, N: int = 1):
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=complex)

    mask = np.isclose(x, 0)
    result[mask] = 1

    masked_x = x[~mask]
    result[~mask] = -2j * np.sin(N / 2 * masked_x) / (N * np.expm1(-1j * masked_x))

    return result


def transmission(t: float | np.ndarray, ω: float | np.ndarray, params: Params):
    ks = make_ks(params.N)

    result = np.zeros_like(t, dtype=complex)

    for k in ks:
        result += dirac_comb(params.Ω * t - k, params.N) / (
            2 * params.d * (np.cos(k) - ω) + 1j * params.η
        )

    return result


def k_coupling(Δk: np.ndarray, params: Params):
    ns = np.arange(-params.N // 2, params.N // 2 - 1)
    return np.sum(np.exp(-1j * Δk[:, None] * ns[None, :]), axis=1) / params.N


# %% interactive stuff


def test_comb():
    xs = np.linspace(0, 10, 10000)
    plt.cla()
    plt.plot(xs, np.abs(dirac_comb(xs, 1000)))


def test_k_coupling():
    Δks = np.linspace(-np.pi, np.pi, 1000)
    plt.cla()
    plt.plot(Δks, np.abs(k_coupling(Δks, Params(N=1000))))
    plt.xlabel("Δk")
    plt.ylabel("Abs[k_coupling]")
    plt.title("Coupling between k modes")


def test_transmission():
    """This prints the non-stationary transmission for a fixed laser frequency."""
    params = Params(N=10000)
    ts = np.linspace(0, 2 * np.pi, 1000)

    plt.cla()
    plt.plot(ts, np.imag(transmission(ts, -0.9, params)), label="ω = -0.9")
    plt.plot(ts, np.imag(transmission(ts, -0, params)), label="ω = 0")
    plt.xlabel("t")
    plt.ylabel("Im[transmission]")
    plt.legend()
    plt.title(f"Non-stationary transmission\n N = {params.N}")


def fake_transmission_steps():
    """
    This aims to reproduce the density of states like picute seen in
    the oscilloscope.

    The way this is done is  a bit stupid but this is only a sketch.
    """
    params = Params(N=1000)

    num_steps = 100
    ωs = np.linspace(-0.9, 0.9, num_steps)

    δt = 2 * np.pi / params.Ω

    T = num_steps * δt
    points_per_step = 1000

    result = np.empty(num_steps * points_per_step)

    for step, ω in enumerate(ωs):
        result[step * points_per_step : (step + 1) * points_per_step] = np.repeat(
            np.imag(
                transmission(np.linspace(0, δt, points_per_step), ω, params)
            ).mean(),
            points_per_step,
        )

    plt.cla()
    ts = np.linspace(0, T, num_steps * points_per_step)
    result = np.abs(result)
    plt.plot(ts, result, label="mean transmission")
    plt.plot(
        ts,
        np.min(result) - 1 + 1 / np.sin(np.linspace(0.1, np.pi - 0.1, len(ts))),
        label="Density of states",
    )
    plt.xlabel("t")
    plt.ylabel("Abs[Transmission]")
    plt.legend()
    plt.title(f"Fake transmission steps\n N = {params.N}")
