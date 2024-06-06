"""
Just a quick hack prior to the re-calculation of the full phase
diagram.
"""

import numpy as np
from rabifun.system import *
from rabifun.plots import *
from rabifun.utilities import *
import scipy
import matplotlib.pyplot as plt


def self_energy(ω, g, ε):
    """
    Calculate the self-energy of the system.

    :param g: The coupling strengths.
    :param ε: The bath energies.
    """
    return np.sum(np.abs(g[None, :] / (ω[:, None] - ε[None, :])) ** 2, axis=1)


def coefficients(ω, g, ε):
    coeff = 1 / (1 + self_energy(ω, g, ε))

    return coeff


def characteristic_poly(g: np.ndarray, ε: np.ndarray, η_A: float = 0):
    g2 = np.abs(g) ** 2

    def poly(ω):
        s = ω[0] + ω[1] * 1j
        res = s + 1j * η_A - np.sum(g2 / (s - ε))
        return (res * res.conjugate()).real

    return poly


def hamiltonian(g: np.ndarray, ε: np.ndarray, ε_A: float = 0, η_A: float = 0):
    H = np.diag([ε_A - 1j * η_A, *ε])
    H[0, 1:] = g
    H[1:, 0] = np.conj(g)
    return H


def a_site_population(t, ω, coeff, lower_cutoff=0):
    return (
        np.abs(
            np.sum(
                coeff[None, lower_cutoff:]
                * np.exp(-1j * ω[None, lower_cutoff:] * t[:, None]),
                axis=1,
            )
        )
        ** 2
    )


def make_params(ω_c=0.1 / 2, N=10, gbar=1 / 3):
    """
    Make a set of parameters for the system with the current
    best-known settings.
    """
    return Params(
        η=0.5,
        Ω=13,
        δ=1 / 4,
        ω_c=ω_c,
        g_0=ω_c * gbar,
        laser_detuning=0,
        N=N,
        N_couplings=N,
        measurement_detuning=0,
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=True,
        laser_off_time=0,
    )


def test():
    params = make_params(N=30, gbar=1 / 4)
    params.flat_energies = False
    params.α = 2
    params.η = 0

    params.correct_lamb_shift = 1
    runtime = RuntimeParams(params)
    t = time_axis(params, recurrences=1.1)
    g = runtime.g
    ε = runtime.bath_ε

    H = hamiltonian(g, ε.real, ε_A=runtime.a_shift, η_A=0 * params.η / 2)

    ω = np.linalg.eigvalsh(H)

    idx = np.argsort(ω.real)
    ω = ω[idx]

    # M = np.linalg.eig(H).eigenvectors[:, idx]

    # Minv = np.linalg.inv(M)

    # coeff = M[0, :] * Minv[:, 0]
    # coeff = coeff[idx]
    coeff = coefficients(ω, g, ε.real)

    f = make_figure()

    U_A = np.abs(1 / (1 + np.sum((g / (ω[0] - ε.real)) ** 2))) ** 2
    U_A_coeff = np.max(np.abs(coeff**2))
    ax_t, ax_e = f.subplots(1, 2)
    ax_t.plot(t, a_site_population(t, ω, coeff, 0))
    # ax_t.plot(t, U_A_coeff + .1 * np.sin(ω[0].real * t))
    # ax_t.set_ylim(0, 1.01)
    ax_t.set_xlabel("Time [1/Ω]")
    ax_t.set_ylabel(r"$ρ_A$")

    ax_t.plot(t, np.exp(-np.sum(g**2) * t))
    # ax_t.set_xscale("log")

    # print((np.abs(1 / (1 + np.sum((g / ε) ** 2)))))
    ax_t.axhline(U_A, color="green", linestyle="-.")
    print(U_A)
    ax_t.axhline(U_A_coeff, color="red", linestyle="--")

    # ax_e.axvline(min(ω.real), color="black", linestyle="--")
    # ax_e.axvline(max(ω.real), color="black", linestyle="--")
    # ax_e.axvline(max(ε.real), color="green", linestyle="--")
    for ω_, c in zip(ω, coeff):
        ax_e.vlines(ω_.real, 0, c, color="blue", alpha=0.5)
    ax_e.set_xlabel("Energy")
