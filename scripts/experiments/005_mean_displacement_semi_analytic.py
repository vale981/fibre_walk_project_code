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

    return coeff / np.sum((coeff))


def characteristic_poly(g: np.ndarray, ε: np.ndarray, η_A: float = 0):
    g2 = np.abs(g) ** 2

    def poly(ω):
        s = ω[0] + ω[1] * 1j
        res = s + 1j * η_A - np.sum(g2 / (s - ε))
        return (res * res.conjugate()).real

    return poly


def hamiltonian(g: np.ndarray, ε: np.ndarray, η_A: float = 0):
    H = np.diag([-1j * η_A, *ε])
    H[0, 1:] = g
    H[1:, 0] = np.conj(g)
    return H


def a_site_population(t, ω, coeff):
    return (
        np.abs(np.sum(coeff[None, :] * np.exp(-1j * ω[None, :] * t[:, None]), axis=1))
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
        N=2 * N + 2,
        N_couplings=N,
        measurement_detuning=0,
        α=0,
        rwa=False,
        flat_energies=False,
        correct_lamb_shift=True,
        laser_off_time=0,
    )


def test():
    params = make_params(N=10, gbar=1 / 3)
    params.flat_energies = False
    params.α = 0.0
    params.correct_lamb_shift = True

    runtime = RuntimeParams(params)
    t = time_axis(params, recurrences=1.5)
    g = runtime.g / 2
    ε = runtime.bath_ε
    H = hamiltonian(g, ε.real, 0 * params.η / 2)

    ω = np.linalg.eigvals(H)
    M = np.linalg.eig(H).eigenvectors
    Minv = np.linalg.inv(M)
    v0 = Minv[:, 0]

    coeff = M[0, :] * v0.T
    print(coeff)

    # coeff /= np.abs(np.sum(coeff))
    # coeff = coefficients(ω, g, ε)

    plt.cla()
    plt.plot(t, a_site_population(t, ω, coeff))
    plt.ylim(0, 1)
    return ω
