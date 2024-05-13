import matplotlib.pyplot as plt
import numpy as np


def fake_signal(t, amps, ωs, γs):
    """Generate a fake signal with given amplitudes, frequencies and damping factors.

    :param t: time vector
    :param amps: amplitudes
    :param ωs: frequencies
    :param γs: damping factors
    """
    return np.imag(
        np.sum(
            amps[None, :]
            * np.exp(-1j * ωs[None, :] * t[:, None] - γs[None, :] * t[:, None]),
            axis=1,
        )
    )


def undamped_fft(t, y, γ):
    fft = np.fft.rfft(y * np.exp(γ * t))
    freq = np.fft.rfftfreq(len(t), t[1] - t[0])

    return freq, fft


def laplace_fft(t, y, γs):
    freq = np.fft.rfftfreq(len(t), t[1] - t[0])
    output = np.empty((len(freq), len(γs)), dtype=complex)
    for i, γ in enumerate(γs):
        row = np.fft.rfft(y * np.exp(γ * t))
        output[:, i] = row / np.abs(row).max()

    return freq, output


# %% Interactive
def test_fake_signal():
    ωs = 2 * np.pi * np.array([0.9, 1, 1.1])
    γs = np.array([0.2, 0.3, 0.2])
    amps = np.ones_like(ωs)  # np.random.uniform(0, 1, len(ωs))

    t = np.linspace(0, 2 * np.pi / max(ωs) * 100, 10000)
    y = fake_signal(t, amps, ωs, γs)
    y += np.random.normal(0, 1, len(t))

    print(f"γs: {γs}")
    print(f"ωs: {ωs / (2 * np.pi)}")
    # γ_scan = np.array([0, γs[0], γs[0] * 2])
    γ_scan = np.linspace(0, np.max(γs) * 1.1, 1000)
    freq, fft = laplace_fft(t, y, γ_scan)

    signal = np.abs(fft) ** 2
    signal = np.flip(signal, axis=1)

    fig, axes = plt.subplot_mosaic("AB;CC")
    (ax1, ax2, ax3) = axes.values()

    ax3.plot(t, y)
    ax2.imshow(
        signal.T,
        aspect="auto",
        extent=(0, max(freq), 0, max(γ_scan)),
        norm="log",
        interpolation=None,
    )
    for i, ω in enumerate(ωs):
        ax2.axvline(ω / (2 * np.pi), color=f"C{i}", linestyle="--")

    for i, γ in enumerate(γs):
        ax2.axhline(γ, color=f"C{i}", linestyle="--")

    ax2.set_xlim(0, 1.1 * max(ωs) / (2 * np.pi))

    for i in range(1):
        ax1.plot(freq, np.abs(fft[:, i]) ** 2, alpha=1)

    for i, ω in enumerate(ωs):
        ax1.axvline(
            ω / (2 * np.pi), color=f"C{i}", linestyle="--", alpha=0.2, zorder=-10
        )
    ax1.set_xlim(0, 1.1 * max(ωs) / (2 * np.pi))

    # plt.legend()
