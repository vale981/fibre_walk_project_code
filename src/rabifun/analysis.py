import numpy as np


def fourier_transform(
    t: np.ndarray, signal: np.ndarray, window: tuple[float, float] | None = None
):
    """
    Compute the Fourier transform of a signal from the time array
    ``t`` and the real signal ``signal``.  Optionally, a time window
    can be specified through ``window = ([begin], [end])`` .

    :returns: The (linear) frequency array and the Fourier transform.
    """

    if window:
        mask = (window[1] > t) & (t > window[0])
        t = t[mask]
        signal = signal[mask]

    freq = np.fft.rfftfreq(len(t), t[1] - t[0])
    fft = np.fft.rfft(signal)

    return freq, fft
