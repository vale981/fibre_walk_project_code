import numpy as np
import scipy
import os
from ringfit.data import ScanData
import dataclasses
from dataclasses import dataclass

from scipy.optimize import Bounds


def fourier_transform(
    t: np.ndarray,
    signal: np.ndarray,
    window: tuple[float, float] | None = None,
    low_cutoff: float = 0,
    high_cutoff: float = np.inf,
    ret_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fourier transform of a signal from the time array
    ``t`` and the real signal ``signal``.  Optionally, a time window
    can be specified through ``window = ([begin], [end])`` .  The
    ``low_cuttof (high_cutoff)`` is the lower (upper) bound of frequencies returned.

    :returns: The (linear) frequency array and the Fourier transform.
    """

    if window:
        mask = (window[1] > t) & (t > window[0])
        t = t[mask]
        signal = signal[mask]  # * scipy.signal.windows.hamming(len(t))

        #

    freq = scipy.fft.rfftfreq(len(t), t[2] - t[1])
    fft = scipy.fft.rfft(signal, norm="forward", workers=os.cpu_count())

    mask = (freq > low_cutoff) & (freq < high_cutoff)
    return (freq[mask], fft[mask], t) if ret_time else (freq[mask], fft[mask])


def lorentzian(ω, A, ω0, γ, offset):
    """A Lorentzian function with amplitude ``A``, center frequency
    ``ω0``, and decay rate ``γ`` and offset ``offset``.

    :param ω: Frequency array.
    :param A: Amplitude.
    :param ω0: Center frequency.
    :param γ: Decay rate. The decay of an amplitude is :math:`\frac{γ}{2}`.
    :param offset: Vertical offset.
    """

    return (
        A * (γ / (4 * np.pi)) ** 2 * (1 / ((ω - ω0) ** 2 + (γ / (4 * np.pi)) ** 2))
        + offset
    )


def complex_lorentzian(ω, A, ω0, γ):
    """A Lorentzian function with amplitude ``A``, center frequency
    ``ω0``, and decay rate ``γ`` and offset ``offset``.

    :param ω: Frequency array.
    :param A: Amplitude.
    :param ω0: Center frequency.
    :param γ: Decay rate. The decay of an amplitude is :math:`\frac{γ}{2}`.
    """

    return A * (γ / 2) * 1 / (-1j * (ω - ω0) - (γ / (4 * np.pi)))


###############################################################################
#                             Ringdown Calibration                          #
###############################################################################


@dataclass
class RingdownParams:
    fω_shift: float = 200e6
    """The laser frequency shift in Hz (linear)."""

    fΩ_guess: float = 13e6
    """The best guess of the FSR in Hz (linear)."""

    fδ_guess: float = 3e6
    """The best guess of the mode splitting in Hz (linear)."""

    η_guess: float = 0.5e6
    """The best guess of the worst decay rate in Hz (no 2πs)."""

    mode_window: tuple[int, int] = (10, 10)
    """How many FSRs of frequency to consider around :any`fω_shift`."""

    @property
    def low_cutoff(self) -> float:
        """The low cutoff frequency of the ringdown spectrum fft."""
        return self.fω_shift - self.mode_window[0] * self.fΩ_guess

    @property
    def high_cutoff(self) -> float:
        """The high cutoff frequency of the ringdown spectrum fft."""
        return self.fω_shift + self.mode_window[1] * self.fΩ_guess


@dataclass
class RingdownPeakData:
    freq: np.ndarray
    """The fft frequency array."""

    fft: np.ndarray
    """The fft amplitudes."""

    normalized_power: np.ndarray
    """The normalized power spectrum of the fft."""

    peaks: np.ndarray
    """The indices of the peaks."""

    peak_freqs: np.ndarray
    """The frequencies of the peaks."""

    peak_info: dict
    """The information from :any:`scipy.signal.find_peaks`."""

    peak_widths: np.ndarray | None = None
    """
    The widths of the peaks.
    """

    Δpeak_freqs: np.ndarray | None = None
    """The uncertainty in the peak frequencies."""

    Δpeak_widths: np.ndarray | None = None
    """The uncertainty in the peak widths."""

    @property
    def is_refined(self) -> bool:
        """Whether the peaks have been refined with :any:`refine_peaks`."""
        return self.peak_widths is not None


def find_peaks(
    data: ScanData,
    params: RingdownParams,
    window: tuple[float, float],
    prominence: float = 0.005,
):
    """Determine the peaks of the normalized power spectrum of the
    ringdown data.

    :param data: The oscilloscope data.
    :param params: The ringdown parameters, see :any:`RingdownParams`.
    :param window: The time window to consider.  (to be automated)
    :param prominence: The prominence (vertical distance of peak from
        surrounding valleys) of the peaks.
    """

    freq, fft, t = fourier_transform(
        data.time,
        data.output,
        window=window,
        low_cutoff=params.low_cutoff,
        high_cutoff=params.high_cutoff,
        ret_time=True,
    )
    freq_step = freq[1] - freq[0]

    power = np.abs(fft) ** 2
    power /= power.max()

    distance = params.fδ_guess / 2 / freq_step
    peaks, peak_info = scipy.signal.find_peaks(
        power, distance=distance, wlen=distance // 4, prominence=prominence
    )

    peak_freqs = freq[peaks]

    return RingdownPeakData(
        freq=freq,
        fft=fft,
        peaks=peaks,
        peak_freqs=peak_freqs,
        peak_info=peak_info,
        normalized_power=power,
    )


def refine_peaks(peaks: RingdownPeakData, params: RingdownParams):
    """
    Refine the peak positions and frequencies by fitting Lorentzians.

    :param peaks: The peak data.
    :param params: The ringdown parameters.
    """

    peaks = dataclasses.replace(peaks)
    freqs = peaks.freq
    peak_freqs = peaks.peak_freqs
    power = peaks.normalized_power

    new_freqs = []
    new_widths = []
    Δfreqs = []
    Δwidths = []

    window = params.η_guess * 3
    for peak_freq in peak_freqs:
        mask = (freqs > peak_freq - window) & (freqs < peak_freq + window)
        windowed_freqs = freqs[mask]
        windowed_power = power[mask]

        p0 = [1, peak_freq, params.η_guess, 0]
        bounds = (
            [0, windowed_freqs[0], 0, 0],
            [np.inf, windowed_freqs[-1], np.inf, 1],
        )

        popt, pcov = scipy.optimize.curve_fit(
            lorentzian,
            windowed_freqs,
            windowed_power,
            p0=p0,
            bounds=bounds,
        )
        perr = np.sqrt(np.diag(pcov))

        new_freqs.append(popt[1])
        Δfreqs.append(perr[1])

        new_widths.append(popt[2])
        Δwidths.append(perr[2])

    peaks.peak_freqs = np.array(new_freqs)
    peaks.Δpeak_freqs = np.array(Δfreqs)
    peaks.peak_widths = np.array(new_widths)
    peaks.Δpeak_widths = np.array(Δwidths)

    return peaks


def extract_Ω_δ(
    peaks: RingdownPeakData, params: RingdownParams, threshold: float = 0.1
):
    """
    Extract the FSR and mode splitting from the peaks.

    :param peaks: The peak data.
    :param params: The ringdown parameters.
    """

    if not peaks.is_refined:
        raise ValueError("Peaks must be refined.")

    Ω_guess = params.fΩ_guess
    δ_guess = params.fδ_guess

    all_diff = np.abs(peaks.peak_freqs[:, None] - peaks.peak_freqs[None, :])
    all_diff = np.triu(all_diff)

    bath_mask = (all_diff - Ω_guess) / Ω_guess < threshold

    return np.mean((all_diff[bath_mask]))
