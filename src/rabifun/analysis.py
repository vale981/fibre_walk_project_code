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
        return max(self.fω_shift - self.mode_window[0] * self.fΩ_guess, 0)

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


def refine_peaks(
    peaks: RingdownPeakData, params: RingdownParams, uncertainty_threshold: float = 0.1
):
    """
    Refine the peak positions and frequencies by fitting Lorentzians.

    :param peaks: The peak data.
    :param params: The ringdown parameters.
    :param uncertainty_threshold: The maximum allowed uncertainty in
        the mode frequencies in units of
        :any:`ringdown_params.fΩ_guess`.
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
    deleted_peaks = []
    for i, peak_freq in enumerate(peak_freqs):
        mask = (freqs > peak_freq - window) & (freqs < peak_freq + window)
        windowed_freqs = freqs[mask]
        windowed_power = power[mask]

        p0 = [1, peak_freq, params.η_guess, 0]
        bounds = (
            [0, windowed_freqs[0], 0, 0],
            [np.inf, windowed_freqs[-1], np.inf, 1],
        )

        try:
            popt, pcov = scipy.optimize.curve_fit(
                lorentzian,
                windowed_freqs,
                windowed_power,
                p0=p0,
                bounds=bounds,
            )
            perr = np.sqrt(np.diag(pcov))

            if perr[1] > uncertainty_threshold * params.fΩ_guess:
                deleted_peaks.append(i)
                continue

            new_freqs.append(popt[1])
            Δfreqs.append(perr[1])

            new_widths.append(popt[2])
            Δwidths.append(perr[2])
        except:
            deleted_peaks.append(i)

    peaks.peaks = np.delete(peaks.peaks, deleted_peaks)
    for key, value in peaks.peak_info.items():
        if isinstance(value, np.ndarray):
            peaks.peak_info[key] = np.delete(value, deleted_peaks)

    peaks.peak_freqs = np.array(new_freqs)
    peaks.Δpeak_freqs = np.array(Δfreqs)
    peaks.peak_widths = np.array(new_widths)
    peaks.Δpeak_widths = np.array(Δwidths)

    return peaks


import matplotlib.pyplot as plt


def extract_Ω_δ(
    peaks: RingdownPeakData, params: RingdownParams, threshold: float = 0.1
):
    """
    Extract the FSR and mode splitting from the peaks.  The threshold
    regulates the maximum allowed deviation from the expected FSR.

    :param peaks: The peak data.
    :param params: The ringdown parameters.
    """

    if not peaks.is_refined:
        raise ValueError("Peaks must be refined.")

    peak_indices = peaks.peaks
    peak_freqs = peaks.peak_freqs
    Δpeak_freqs = (
        np.zeros_like(peak_freqs) if peaks.Δpeak_freqs is None else peaks.Δpeak_freqs
    )

    Ω_guess = params.fΩ_guess
    δ_guess = params.fδ_guess

    # first step: we extract the most common frequency spacing
    all_diff = np.abs(peak_freqs[:, None] - peak_freqs[None, :])
    all_diff = np.triu(all_diff)
    all_ΔΩ = np.abs(Δpeak_freqs[:, None] ** 2 + Δpeak_freqs[None, :] ** 2)

    bath_mask = (np.abs((all_diff - Ω_guess)) / Ω_guess < threshold) & (all_diff > 0)
    candidates = all_diff[bath_mask]
    Δcandidates = all_ΔΩ[bath_mask]

    Ω = np.mean(candidates)
    ΔΩ = max(np.sqrt(np.sum(Δcandidates**2)) / len(candidates), np.std(candidates))

    if np.isnan(Ω):
        raise ValueError("No FSR found")

    # second step: we walk through the peaks and label them as for the
    total_peaks = len(peak_indices)
    peak_pool = list(range(total_peaks))

    ladders = []

    current_peak = 0
    current_ladder = []

    possible_diffs = np.array([Ω, Ω - δ_guess, 2 * δ_guess])

    while len(peak_pool) > 1:
        if current_peak == len(peak_pool):
            if current_ladder:
                ladders.append(current_ladder)
            current_ladder = []
            current_peak = peak_pool[0]

        filtered_freqs = peak_freqs[peak_pool]

        diffs = filtered_freqs - filtered_freqs[current_peak]
        diffs[diffs <= 0] = np.inf

        diffs = (
            np.abs(diffs[:, None] - possible_diffs[None, :]) / possible_diffs[None, :]
        )

        min_coords = np.unravel_index(np.argmin(diffs), diffs.shape)
        min_diff = diffs[min_coords]

        mode_index, step_type = min_coords

        if min_diff > threshold:
            if current_ladder:
                ladders.append(current_ladder)
            current_ladder = []
            del peak_pool[current_peak]
            continue

        current_ladder.append((peak_pool[current_peak], step_type, min_diff))
        del peak_pool[current_peak]
        current_peak = mode_index - 1  # we have deleted one peak

    if current_ladder:
        ladders.append(current_ladder)

    # we want at least one bath mode before the A site
    ladders = list(
        filter(
            lambda ladder: sum([1 if x[1] == 1 else 0 for x in ladder]) > 0,
            ladders,
        )
    )

    invalid = []
    for lad_index, ladder in enumerate(ladders):
        length = len(ladder)
        for i, elem in enumerate(ladder):
            if elem[1] == 1:
                if (i + 2) >= length or not (
                    ladder[i + 1][1] == 2 and ladder[i + 2][1] == 1
                ):
                    invalid.append(lad_index)
                break

    ladders = [ladder for i, ladder in enumerate(ladders) if i not in invalid]
    costs = [sum([x[2] for x in ladder]) / len(ladder) for ladder in ladders]

    if len(costs) == 0:
        raise ValueError("No matching modes found.")

    best = ladders[np.argmin(costs)]

    return Ω, ΔΩ, best
