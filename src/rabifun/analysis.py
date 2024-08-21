import numpy as np
import scipy
import os
from ringfit.data import ScanData
import dataclasses
from dataclasses import dataclass

from scipy.optimize import Bounds
from enum import Enum
import lmfit


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


def lorentzian(ω, A, ω0, γ):
    """A Lorentzian function with amplitude ``A``, center frequency
    ``ω0``, and decay rate ``γ`` and offset ``offset``.

    :param ω: Frequency array.
    :param A: Amplitude.
    :param ω0: Center frequency.
    :param γ: Decay rate. The decay of a squared amplitude is :math:`2πγ`.
    """

    return A * (γ / 2) ** 2 * (1 / ((ω - ω0) ** 2 + (γ / 2) ** 2))


def one_over_freq_noise(ω, A, γ):
    """A lorentzian with center frequency 0.
    See :any:`lorentzian` for the parameters.
    """

    return lorentzian(ω, A, 0, γ)


def complex_lorentzian(ω, A, ω0, γ):
    """A Lorentzian function with amplitude ``A``, center frequency
    ``ω0``, and decay rate ``γ`` and offset ``offset``.

    :param ω: Frequency array.
    :param A: Amplitude.
    :param ω0: Center frequency.
    :param γ: Decay rate. The decay of a squared amplitude is :math:`2πγ`.
    """

    return A * (γ / 2) * 1 / (-1j * (ω - ω0) - (γ / (2)))


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

    absolute_low_cutoff: float = 0e6
    """
    The absolute lowest frequency to consider in Hz.  This has
    precedence over the ``mode_window``.
    """

    @property
    def low_cutoff(self) -> float:
        """The low cutoff frequency of the ringdown spectrum fft."""
        return max(
            self.fω_shift - self.mode_window[0] * self.fΩ_guess,
            self.absolute_low_cutoff,
        )

    @property
    def high_cutoff(self) -> float:
        """The high cutoff frequency of the ringdown spectrum fft."""
        return self.fω_shift + self.mode_window[1] * self.fΩ_guess


@dataclass
class RingdownPeakData:
    freq: np.ndarray
    """The fft frequency array."""

    power: np.ndarray
    """The normalized power spectrum of the fft."""

    peaks: np.ndarray
    """The indices of the peaks."""

    peak_freqs: np.ndarray
    """The frequencies of the peaks."""

    peak_info: dict
    """The information from :any:`scipy.signal.find_peaks`."""

    σ_power: np.ndarray | None = None
    """The standard deviation of the power spectrum."""

    peak_widths: np.ndarray | None = None
    """
    The widths of the peaks.
    """

    Δpeak_freqs: np.ndarray | None = None
    """The uncertainty in the peak frequencies."""

    Δpeak_widths: np.ndarray | None = None
    """The uncertainty in the peak widths."""

    lorentz_params: list | None = None
    """The lorentzian fit params to be fed into :any:`lorentzian`."""

    lm_result: lmfit.model.ModelResult | None = None
    """The fit result from :any:`lmfit`."""

    noise_floor: float = 0
    """The noise floor of the spectrum."""

    @property
    def is_refined(self) -> bool:
        """Whether the peaks have been refined with :any:`refine_peaks`."""
        return self.lm_result is not None

    def __post_init__(self):
        norm = np.max(self.power)

        self.power /= norm
        if self.σ_power is not None:
            self.σ_power /= norm


def find_peaks(
    freq: np.ndarray,
    power_spectrum: np.ndarray,
    params: RingdownParams,
    prominence: float = 0.005,
    height: float = 0.1,
    σ_power: np.ndarray | None = None,
) -> RingdownPeakData:
    """Determine the peaks of the power spectrum of the
    ringdown data.

    :param freq: The frequency axis data.
    :params power_spectrum: The FFT power spectrum.
    :param params: The ringdown parameters, see :any:`RingdownParams`.
    :param prominence: The prominence (vertical distance of peak from
        surrounding valleys) of the peaks.
    :param height: The minimum height of the peaks.
    :param σ_power: The standard deviation of the power spectrum.
    """

    freq_step = freq[1] - freq[0]

    distance = params.fδ_guess / 2 / freq_step
    if distance < 1:
        raise ValueError("Insufficient frequency resolution.")

    normalized = power_spectrum - np.median(power_spectrum)
    normalized /= normalized.max()
    peaks, peak_info = scipy.signal.find_peaks(
        normalized,
        distance=distance,
        wlen=distance,
        prominence=prominence,
        height=height,
    )

    peak_freqs = freq[peaks]

    return RingdownPeakData(
        freq=freq,
        peaks=peaks,
        peak_freqs=peak_freqs,
        peak_info=peak_info,
        power=power_spectrum,
        σ_power=σ_power,
    )


def offset(ω, offset):
    return offset


def filter_peaks(
    peaks: RingdownPeakData,
    params: RingdownParams,
    uncertainty_threshold: float = 0.2,
    height_cutoff: float = 0.1,
    to_be_deleted: list = [],
):
    deleted_peaks = []
    if not peaks.is_refined:
        return peaks

    peaks = dataclasses.replace(peaks)

    for i in reversed(range(len(peaks.peak_freqs))):
        A, ω0, γ = peaks.lorentz_params[i]
        Δω0, Δγ = peaks.Δpeak_freqs[i], peaks.Δpeak_widths[i]

        if (
            i in to_be_deleted
            or Δω0 > uncertainty_threshold * params.fΩ_guess
            or A < height_cutoff
            or A > 5
            or Δγ > uncertainty_threshold * params.fΩ_guess
        ):
            peaks.peaks = np.delete(peaks.peaks, i)
            peaks.peak_freqs = np.delete(peaks.peak_freqs, i)
            peaks.Δpeak_freqs = np.delete(peaks.Δpeak_freqs, i)
            peaks.peak_widths = np.delete(peaks.peak_widths, i)
            peaks.Δpeak_widths = np.delete(peaks.Δpeak_widths, i)

            del peaks.lorentz_params[i]
            deleted_peaks.append(i)
            continue

    for key, value in peaks.peak_info.items():
        if isinstance(value, np.ndarray):
            peaks.peak_info[key] = np.delete(value, deleted_peaks)

    return peaks


def refine_peaks(
    peaks: RingdownPeakData,
    params: RingdownParams,
    uncertainty_threshold: float = 0.2,
    height_cutoff: float = 0.1,
) -> RingdownPeakData:
    """
    Refine the peak positions and frequencies by fitting a sum of
    Lorentzians.  The peaks are filtered according to the
    ``height_cutoff``, ``uncertainty_threshold`` and other criteria
    and the fit repeated until nothing changes.  The results are
    stored in a copy of ``peaks``, among them the last successful
    :any:`lmfit` fit result.

    :param peaks: The peak data.
    :param params: The ringdown parameters.
    :param uncertainty_threshold: The maximum allowed uncertainty in
        the mode frequencies in units of
        :any:`ringdown_params.fΩ_guess`.
    """

    if len(peaks.peaks) == 0:
        return peaks

    peaks = dataclasses.replace(peaks)
    freqs = peaks.freq
    peak_freqs = peaks.peak_freqs
    power = peaks.power

    total_model = None
    model_params = []

    scaled_power = power

    σ = np.zeros_like(power) if peaks.σ_power is None else peaks.σ_power

    for i, (A, ω0) in enumerate(zip(peaks.peak_info["peak_heights"], peak_freqs)):
        model = lmfit.Model(lorentzian, prefix=f"peak_{i}_")

        initial_params = model.make_params(
            A=dict(value=A, min=0, max=np.inf),
            ω0=dict(value=ω0, min=0, max=np.inf),
            γ=dict(value=params.η_guess, min=0, max=np.inf),
        )

        if total_model is None:
            total_model = model
        else:
            total_model += model

        model_params.append(initial_params)

    model = lmfit.Model(one_over_freq_noise, prefix=f"zero_peak")

    initial_params = model.make_params(
        A=dict(value=1, min=0, max=np.inf),
        γ=dict(value=params.η_guess, min=0, max=np.inf),
    )

    total_model += model
    model_params.append(initial_params)

    aggregate_params = total_model.make_params()
    for lm_params in model_params:
        aggregate_params.update(lm_params)

    offset_model = lmfit.Model(offset)
    aggregate_params.update(offset_model.make_params(offset=0, min=0, max=1))
    total_model += offset_model

    lm_result = total_model.fit(
        scaled_power,
        params=aggregate_params,
        ω=freqs,
        weights=1 / σ if np.all(σ > 0) else None,
    )

    peaks.peak_freqs = np.zeros_like(peaks.peak_freqs)
    peaks.Δpeak_freqs = np.zeros_like(peaks.peak_freqs)
    peaks.peak_widths = np.zeros_like(peaks.peak_freqs)
    peaks.Δpeak_widths = np.zeros_like(peaks.peak_freqs)

    peaks.lorentz_params = [None] * len(peaks.peak_freqs)

    for i in range(len(peaks.peak_freqs)):
        peak_prefix = f"peak_{i}_"

        A, ω0, γ = (
            lm_result.best_values[peak_prefix + "A"],
            lm_result.best_values[peak_prefix + "ω0"],
            lm_result.best_values[peak_prefix + "γ"],
        )

        ΔA, Δω0, Δγ = (
            lm_result.params[peak_prefix + "A"].stderr,
            lm_result.params[peak_prefix + "ω0"].stderr,
            lm_result.params[peak_prefix + "γ"].stderr,
        )

        peaks.peak_freqs[i] = ω0
        peaks.Δpeak_freqs[i] = Δω0

        peaks.peak_widths[i] = γ
        peaks.Δpeak_widths[i] = Δγ

        peaks.lorentz_params[i] = A, ω0, γ

    peaks.lm_result = lm_result
    peaks.noise_floor = lm_result.best_values["offset"]

    before_filter = len(peaks.peaks)
    peaks = filter_peaks(peaks, params, uncertainty_threshold, height_cutoff)

    if len(peaks.peaks) < before_filter:
        return refine_peaks(peaks, params, uncertainty_threshold, height_cutoff)

    return peaks


class StepType(Enum):
    BATH = 0
    BATH_TO_A = 1
    A_TO_A = 2


def extract_Ω_δ(
    peaks: RingdownPeakData,
    params: RingdownParams,
    Ω_threshold: float = 0.1,
    ladder_threshold: float = 0.1,
    bifurcations: int = 3,
    start_peaks: int = 2,
    min_length: int = 4,
):
    """
    Extract the FSR and mode splitting from the peaks.  The threshold
    regulates the maximum allowed deviation from the expected FSR.

    The basis of the algorithm is, that one knows a rough estimate of
    the FSR ``params.Ω_guess`` of the big loop (maybe up to 1%
    deviation).  The algorithm then identifies all peaks are
    ``Ω_threshold * params.Ω_guess`` off from the guess.  Subsequently
    the estimate of the FSR is refined.  The algorithm then
    recursively tries to match peaks into the pattern or "ladder"
    ``[bath, bath, ..., bath, hybridized, hybridized, bath, ...]``.
    The ladder that fits best is returned.  One can start to construct
    these ladders beginning at the first up to the ``start_peaks``st
    peak.  At each step, the algorithm accepts modes that fit within
    the relative ``ladder_threshold`` up to a maximum of modes given
    by the integer ``biffurcation``.  If a ladder is comprised of less
    than ``min_length`` modes, it is discarded.

    :param peaks: The peak data.
    :param params: The ringdown parameters.
    :param Ω_threshold: The maximum allowed relative deviation from
        the expected FSR for the rough search.
    :param ladder_threshold: The maximum allowed relative deviation
        from the expected step sizes for the ladder search.
    :param bifurcations: The number of bifurcations to consider in the
        ladder search, i.e. how many possible new steps are accepted
        at each step.
    :param start_peaks: The number of peaks to start the ladder search
        (from the left).
    :param min_length: The minimum length of a ladder to be considered
        valid.
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
    all_ΔΩ = np.sqrt(Δpeak_freqs[:, None] ** 2 + Δpeak_freqs[None, :] ** 2)

    bath_mask = (np.abs((all_diff - Ω_guess)) / Ω_guess < Ω_threshold) & (all_diff > 0)
    candidates = all_diff[bath_mask]
    Δcandidates = all_ΔΩ[bath_mask]

    Ω = np.mean(candidates)
    ΔΩ = max(np.sqrt(np.sum(Δcandidates**2)) / len(candidates), np.std(candidates))

    if np.isnan(Ω):
        raise ValueError("No bath modes found!")

    # second step: we walk through the peaks and label them as for the
    total_peaks = len(peak_indices)
    peak_pool = list(range(total_peaks))

    ladders = []

    current_peak = 0
    current_ladder = []

    possible_diffs = np.array([Ω, Ω - δ_guess, 2 * δ_guess])

    # entry in a ladder: (peak_index, step_type)

    def walk_ladder(
        current_peak,
        last_type=StepType.BATH,
        second_last_type=StepType.BATH,
        bifurcations=bifurcations,
    ):
        if current_peak == total_peaks - 1:
            return [], []

        match last_type:
            case StepType.BATH:
                allowed_steps = [StepType.BATH, StepType.BATH_TO_A]
            case StepType.BATH_TO_A:
                match second_last_type:
                    case StepType.BATH:
                        allowed_steps = [StepType.A_TO_A]

                    case StepType.A_TO_A:
                        allowed_steps = [StepType.BATH]
            case StepType.A_TO_A:
                allowed_steps = [StepType.BATH_TO_A]

        allowed_step_indices = [x.value for x in allowed_steps]
        allowed_possible_diffs = possible_diffs[allowed_step_indices]

        diffs = peak_freqs - peak_freqs[current_peak]
        diffs[diffs <= 0] = np.inf

        diffs = (
            np.abs(diffs[:, None] - allowed_possible_diffs[None, :])
            / allowed_possible_diffs[None, :]
        )

        ladders = []
        costs = []
        min_candidates = np.argpartition(diffs.flatten(), bifurcations)[:bifurcations]

        for min_coords in min_candidates:
            min_coords = np.unravel_index(min_coords, diffs.shape)
            min_diff = diffs[min_coords]
            peak_index, step_type = min_coords

            step_type = StepType(allowed_step_indices[step_type])

            this_rung = [(current_peak, step_type)]
            if min_diff < ladder_threshold + (ΔΩ / Ω):
                new_ladders, new_costs = walk_ladder(
                    peak_index,
                    step_type,
                    last_type,
                    bifurcations,
                )

                if len(new_ladders) == 0:
                    ladders.append(this_rung)
                    costs.append(min_diff)
                else:
                    for new_ladder, new_cost in zip(new_ladders, new_costs):
                        if new_ladder:
                            ladders.append(this_rung + new_ladder)
                            costs.append(new_cost + min_diff)

        return ladders, costs

    ladders = []
    costs = []

    for start_index in range(min(total_peaks, start_peaks)):
        new_ladders, new_costs = walk_ladder(
            start_index, StepType.BATH, StepType.BATH, bifurcations
        )
        ladders += new_ladders
        costs += new_costs

    invalid = []
    for lad_index, ladder in enumerate(ladders):
        if len(ladder) < min_length:
            invalid.append(lad_index)
            continue

        is_invalid = True
        for elem in ladder:
            if elem[1] == StepType.A_TO_A:
                is_invalid = False
                break

        if is_invalid:
            invalid.append(lad_index)

    ladders = [ladder for i, ladder in enumerate(ladders) if i not in invalid]
    costs = [cost for i, cost in enumerate(costs) if i not in invalid]

    costs = [cost / len(ladder) for cost, ladder in zip(costs, ladders)]

    if len(costs) == 0:
        print("No valid ladders/spectra found.")

        return Ω, ΔΩ, None, None, None

    best = np.argmin(costs)
    best_ladder = ladders[best]

    Ωs = []
    δs = []
    Δδs = []
    Ω_m_δs = []

    for (i, (begin_index, begin_type)), (end_index, _) in zip(
        enumerate(best_ladder[:-1]), best_ladder[1:]
    ):
        match begin_type:
            case StepType.BATH:
                Ωs.append(peak_freqs[end_index] - peak_freqs[begin_index])
            case StepType.BATH_TO_A:
                Ω_m_δs.append(peak_freqs[end_index] - peak_freqs[begin_index])
            case StepType.A_TO_A:
                δs.append((peak_freqs[end_index] - peak_freqs[begin_index]) / 2)
                Δδs.append(
                    np.sqrt(Δpeak_freqs[end_index] ** 2 + Δpeak_freqs[begin_index] ** 2)
                )

    Ω = np.mean(Ωs)
    ΔΩ = np.std(Ωs)
    δ = np.mean(δs)
    Δδ = np.mean(Δδs)

    return Ω, ΔΩ, δ, Δδ, best_ladder
