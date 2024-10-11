import collections
from typing import Tuple

import numpy as np

from . import spectrum
from .utils import view_as_windows
from .segmentation import (
    segmentation_array_to_table,
    apply_windows_to_segmentation,
    extract_state_sections,
)

SOUND_STATES = ("S1", "S2", "murmur")
NOISE_STATES = ("diastole", "systole")
STATE_ORDER = ("S1", "systole", "S2", "diastole")


def signal_to_noise_ratio(
    series,
    fs,
    segmentation,
    frequency_resolution: int = 25,
    f_min: float = 25,
    f_max: float = 200,
    sound_states: list = SOUND_STATES,
    noise_states: list = NOISE_STATES,
    method="2d_spectrogram_average",
) -> float:
    snr_vs_freq, freq = signal_to_noise_ratio_vs_frequency(
        series,
        fs,
        segmentation,
        frequency_resolution,
        sound_states,
        noise_states,
        method,
    )

    # Finally calculate SNR in a specific frequency band
    low_freq_snr = calculate_snr_in_frequency_band(freq, snr_vs_freq, f_min, f_max)
    return low_freq_snr


def signal_to_noise_ratio_vs_frequency(
    series,
    fs,
    segmentation,
    frequency_resolution: int = 25,
    sound_states: list = SOUND_STATES,
    noise_states: list = NOISE_STATES,
    method="2d_spectrogram_average",
) -> Tuple[np.array, np.array]:
    # This is a very simple function that just divides the result of the below function
    # (we need it so that we can expose signal_and_noise_vs_frequency for external use)

    max_sound_psd, min_noise_psd, f_psd = signal_and_noise_vs_frequency(
        series,
        fs,
        segmentation,
        frequency_resolution,
        sound_states,
        noise_states,
        method,
    )

    return max_sound_psd / min_noise_psd, f_psd


def signal_and_noise_vs_frequency(
    series,
    fs,
    segmentation,
    frequency_resolution: int = 25,
    sound_states: list = SOUND_STATES,
    noise_states: list = NOISE_STATES,
    method: str = "2d_spectrogram_average",
):
    # Calculate averaged power spectral densities (PSDs) for each sound (S1, S2, systole, ...)
    f_psd, averaged_psds = calculate_averaged_state_psds(
        series,
        fs,
        segmentation,
        frequency_resolution,
        method=method,
    )

    # Get an overall 'signal' PSD and 'noise' PSD
    max_sound_psd, min_noise_psd = extract_noise_and_signal_psds(
        averaged_psds, sound_states, noise_states
    )
    return max_sound_psd, min_noise_psd, f_psd


def extract_noise_and_signal_psds(
    median_psds, sound_states: list = SOUND_STATES, noise_states: list = NOISE_STATES
):
    # Compute an overall signal-to-noise ratio by comparing "sound" states to "silence" states
    # Extract sound states from PSD dictionary
    sound_psds = {k: v for k, v in median_psds.items() if k in sound_states}
    if len(sound_psds) == 0:
        raise RuntimeError("No sound states in segmentation. Cannot compute SNR")

    # Similarly, extract 'noise' silent states (systole or diastole) from PSD dictionary
    noise_psds = {k: v for k, v in median_psds.items() if k in noise_states}
    if len(noise_psds) == 0:
        raise RuntimeError("No noise states in segmentation. Cannot compute SNR")

    max_sound_psd = np.max(list(sound_psds.values()), axis=0)
    min_noise_psd = np.min(list(noise_psds.values()), axis=0)
    return max_sound_psd, min_noise_psd


def calculate_snr_in_frequency_band(f, snr_vs_freq, f_min=25, f_max=200):
    # Compute full SNR by taking ratio of PSD (not in log space)
    f_indices = (f >= f_min) & (f < f_max) # slight buffer on f_max as technically 200.7
    snr = np.mean(snr_vs_freq[f_indices])
    return 10 * np.log10(snr)


def find_max_snr(f, snr_vs_freq, width, f_min, f_max):
    # Calculate SNR in each bin across frequency band (f_min, f_max)
    f_indices = (f >= f_min) & (f <= f_max)
    snr_vs_freq = snr_vs_freq[f_indices]

    # Compute moving average of SNR with specified width
    freq_resolution = f[1] - f[0]
    num_window_vals = 1 + int(np.ceil(width / freq_resolution))
    windowed_snr = view_as_windows(snr_vs_freq, num_window_vals).mean(axis=1)

    # Pick average window with highest SNR and return it and the frequency range
    max_idx = np.argmax(windowed_snr)
    max_snr = windowed_snr[max_idx]
    f_valid = f[f_indices]
    f_max_snr = f_valid[max_idx : max_idx + num_window_vals]
    return 10 * np.log10(max_snr), f_max_snr[0].item(), f_max_snr[-1].item()


def calculate_spectrogram_for_psd(series, fs, frequency_resolution):
    # Remove any DC offset from the time series
    series = series - np.mean(series)

    # Compute spectrogram window parameters to achieve desired frequency resolution
    required_window_length = 1 / frequency_resolution
    required_window_step = required_window_length / 2  # 50% overlap for Hann window

    # Compute spectrogram using specified parameters
    new_spec, _, spec_freqs = spectrum.spectrogram(
        series,
        fs,
        window_length=required_window_length,
        window_step=required_window_step,
        window="hann",
    )
    return new_spec, spec_freqs, required_window_length, required_window_step


def calculate_averaged_state_psds(
    series: np.array, fs: float, segmentation: np.array, frequency_resolution: int, method: str
):
    """Calculate PSD for each individual state section in the segmentation.

    Taking S1 as an example, the method extract the relevant spectrogram frames relating to S1,
    and then produce a PSD by averaging the relevant frames for each individual sound.

    Args:
        series (np.array): time series
        fs (float): sampling frequency of time series (Hz)
        segmentation (np.array): segmentation of time series
        frequency_resolution (int): desired frequency resolution of output PSD
        seg_window_length (float, optional): window length (seconds) used to for segmentation. Defaults to 0.025.
        seg_window_step (float, optional): window step (seconds) used for segmentation. Defaults to 0.010.

    Returns:
        (array, dict): two element output:
            (1) array of frequency indexes for the PSDs
            (2) dict of state PSDs. Each dict's value is a list of PSDs for each state.
    """
    if len(segmentation) != len(series):
        raise ValueError(
            "Segmentation must be same length as series array. Use scale_segmentation_to_series()"
        )

    if method == "median_average":
        return _state_psds_from_median_averaged_sections(
            series, fs, segmentation, frequency_resolution
        )
    elif method == "2d_spectrogram_average":
        return _state_psds_from_2d_spectrogram_average(
            series, fs, segmentation, frequency_resolution
        )
    else:
        raise ValueError(f"Unknown method {method}")


def _state_psds_from_median_averaged_sections(
    series: np.array, fs: float, segmentation: np.array, frequency_resolution: int
):
    # Calculate spectrogram for required frequency resolution
    (
        new_spec,
        spec_freqs,
        required_window_length,
        required_window_step,
    ) = calculate_spectrogram_for_psd(series, fs, frequency_resolution)

    # Now scale segmentation down to new spectrogram windows
    scaled_seg = apply_windows_to_segmentation(
        segmentation, fs, required_window_length, required_window_step, window_type="hann"
    )
    assert new_spec.shape[-1] == len(scaled_seg)

    # Remove start or end states from spectrogram as they are 'incomplete'
    # (first, remove unspecified last state if it exists, as this means second-to-last incomplete)
    state_df = segmentation_array_to_table(scaled_seg)
    if state_df.iloc[-1]["state"] == "unspecified":
        state_df = state_df.iloc[:-1]
    state_df = state_df.iloc[1:-1]
    state_df = state_df[state_df["state"] != "noise"]  # actual noise we want is systole/diastole

    # Collate individual spectrograms for each state section
    state_psds = collections.defaultdict(list)
    for _, row in state_df.iterrows():
        # Skip states that cover less than 50 ms as they are likely erroneous
        state_duration = (row["end"] + 1 - row["start"]) * required_window_step
        if state_duration < 0.05:
            continue

        # Extract spectrogram section and take its average to get a PSD (over time)
        spec_section = new_spec[:, row["start"] : row["end"] + 1]
        averaged_section = spec_section.mean(axis=-1)
        state_psds[row["state"]].append(averaged_section)

    # Compute median of all PSDS for each state
    median_psds = {state: np.median(np.stack(specs), axis=0) for state, specs in state_psds.items()}

    return spec_freqs, median_psds


def _state_psds_from_2d_spectrogram_average(
    series: np.array, fs: float, segmentation: np.array, frequency_resolution: int
):
    if len(segmentation) != len(series):
        raise ValueError("Segmentation must be same length as series array")

    # Remove low frequency component through highpass filter at 20 Hz
    series = series - np.mean(series)
    series = spectrum.bandpass(series, fs, low=20, high=None, order=4, zero_phase=True).copy()

    # Compute spectrogram window parameters to achieve desired frequency resolution
    required_window_length = 1 / frequency_resolution
    required_window_step = required_window_length / 2  # 50% overlap for Hann window

    # Use segmentation to extract state sections from audio
    # And then align the S1 and S2 sections by finding a peak in their cross correlation
    extracted_sections = {
        state: extract_state_sections(
            series,
            fs,
            segmentation,
            state,
            align=state in ["S1", "S2"],
            window_step=required_window_step,
            window_length=required_window_length,
            pad=True,
            pad_quantile=0.25,
        )
        for state in STATE_ORDER
    }

    # Compute spectrogram for each individual state section
    # And then take a median of all sound sections to generate averaged spectrogram
    hb_spectrograms, f = _compute_median_spectrograms_from_states(
        extracted_sections, fs, required_window_length, required_window_step
    )
    return f, {state: spectrogram.mean(axis=1) for state, spectrogram in hb_spectrograms.items()}


def _compute_median_spectrograms_from_states(
    extracted_states,
    fs,
    window_length,
    window_step,  # NB: not segmentation window length..
):
    hb_specs = {}
    for state, result in extracted_states.items():
        state_specs = []
        for r in result:
            spec, _, f, =  spectrum.spectrogram(r, fs, window_length, window_step)
            state_specs.append(spec)
        hb_specs[state] = np.median(state_specs, axis=0)
    return hb_specs, f
