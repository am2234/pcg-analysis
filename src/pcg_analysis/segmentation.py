import itertools
import operator

import torch
import numpy as np
import pandas as pd
import scipy.signal

from pcg_analysis.spectrum import get_window
from pcg_analysis.utils import view_as_windows


# Mapping of state indexes to string names
INDICES_TO_STATES = {
    0: "S1",
    1: "systole",
    2: "S2",
    3: "diastole",
    4: "murmur",
    -1: "noise",
    -2: "unspecified",
}
STATES_TO_INDICES = {v: k for k, v in INDICES_TO_STATES.items()}


def scale_segmentation_to_series(
    segmentation: np.array,
    series_fs: float,
    series_length: int,
    seg_window_length: float,
    seg_window_step: float,
):
    segmentation = np.asarray(segmentation)
    if len(segmentation) == series_length:
        print("WARNING: No need to scale segmentation as already same length as series.")
        return segmentation

    scaled_segmentation = scale_segmentation_using_window_voting(
        segmentation, series_fs, series_length, seg_window_length, seg_window_step
    )

    if len(scaled_segmentation) < series_length:
        scaled_segmentation = np.pad(
            scaled_segmentation,
            (0, series_length - len(scaled_segmentation)),
            mode="constant",
            constant_values=-2,
        )

    return scaled_segmentation


def scale_segmentation_using_window_voting(
    segmentation: np.array,
    series_fs: float,
    series_length: int,
    seg_window_length: float,
    seg_window_step: float,
):
    # Original segmentation window parameters, in original recording sampling rate (e.g. 4 kHz)
    seg_step_samples = int(seg_window_step * series_fs)
    seg_window_samples = int(seg_window_length * series_fs)

    # Calculate series length that segmentation was performed for
    # Note this may be less than the actual recording length (series_length), because end samples
    # may have been truncated because they didn't fill a window
    segmented_series_length = ((len(segmentation) - 1) * seg_step_samples) + seg_window_samples
    if segmented_series_length > series_length:
        raise ValueError("Expected segmented length is greater than original series length")

    # Iterate through segmentation, and for each frame add some 'ownership' to time-series samples
    # weighted by the original segmentation window
    # NB: we need to add a small epsilon because windows start at zero and we need some ownership
    eps = 1e-7
    hann_window = np.asarray(get_window("hann", seg_window_samples, dtype=torch.double))
    unique_classes = list(np.unique(np.asarray(segmentation)))  # can contain murmur or noise state
    vote_array = np.zeros((len(unique_classes), segmented_series_length), dtype=np.float64)
    for i, value in enumerate(segmentation):
        win_start_sample = i * seg_step_samples  # int(np.round(series_fs * i * seg_window_step))
        win_end_sample = win_start_sample + seg_window_samples
        vote_array[unique_classes.index(value), win_start_sample:win_end_sample] += (
            hann_window + (len(segmentation) - i) * eps
        )

    # Check that all samples got atleast one vote from a window (an earlier bug)
    assert not (vote_array == 0).all(axis=0).any(), "All vote array samples should get >= 1 vote"

    # Now pick the most voted for state at each time-series sample
    # and convert back to original state labels
    scaled_segmentation = np.argmax(vote_array, axis=0)
    scaled_segmentation = np.asarray(unique_classes)[scaled_segmentation]
    return scaled_segmentation


def map_windowed_signal_to_original(
    windowed_length,
    original_fs,
    window_length,
    window_step,
    window_function="hann",
):
    # NB: this will not always be the same as the 'voting' method for segmentation
    # because in that method, all windows are considered in the region, not just the maximum one
    # this makes a difference when overlap >50% and >2 windows contribute to decision
    if window_step / window_length < 0.5:
        print(
            "WARNING: window overlap is >50%. Upscale method may be same as segmentation upscale."
        )

    # Original segmentation window parameters, in original recording sampling rate (e.g. 4 kHz)
    step_samples = int(window_step * original_fs)
    window_samples = int(window_length * original_fs)
    window_values = np.asarray(get_window(window_function, window_samples))

    # For these specific window values, work out point at which maximum frame swaps to next
    # subtract small eps from second window, so second window loses ties
    # (this is because float comparison of equivalent numbers is not always reliable)
    eps = 1e-6
    test_windows = np.zeros((3, 2 * step_samples + window_samples))
    for i in range(3):
        start = step_samples * i
        test_windows[i, start : start + window_samples] = window_values + (3 - i) * eps
    peak_area = np.argmax(test_windows, axis=0) == 1
    peak_start = np.argmax(peak_area)
    # need small eps even on last as one of the window values at start is zero.

    # For main part of sequence (ignoring start and end), get window start points
    # and calculate which the maximum window frame for each time point
    # if a time point is represented by two windows of equal value, break ties to the earlier one
    segmented_series_length = get_original_signal_length(
        windowed_length, original_fs, window_length, window_step
    )
    indices = step_samples + np.arange(0, segmented_series_length) - peak_start
    window_starts = indices // step_samples
    window_starts = np.clip(window_starts, 0, windowed_length - 1)
    return window_starts


def apply_windows_to_segmentation(segmentation, fs, window_length, window_step, window_type="hann"):
    return apply_windows_to_signal(
        segmentation, fs, window_length, window_step, method="mode", window_type=window_type
    )


def apply_windows_to_signal(signal, fs, window_length, window_step, method, window_type="hann"):
    # Get windows in the same format as spectrogram() would
    window_samples = int(fs * window_length)
    step_samples = int(fs * window_step)

    if method == "mode":
        func = _pick_weighted_mode
    elif method == "mean":
        func = _weighted_mean
    else:
        raise NotImplementedError(f"No window reducing method {method}")

    # Compute the overlapping frames of the segmentation.
    # Within each frame, pick the state the occurs the most frequently, but weighted by the
    # the windowing function so that states in centre are correctly prioritised
    # (this replicates what happened during original segmentation and upscaling)
    windowed = view_as_windows(signal, window_samples, step_samples, allow_incomplete=False)
    window = np.asarray(
        get_window(window_type, window_samples) + 1e-6
    )  # epsilon for consistency with upscale
    return np.asarray([func(frame, weights=window) for frame in windowed])


def _pick_weighted_mode(array, weights):
    """Pick the most common value in 'array', but weighted by 'weights'"""
    unique_values = np.unique(array)
    if len(unique_values) == 1:
        return unique_values.item()
    weighted_sums = {value: weights[array == value].sum() for value in unique_values}
    return max(weighted_sums, key=weighted_sums.get)


def _weighted_mean(array, weights):
    return np.mean(array * weights)


def get_num_windows(array_length, fs, window_length, window_step):
    """Get number of complete overlapping windows (no partials allowed)"""
    frame_length = int(fs * window_length)
    frame_step = int(fs * window_step)
    return int(np.floor((array_length - frame_length) / frame_step) + 1)


def get_segmentable_length(array_length, fs, window_length, window_step):
    num_windows = get_num_windows(array_length, fs, window_length, window_step)
    frame_length = int(fs * window_length)
    frame_step = int(fs * window_step)
    return (num_windows - 1) * frame_step + frame_length


def get_original_signal_length(windowed_length, fs, window_length, window_step):
    return (windowed_length - 1) * int(fs * window_step) + int(fs * window_length)


def segmentation_array_to_table(segmentation: np.array):
    """Split segmentation into individual state sections"""
    # Get individual state sections of the segmentation array
    sound_sections = [
        (state, [i for i, value in it])
        for state, it in itertools.groupby(enumerate(segmentation), key=operator.itemgetter(1))
    ]

    # Get the start and end indices of each section
    sound_start_end = [
        (INDICES_TO_STATES[state], section[0], section[-1]) for state, section in sound_sections
    ]
    return pd.DataFrame(sound_start_end, columns=("state", "start", "end"))


def segmentation_table_to_array(table: pd.DataFrame) -> np.array:
    array_length = table.iloc[-1]["end"] + 1
    array = -99 * np.ones(array_length, dtype=int)
    for _, row in table.iterrows():
        array[row["start"] : row["end"] + 1] = STATES_TO_INDICES[row["state"]]

    assert not np.any(array == -99)
    assert np.all(np.isfinite(array))
    return array


def extract_state_sections(
    series: np.array,
    fs: int,
    segmentation: np.array,
    state: str,
    align=True,
    window_length=None,
    window_step=None,
    pad=True,
    pad_quantile=0.5,
):
    section_ends = _align_state_sections(
        series,
        fs,
        segmentation,
        state,
        align,
        window_length,
        window_step,
        pad,
        pad_quantile,
    )

    sections = []
    for start, end in section_ends:
        series_section = series[start:end]
        if len(series_section) == 0:
            raise ValueError("Empty section")
        sections.append(series_section)
    return sections


def _align_state_sections(
    series: np.array,
    fs: int,
    segmentation: np.array,
    state: str,
    align=True,
    window_length=None,
    window_step=None,
    pad=True,
    pad_quantile=0.5,
):
    """Extract aligned equal-length time series sections for one heart sound state"""
    if state not in {"S1", "systole", "S2", "diastole"}:
        raise ValueError(f"State must be one of S1, systole, S2, diastole. Got {state}.")

    if align and not pad:
        raise ValueError("You must enable padding if you are aligning")

    segmentation = np.asarray(segmentation.copy())
    segmentation[segmentation == 4] = 1
    if len(segmentation) != len(series):
        raise ValueError("Segmentation must be same length as series")

    seg_df = segmentation_array_to_table(segmentation)
    seg_df["duration"] = seg_df["end"] + 1 - seg_df["start"]

    # Remove start or end states from spectrogram as they are 'incomplete'
    # (first, remove unspecified last state if it exists, as this means second-to-last incomplete)
    if seg_df.iloc[-1]["state"] == "unspecified":
        seg_df = seg_df.iloc[:-1]
    seg_df = seg_df.iloc[1:-1]
    seg_df_state = seg_df[seg_df["state"] == state].copy()

    # Filter away very short states
    seg_df_state = seg_df_state[seg_df_state["duration"] > int(0.025 * fs)]

    quantile_state_length = int(seg_df_state["duration"].quantile(pad_quantile))
    if window_step is not None:
        window_samples = int(fs * window_length)
        step_samples = int(fs * window_step)
        num_windows = int(np.floor((quantile_state_length - window_samples) / step_samples) + 1)
        quantile_state_length = (num_windows * step_samples) + window_samples

    sections = []
    for _, row in seg_df_state.iterrows():
        start = row["start"]
        if pad:
            end = start + quantile_state_length
        else:
            end = row["end"] + 1
        if end > len(series):
            continue
        sections.append((start, end))
    sections = np.asarray(sections)

    if not align:
        return sections
    else:
        # Added padding to sections, and remove any padded sections
        # that would now overlap with start/end of sequence
        padding = int(0.02 * fs)  # 20 ms
        padded_sections = sections + [-padding, padding]
        padded_sections = padded_sections[
            ((padded_sections >= 0) & (padded_sections < len(series))).all(axis=1)
        ]

        # Collate all padded audio segments and compute the unpadded median reference template
        series_sections = np.stack([series[start:end] for start, end in padded_sections])
        series_reference = np.median(series_sections, axis=0, keepdims=True)[:, padding:-padding]

        # Cross-correlate all segments with unpadded reference to find best alignment of signals
        correlation = scipy.signal.correlate(series_sections, series_reference, mode="valid")
        offsets = (correlation.shape[1] // 2) - np.argmax(correlation, axis=1, keepdims=True)

        # Finally add offsets to section indices and remove padding
        padded_sections = padded_sections - offsets
        return padded_sections + [padding, -padding]
