""" Code to detect spikes in heart sound recordings """

from typing import List, Tuple

import numpy as np
import pandas as pd

import pcg_analysis.segmentation
import pcg_analysis.spectrum


def detect_spikes(
    series: np.array, fs: int, segmentation: pd.DataFrame, border: float = 20e-3, bandpassed=False
):
    """Find start/end point of spikes in segmented recording"""
    if not bandpassed:
        series = pcg_analysis.spectrum.bandpass(
            series, fs, low=20, high=600, order=4, zero_phase=True
        )

    state_thresholds = get_state_thresholds(series, fs, segmentation, bandpassed=True)
    thresholds = get_thresholds_over_time(segmentation, fs, state_thresholds, border)

    spike_array = np.zeros_like(series)
    spike_array[(series < thresholds[0]) | (series > thresholds[1])] = 1
    spikes = _greater_than_zero(spike_array)
    return spikes


def get_state_thresholds(
    series: np.array,
    fs: int,
    segmentation: pd.DataFrame,
    bandpassed: bool = False,
    quantile=0.98,
    multiplier=3,
    states=("S1", "systole", "S2", "diastole"),
):
    if not bandpassed:
        series = pcg_analysis.spectrum.bandpass(
            series, fs, low=20, high=600, order=4, zero_phase=True
        )
    thresholds = {}
    for state in states:
        sub_segmentation = segmentation[segmentation["state"] == state]
        all_values = []
        for _, row in sub_segmentation.iterrows():
            all_values.append(series[row["start"] : row["end"] + 1])
        all_values = np.concatenate(all_values)

        lq, hq = np.quantile(all_values, [1 - quantile, quantile]) * multiplier
        thresholds[state] = (lq, hq)

    return thresholds


def get_thresholds_over_time(segmentation, fs, state_thresholds, border: float = 20e-3):
    seg_array = pcg_analysis.segmentation.segmentation_table_to_array(segmentation)

    thresholds = np.zeros((2, len(seg_array)))
    for index, state in pcg_analysis.segmentation.INDICES_TO_STATES.items():
        if index < 0:
            state = "diastole"  # use diastolic quantiles for noise/unspecified
        if state == "murmur":
            continue  # ignore murmur state
        lq, hq = state_thresholds[state]
        thresholds[0, seg_array == index] = lq
        thresholds[1, seg_array == index] = hq

    assert not (thresholds == 0).any()

    thresholds = _add_border_to_thresholds(thresholds, segmentation, fs, border)

    return thresholds


def mark_spikes_in_segmentation(segmentation: pd.DataFrame, spikes: List[Tuple[int, int]]):
    segmentation = segmentation.copy()
    for index, row in segmentation.iterrows():
        for spike_start, spike_end in spikes:
            overlap_found = _overlap((row["start"], row["end"]), (spike_start, spike_end))
            if overlap_found:
                segmentation.at[index, "state"] = "noise"
                break
    return segmentation


def _add_border_to_thresholds(thresholds, segmentation, fs, border=20e-3):
    num_thresholds, threshold_length = thresholds.shape
    assert num_thresholds == 2

    border_size = int(border * fs)

    new_thresh = thresholds.copy()
    for _, row in segmentation.query("state in ['S1', 'S2']").iterrows():
        sound_start = row["start"]
        sound_end = row["end"]

        border_start = max(sound_start - border_size, 0)
        border_end = min(sound_end + border_size, threshold_length - 1)

        for i, thresh_line in enumerate(new_thresh):
            if sound_start > border_start:
                ramp = np.linspace(
                    thresholds[i, border_start],
                    thresholds[i, sound_start],
                    sound_start - border_start,
                )
                thresh_line[border_start:sound_start] = ramp
            if border_end > sound_end:
                ramp = np.linspace(
                    thresholds[i, sound_end], thresholds[i, border_end], border_end - sound_end
                )
                thresh_line[sound_end:border_end] = ramp
    return new_thresh


def _greater_than_zero(a):
    isntzero = np.concatenate(([0], np.greater(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isntzero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])
