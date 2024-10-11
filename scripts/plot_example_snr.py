""" Plot SNR calculation for an example recording """

from typing import Dict, Sequence
import itertools

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pcg_analysis.utils import PROJECT_ROOT
from pcg_analysis.plots import draw_segmentation_legend, plot_segmented_series
from pcg_analysis.segmentation import extract_state_sections, scale_segmentation_to_series
from pcg_analysis.snr import (
    _compute_median_spectrograms_from_states,
    calculate_averaged_state_psds,
    calculate_snr_in_frequency_band,
    extract_noise_and_signal_psds,
    signal_to_noise_ratio,
)
from pcg_analysis.spectrum import bandpass

FREQ_RESOLUTION = 25
STATE_ORDER = ["S1", "systole", "S2", "diastole"]
WINDOW_LENGTH = 1 / FREQ_RESOLUTION
WINDOW_STEP = WINDOW_LENGTH / 2
DATA_FOLDER = PROJECT_ROOT / "tests" / "test_data"

DPI = 100
FONT_SIZE = 8
VMIN = 0
F_MAX = 600


def main():
    audio_series = np.loadtxt(DATA_FOLDER / "adxl_series.csv")
    fs = 2000
    segmentation = np.loadtxt(DATA_FOLDER / "adxl_segmentation.csv")
    segmentation = segmentation.astype(int)
    segmentation = scale_segmentation_to_series(segmentation, fs, len(audio_series), 0.025, 0.010)

    plot_new_snr(audio_series, fs, segmentation)


def plot_new_snr(series, fs, segmentation_array, vmax=None):

    signal_to_noise_ratio(series, fs, segmentation_array)

    # Do this first before any pre-processing to the series
    # As this will be handled by the actual SNR function
    f, state_psds = calculate_averaged_state_psds(
        series, fs, segmentation_array, FREQ_RESOLUTION, method="2d_spectrogram_average"
    )

    series = series - np.mean(series)
    series = bandpass(series, fs, low=20, high=None, order=4, zero_phase=True).copy()

    extracted_sections = {
        state: extract_state_sections(
            series,
            fs,
            segmentation_array,
            state,
            align=state in ["S1", "S2"],
            window_step=WINDOW_STEP,
            window_length=WINDOW_LENGTH,
            pad=True,
            pad_quantile=0.25,
        )
        for state in STATE_ORDER
    }

    fig = plt.figure(figsize=(1920 / DPI, 1080 / DPI), dpi=DPI)

    axes = fig.add_subplot(4, 1, 1)
    _plot_series_and_segmentation(series, fs, segmentation_array, axes)

    axes = fig.add_subplot(4, 1, 2)
    _plot_aligned_sections(fs, extracted_sections, axes)

    hb_spectrograms, f = _compute_median_spectrograms_from_states(
        extracted_sections, fs, WINDOW_LENGTH, WINDOW_STEP
    )

    concat_spec = np.concatenate([hb_spectrograms[k] for k in STATE_ORDER], axis=1)

    offset = 0
    times, section_time_intervals = [], [0]
    for state in STATE_ORDER:
        spec = hb_spectrograms[state]
        length = spec.shape[1]
        section_time = offset + np.arange(length) * WINDOW_STEP  # spec_fs
        times.extend(section_time)
        offset += length * WINDOW_STEP + WINDOW_LENGTH - (section_time[-1] - section_time[-2])
        section_time_intervals.append(section_time[-1] + WINDOW_LENGTH)
    times.append(times[-1] + WINDOW_LENGTH)

    axes = fig.add_subplot(4, 1, 3)

    f_plot = np.append(f, 2 * f[-1] - f[-2])
    T, F = np.meshgrid(times, f_plot)
    mesh = axes.pcolormesh(T, F, 10 * np.log10(concat_spec), vmin=VMIN, vmax=None)

    ax1_divider = make_axes_locatable(axes)
    cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    fig.colorbar(mesh, cax=cax1, label="Power (dB/Hz)")

    for (start, end), state in zip(pairwise(section_time_intervals), STATE_ORDER):
        if state != "diastole":
            axes.axvline(end, c="white")
        axes.text(
            0.5 * (start + end),
            0.96,
            state,
            c="white",
            ha="center",
            va="top",
            size=FONT_SIZE,
            transform=axes.get_xaxis_transform(),
        )
    axes.set_ylabel("Frequency (Hz)")
    axes.set_xlabel("Time (seconds)")
    axes.set_ylim(0, F_MAX)

    axes = fig.add_subplot(4, 2, 7)
    for state in ["S1", "S2", "systole", "diastole"]:
        state_psd = state_psds[state]
        axes.plot(
            f,
            10 * np.log10(state_psd),
            c=cols[state],
            ls="--" if state in ["diastole", "S2"] else "-",
            label=state.title(),
        )
    axes.set_xlim(0, F_MAX)
    axes.set_ylim(VMIN)
    axes.set_ylabel("PSD (dB/Hz)")
    axes.set_xlabel("Frequency (Hz)")
    axes.legend(fontsize=FONT_SIZE)

    axes = fig.add_subplot(4, 2, 8)

    sound_psd, noise_psd = extract_noise_and_signal_psds(state_psds)
    snr_vs_freq = sound_psd / noise_psd

    axes.plot(f, 10 * np.log10(sound_psd), label="Signal")
    axes.plot(f, 10 * np.log10(noise_psd), label="Noise")
    axes.set_xlim(0, F_MAX)
    axes.set_ylim(VMIN)
    axes.set_ylabel("PSD (dB/Hz)")
    axes.set_xlabel("Frequency (Hz)")

    f_index = (f >= 25) & (f < 200)
    axes.fill_between(
        f[f_index],
        10 * np.log10(noise_psd[f_index]),
        10 * np.log10(sound_psd[f_index]),
        label="Signal to noise gap",
        facecolor="lightgrey",
    )
    axes.legend(fontsize=FONT_SIZE)

    snr = calculate_snr_in_frequency_band(f, snr_vs_freq, f_min=25, f_max=200)
    print(f"SNR = {snr:.1f} dB")

    fig.align_ylabels()
    fig.tight_layout()

    text_args = dict(fontweight="bold", va="top", transform=fig.transFigure, size=FONT_SIZE)

    fig.text(0.01, 0.97, "a", **text_args)
    fig.text(0.01, 0.73, "b", **text_args)
    fig.text(0.01, 0.50, "c", **text_args)
    fig.text(0.01, 0.255, "d", **text_args)
    fig.text(0.52, 0.255, "e", **text_args)

    plt.show()


def _plot_aligned_sections(fs, extracted_sections, axes):
    plot_averaged_phonocardiogram_from_states(axes, extracted_sections, fs, font_size=FONT_SIZE)
    axes.autoscale(axis="x", tight=True)
    axes.set_ylabel("Amplitude (V)")
    axes.set_xlabel("Time (seconds)")

    ax1_divider = make_axes_locatable(axes)
    cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
    cax1.axis("off")


def _plot_series_and_segmentation(series, fs, segmentation_array, axes):
    plot_segmented_series(axes, series, fs, segmentation_array, lw=1, legend=False)
    axes.autoscale(axis="y")
    axes.set_ylabel("Amplitude (V)")
    axes.set_xlabel("Time (seconds)")
    draw_segmentation_legend(
        axes,
        segmentation_array,
        fontsize=FONT_SIZE,
        loc="lower right",
        bbox_to_anchor=(1, 1.06),
        borderaxespad=0,
        edgecolor="k",
    )


def pairwise(iterable):
    # as not in current python version:
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


cols = {"S1": "#453c66", "S2": "#00a35f", "systole": "#fac682", "diastole": "lightgrey"}


def plot_averaged_phonocardiogram_from_states(
    axes: plt.Axes, extracted_states: Dict[str, Sequence[np.array]], fs, font_size=8, lw=None
):
    start_point = 0
    for state in ["S1", "systole", "S2", "diastole"]:
        result = extracted_states[state]

        state_length = len(result[0])
        x = np.arange(start_point, start_point + state_length) / fs
        for r in result:
            axes.plot(x, r, c=cols[state], alpha=0.1, lw=lw)
        start_point += state_length

        if state != "diastole":
            axes.axvline(start_point / fs, c="gray")
        axes.plot(x, np.median(result, axis=0), c=cols[state], lw=lw)
        axes.text(
            x[len(x) // 2],
            0.96,
            state,
            ha="center",
            va="top",
            size=font_size,
            transform=axes.get_xaxis_transform(),
        )

    axes.set_ylabel("Amplitude")


if __name__ == "__main__":
    main()
