import matplotlib.pyplot as plt
import numpy as np
from pcg_analysis.segmentation import (
    scale_segmentation_to_series,
    segmentation_array_to_table,
    INDICES_TO_STATES,
)
from matplotlib.collections import LineCollection
from . import snr

STATE_COLOURS = {
    "S1": "#453c66",
    "S2": "#00a35f",
    "diastole": "lightgrey",
    "systole": "#fac682",
    "murmur": "red",
    "noise": "orange",
    "unspecified": "orange",
}


def plot_segmented_series(
    axes: plt.Axes,
    series,
    fs,
    segmentation,
    seg_window_length=0.025,
    seg_window_step=0.010,
    t0=0,
    t1=None,
    legend: bool = True,
    legend_args: dict = None,
    **kwargs,
):
    if len(segmentation) != len(series):
        segmentation = scale_segmentation_to_series(
            segmentation, fs, len(series), seg_window_length, seg_window_step
        )

    if t1 is None:
        t1 = len(series) / fs
    series = series[int(t0 * fs) : int(t1 * fs) + 2]
    segmentation = segmentation[int(t0 * fs) : int(t1 * fs) + 2]

    seg_table = segmentation_array_to_table(segmentation)
    segments = []
    colors = []
    for _, row in seg_table.iterrows():
        start_index = row["start"]
        end_index = min(row["end"] + 2, len(series))  # add 1 to index so lines link up
        t = np.arange(start_index, end_index) / fs
        segments.append(np.stack([t, series[start_index:end_index]], axis=1))
        colors.append(STATE_COLOURS[row["state"]])

    collection = LineCollection(segments, colors=colors, **kwargs)
    axes.add_collection(collection)
    axes.set_xlim(0, len(series) / fs)

    if legend:
        if legend_args is None:
            legend_args = {}
        draw_segmentation_legend(axes, segmentation, **legend_args)


def draw_segmentation_legend(axes, segmentation=None, states=None, **kwargs):
    if segmentation is not None:
        indices = np.unique(segmentation, return_index=True)[1]
        states = [segmentation[index] for index in sorted(indices)]

    proxies, names = [], []
    for state in states:
        state_name = INDICES_TO_STATES[state]
        proxies.append(plt.Line2D([0, 1], [0, 1], color=STATE_COLOURS[state_name]))
        names.append(state_name.title())

    axes.legend(proxies, names, ncol=len(states), **kwargs)



state_cols = {
    "S1": "#453c66",
    "S2": "#00a35f",
    "diastole": "lightgrey",
    "systole": "#fac682",
    "murmur": "red",
    "noise": "orange",
    "unspecified": "orange",
}


def plot_old_snr_calculation(array: np.array,  fs: int, segmentation: np.array, fig=None, y_max=700, vmin=None):
    if fig is None:
        fig, axes = plt.subplots(2, 2, sharex="col")        
        ax_series, ax_spec = axes[0][0], axes[1][0]
        ax_psd, ax_snr = axes[0][1], axes[1][1]
    else:
        ax_series = fig.add_subplot(2,2,1)
        ax_spec = fig.add_subplot(2,2,3)
        ax_psd = fig.add_subplot(2,2,2)
        ax_snr = fig.add_subplot(2,2,4)

    # Axis 1 - segmented time series
    upscaled_segmentation = scale_segmentation_to_series(segmentation, fs, len(array), 0.025, 0.010)
    df_seg = segmentation_array_to_table(upscaled_segmentation)
    for _, row in df_seg.iterrows():
        start_index = row["start"]
        end_index = min(row["end"] + 2, len(array))
        t = np.arange(start_index, end_index) / fs  # add 1 to index so lines link up
        ax_series.plot(t, array[start_index:end_index], c=state_cols[row["state"]])
    ax_series.set_ylabel("Amplitude")

    # Axis 2 - spectrogram
    spec, freqs, win_len, win_step = snr.calculate_spectrogram_for_psd(
        array, fs, frequency_resolution=20
    )
    x = np.arange(spec.shape[-1]) * win_step
    ax_spec.pcolormesh(x, freqs, 10 * np.log10(spec), vmin=vmin)
    ax_spec.set_ylim(0, y_max)
    ax_spec.set_xlabel("Time (seconds)")
    ax_spec.set_ylabel("Frequency (Hz)")

    # Axis 3 - state PSDs
    f_freqs, state_psds = snr.calculate_state_psds(array, fs, segmentation, frequency_resolution=20)
    for state, psd in state_psds.items():
        ax_psd.plot(f_freqs, 10 * np.log10(np.stack(psd)).T, c=state_cols[state], alpha=0.1)
        ax_psd.plot(
            f_freqs,
            10 * np.log10(np.median(np.stack(psd), axis=0)),
            c=state_cols[state],
            label=state.title(),
        )
    ax_psd.legend()
    ax_psd.set_xlim(0, y_max)
    ax_psd.set_ylabel("Power spectral density (dB)")
    ax_psd.grid(alpha=0.2)

    # Axis 4 - SNR PSDs
    signal_psd, noise_psd = snr.extract_noise_and_signal_psds(state_psds)
    ax_snr.plot(f_freqs, 10 * np.log10(noise_psd), c="gray", label="Noise")
    ax_snr.plot(f_freqs, 10 * np.log10(signal_psd), c="blue", label="Signal")
    f_indices = (f_freqs >= 20) & (f_freqs <= 200)
    ax_snr.fill_between(
        f_freqs[f_indices],
        10 * np.log10(noise_psd)[f_indices],
        10 * np.log10(signal_psd)[f_indices],
        facecolor="green",
        alpha=0.3,
        label="SNR Gap",
    )
    ax_snr.legend()
    overall_snr = snr.signal_to_noise_ratio(array, fs, segmentation)
    ax_snr.text(1, 1, f"Overall SNR = {overall_snr:.1f} dB", transform=ax_snr.transAxes)
    ax_snr.set_xlabel("Frequency (Hz)")
    ax_snr.set_ylabel("Power spectral density (dB)")
    ax_snr.grid(alpha=0.2)
    ax_snr.set_xlim(0, y_max)

    fig.tight_layout(h_pad=0)
