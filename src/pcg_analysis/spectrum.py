import numpy as np
import scipy.signal
import torch

__all__ = ["spectrogram", "welch"]


def spectrogram(series: np.array, fs: int, window_length, window_step, window="hann", nfft=None):
    # Turn quantities in seconds into samples
    nperseg = int(window_length * fs)
    nstep = int(window_step * fs)
    if nfft is None:
        nfft = nperseg

    # Call PyTorch version of spectrogram function for consistency across code
    spec = _left_spectrogram(
        x=torch.as_tensor(series, dtype=torch.float),
        fs=fs,
        win_length=nperseg,
        win_step=nstep,
        nfft=nfft,
        window=window,
    )

    spec_fs = fs / nstep
    return spec, spec_fs, _fft_freqs(fs, nfft)


def welch(
    series: np.array,
    fs: int,
    window_length: float,
    window_step: float,
    window: str = "hann",
    nfft: int = None,
):
    """Compute Welch estimate of power spectral density

    This method computes the spectrogram of a signal using overlapping time windows, and then
    averages the result across time to produce a reduced-noise estimate of the PSD.
    """
    # Compute spectrogram, and then average frames across time to get Welch PSD estimate
    spec, _, freqs = spectrogram(series, fs, window_length, window_step, window, nfft)
    s_xx = np.mean(np.asarray(spec), axis=-1)
    return s_xx, freqs


def bandpass(series: np.array, fs: int, low, high, order, zero_phase=False):
    """Bandpass filter a time series"""
    # SciPy doc recommends using second-order sections when filtering
    # to avoid numerical errors with b, a format.
    if low == 0:
        btype = "lowpass"
        critical_freqs = high
    elif high is None:
        btype = "highpass"
        critical_freqs = low
    else:
        critical_freqs = (low, high)
        btype = "bandpass"
    sos = scipy.signal.butter(order, critical_freqs, btype=btype, fs=fs, output="sos")
    filter_func = scipy.signal.sosfiltfilt if zero_phase else scipy.signal.sosfilt
    filtered = filter_func(sos, series)
    return filtered


def _fft_freqs(fs: int, nfft: int) -> torch.Tensor:
    """Compute FFT frequencies that correspond to bins

    Args:
        fs (int): sample rate of FFT'd signal (Hz)
        nfft (int): number of fft bins

    Returns:
        torch.Tensor: FFT frequencies
    """
    return torch.linspace(0.0, float(fs) / 2, nfft // 2 + 1)


def _left_stft(
    x: torch.Tensor,
    win_length: int,
    win_step: int,
    nfft: int,
    window: str,
    normalized: bool = False,
) -> torch.Tensor:
    """Compute left-aligned short-time Fourier transform (STFT) of tensor.

    In contrast to default torch.stft, the frames here are left-aligned so that frame 0 starts at
    time 0, instead of being centered on it. This also fixes torch.stft problem where frames
    are created using nfft not win_length, so frames could be lost at end.

    The number of output frames, num_frames can be calculated as
        `1 + (x.shape[1] - win_length) // win_step`

    Args:
        x (torch.Tensor): input tensor of shape [B, T]
        win_length (int): length of window
        win_step (int): step between window
        nfft (int): fft size
        window (str): type of window to use (hann, hamming, blackman, rectangular, bartlett)
        normalized (bool, optional): normalise STFT by FFT size. Defaults to False.

    Returns:
        torch.Tensor: time-frequency STFT, of shape [B,  nfft // 2 + 1, 2, num_frames]
    """
    # torch.stft reshapes into frames of size nfft (not win_length)
    # so need to pad end so actual final frames included when win_length < nfft
    # see librosa issues 596, 695, 733
    x = torch.nn.functional.pad(x, (0, nfft - win_length), mode="constant")

    # window created in stft is center-padded when it should be left-aligned
    # so create own window vector that is post-padded to nfft length.
    window_array = get_window(window, win_length, x.dtype, x.device)
    window_array = torch.nn.functional.pad(window_array, (0, nfft - win_length), mode="constant")

    stft = torch.stft(
        x,
        n_fft=nfft,
        hop_length=win_step,
        window=window_array,
        center=False,
        normalized=normalized,
        return_complex=True,
    )
    stft = torch.view_as_real(stft)  # convert back into old format with separate axis for [re, im]

    return stft.transpose(-1, -2)  # swap T and [re,im] dims so T is last.


def _left_spectrogram(
    x: torch.Tensor,
    fs: int,
    win_length: int,
    win_step: int,
    nfft: int,
    window: str,
    normalized: bool = False,
) -> torch.Tensor:
    """Compute left-aligned spectrogram of input tensor

    Args:
        x (torch.Tensor): input tensor of shape [B, T]
        win_length (int): length of window
        win_step (int): step between window
        nfft (int): fft size
        window (str): type of window to use (hann, hamming, blackman, rectangular, bartlett)
        normalized (bool, optional): normalise STFT by FFT size. Defaults to False.

    Returns:
        torch.Tensor: time-frequency STFT, of shape [B, nfft // 2 + 1, num_frames]
    """
    stft = _left_stft(x, win_length, win_step, nfft, window, normalized)
    spec = stft.pow(2).sum(dim=-2)

    # Apply normalisation factor to match SciPy and other literature
    window_array = get_window(window, win_length, x.dtype, x.device)
    normalisation = fs * torch.sum(window_array**2)
    spec /= normalisation

    # Include complex conjugate power in calculation
    if nfft % 2:
        spec[1:, :] *= 2
    else:
        spec[1:-1, :] *= 2
    return spec


def get_window(
    name: str, length: int, dtype: int = torch.float, device: torch.device = "cpu"
) -> torch.Tensor:
    """Create window vector for spectral analysis

    Note we use 'periodic' windows rather than 'symmetric' windows. Periodic windows are
    recommended for spectral analysis, whilst symmetric windows are used for filter design.
    For a periodic window, a window of length L+1 is created and then truncated to L samples.
    See e.g. https://uk.mathworks.com/help/signal/ref/hann.html

    Args:
        name (str): name of window type (hann, hamming, blackman, rectangular, bartlett)
        length (int): length of window
        dtype (int): dtype of signal
        device (str): device of signal

    Raises:
        ValueError: if window name is not one of the supported above

    Returns:
        torch.Tensor: window vector, shape [length]
    """
    # until JIT supports callables have to do a manual if/else
    if name == "hann":
        return torch.hann_window(length, periodic=True, dtype=dtype, device=device)
    elif name == "hamming":
        return torch.hamming_window(length, periodic=True, dtype=dtype, device=device)
    elif name == "blackman":
        return torch.blackman_window(length, periodic=True, dtype=dtype, device=device)
    elif name == "rectangular":
        return torch.ones(length, dtype=dtype, device=device)
    elif name == "bartlett":
        return torch.bartlett_window(length, periodic=True, dtype=dtype, device=device)
    else:
        raise ValueError("Unknown")
