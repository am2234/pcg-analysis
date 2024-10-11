import numpy as np
import pytest

from pcg_analysis import snr
from pcg_analysis.segmentation import scale_segmentation_to_series


@pytest.fixture
def test_data_folder(request):
    return request.config.rootpath / "tests" / "test_data"


def test_full_snr_calculation(test_data_folder):
    """Test that the SNR function with default parameters returns expected value

    This is recording 8074 from the patch database, and its segmentation.
    """
    array = np.loadtxt(test_data_folder / "adxl_series.csv")
    fs = 2000
    segmentation = np.loadtxt(test_data_folder / "adxl_segmentation.csv")
    segmentation = segmentation.astype(int)
    segmentation = scale_segmentation_to_series(segmentation, fs, len(array), 0.025, 0.010)

    snr_result = snr.signal_to_noise_ratio(array, fs, segmentation)  # use default parameters
    assert np.isclose(snr_result, 19.4989)


def test_snr_variable_sound_state(test_data_folder):
    array = np.loadtxt(test_data_folder / "adxl_series.csv")
    fs = 2000
    segmentation = np.loadtxt(test_data_folder / "adxl_segmentation.csv")
    segmentation = segmentation.astype(int)
    segmentation = scale_segmentation_to_series(segmentation, fs, len(array), 0.025, 0.010)

    snr_S1 = snr.signal_to_noise_ratio(array, fs, segmentation, sound_states=["S1"])
    snr_S2 = snr.signal_to_noise_ratio(array, fs, segmentation, sound_states=["S2"])
    assert np.isclose(snr_S1, 16.2074)
    assert np.isclose(snr_S2, 18.7700)


def _make_test_wave():
    fs = 2000  # Hz
    segmentation_fs = 100  # Hz, to match 0.01 ms window step
    freq = 120  # Hz
    duration = 10  # seconds
    sound_amplitude = 100
    noise_amplitude = 4
    expected_snr_ratio = 10 * np.log10(sound_amplitude**2 / noise_amplitude**2)

    # Make fake time series.
    # First 5 seconds are 140 Hz sine wave with amplitude = 100
    # Last 5 seconds are 140 Hz sine wave amplitude = 1
    t = np.linspace(0, duration, fs * duration)
    sine_sound = sound_amplitude * np.sin(2 * np.pi * freq * t[: len(t) // 2])
    sine_noise = noise_amplitude * np.sin(2 * np.pi * freq * t[len(t) // 2 :])
    series = np.concatenate([sine_sound, sine_noise])

    # Make test segmentation to match the above series
    segmentation = np.ones((segmentation_fs * duration) - 10)
    segmentation[:500] = 0  # High amplitude sine wave is S1
    segmentation[500:] = 3  # Low amplitude sine wave is diastole
    segmentation[490:510] = -1  # Don't use discontinuity in middle of signal
    segmentation[:50] = 1  # Need 'start' and 'end' incomplete states to throwaway
    segmentation[-30:] = 1

    return series, fs, segmentation, expected_snr_ratio


def test_old_snr_sine_wave():
    """Test SNR results make physical sense with simple sine wave example"""
    series, fs, segmentation, expected_snr_ratio = _make_test_wave()
    segmentation = scale_segmentation_to_series(segmentation, fs, len(series), 0.025, 0.010)

    # Use better frequency resolution than typical to improve estimate of peak
    snr_result = snr.signal_to_noise_ratio(
        series, fs, segmentation, frequency_resolution=5, method="median_average"
    )
    assert np.isclose(snr_result, expected_snr_ratio, atol=0.1, rtol=0)  # allow 0.1 dB error


def test_max_old_snr_sine_wave():
    """Test max SNR finding function locates the correct frequencies"""
    series, fs, segmentation, _ = _make_test_wave()
    segmentation = scale_segmentation_to_series(segmentation, fs, len(series), 0.025, 0.010)

    # Add some small random noise to the entire signal
    # Otherwise SNR is very high throughout signal because of comparing tiny amplitudes
    series = series + 2 * np.random.randn(len(series))

    snr_vs_freq, f = snr.signal_to_noise_ratio_vs_frequency(
        series, fs, segmentation, frequency_resolution=5, method="median_average"
    )
    _, f_min, f_max = snr.find_max_snr(f, snr_vs_freq, width=10, f_min=20, f_max=1000)

    # Signal is a 120 Hz sine wave, so we expect the 10 Hz max band to be centred around this
    assert f_min == 115
    assert f_max == 125
