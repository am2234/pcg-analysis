import numpy as np

from pcg_analysis import spectrum


def test_welch_power_value():
    """Test that the Welch PSD function calculates the correct power in a sine wave"""
    fs = 2000
    duration = 5
    freq = 121  # arbitrary
    amplitude = 23  # arbitrary
    t = np.linspace(0, duration, fs * duration)
    sine_wave = amplitude * np.sin(2 * np.pi * freq * t)

    s_xx, _ = spectrum.welch(sine_wave, fs, window_length=1, window_step=0.5)

    expected_power = amplitude**2 / 2  # theoretical power of a sine wave
    calculated_power = sum(s_xx)
    assert np.isclose(expected_power, calculated_power), "Welch-calculated powers should match"
