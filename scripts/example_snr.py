""" SNR calculation for an example heart sound signal """

import numpy as np

from pcg_analysis.utils import PROJECT_ROOT
from pcg_analysis.snr import signal_to_noise_ratio
from pcg_analysis.segmentation import scale_segmentation_to_series

DATA_FOLDER = PROJECT_ROOT / "tests" / "test_data"

# Load in example heart sound signal
audio_series = np.loadtxt(DATA_FOLDER / "adxl_series.csv")
fs = 2000

# Load in corresponding example heart sound segmentation
# Heart sound states are coded numerically:
# 0: S1, 1: systole, 2: S2, 3: diastole
segmentation = np.loadtxt(DATA_FOLDER / "adxl_segmentation.csv")
segmentation = segmentation.astype(int)
segmentation = scale_segmentation_to_series(
    segmentation, fs, len(audio_series), seg_window_length=0.025, seg_window_step=0.010
)

# Calculate SNR using 2D-spectrogram averaging method
snr = signal_to_noise_ratio(audio_series, fs, segmentation)
print(f"SNR = {snr:.1f} dB")
