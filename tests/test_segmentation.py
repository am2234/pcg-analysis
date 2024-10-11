import numpy as np
import pytest

from pcg_analysis import segmentation, utils


# Example (length, step) windowing parameters with 0 to 50% overlap.
# Some of the upscaling/downscaling match/reversible tests won't work for > 50% overlap.
# due to fundamental ways in which the algorithm works.
WINDOW_PARAMS = [
    (0.010, 0.010),
    (0.010, 0.015),
    (0.010, 0.020),
    (0.025, 0.025),
    (0.025, 0.040),
    (0.025, 0.050),
    (0.05, 0.05),
    (0.05, 0.075),
    (0.05, 0.1),
    (0.1, 0.1),
]
WINDOW_IDS = [f"step={x[0]*1000:.0f}ms, length={x[1]*1000:.0f}ms" for x in WINDOW_PARAMS]


@pytest.mark.parametrize("window_param", WINDOW_PARAMS, ids=WINDOW_IDS)
def test_upscaling_downscaling_reversible(window_param):
    window_step, window_length = window_param
    # won't always be the case for all segs. '1 state' can get lost if surrounded by two the same.
    seg = [4, 2, 2, 2, 3, 1, 1, 4, 1, 1, 2, 3, 3, 3, 3, 1, 2, 3, 3, 0, 4]  # no 1 state!
    series_fs = 4000

    result = segmentation.scale_segmentation_to_series(
        seg,
        series_fs=series_fs,
        series_length=segmentation.get_original_signal_length(
            len(seg), series_fs, window_length, window_step
        )
        + 5,
        seg_window_length=window_length,
        seg_window_step=window_step,
    )

    downscale = segmentation.apply_windows_to_segmentation(
        result, series_fs, window_length, window_step
    )
    np.testing.assert_array_equal(seg, downscale)


@pytest.mark.parametrize("window_param", WINDOW_PARAMS, ids=WINDOW_IDS)
def test_new_and_old_upscaling_same(window_param):
    window_step, window_length = window_param
    seg = np.asarray([4, 2, 2, 2, 3, 1, 1, 4, 1, 1, 2, 3, 3, 3, 3, 1, 2, 3, 3, 0, 4], dtype=int)
    series_fs = 4000
    series_length = segmentation.get_original_signal_length(
        len(seg), series_fs, window_length, window_step
    )

    old_result = segmentation.scale_segmentation_using_window_voting(
        seg, series_fs, series_length, window_length, window_step
    )

    mapping = segmentation.map_windowed_signal_to_original(
        len(seg), series_fs, window_length, window_step
    )
    new_result = seg[mapping]

    np.testing.assert_array_equal(old_result, new_result)


def test_upscaling_downscaling_random_seg():
    seg = np.random.randint(low=0, high=5, size=141)
    series_fs = 4000
    window_length = 0.025
    window_step = 0.0125
    result = segmentation.scale_segmentation_to_series(
        seg,
        series_fs=series_fs,
        series_length=segmentation.get_original_signal_length(
            len(seg), series_fs, window_length, window_step
        )
        + 5,
        seg_window_length=window_length,
        seg_window_step=window_step,
    )

    downscale = segmentation.apply_windows_to_segmentation(
        result, series_fs, window_length, window_step
    )
    np.testing.assert_array_equal(seg, downscale)


def test_view_as_windows():
    array = np.random.randint(low=0, high=5, size=100)
    window_shape = 10
    window_step = 4

    result = utils.view_as_windows(array, window_shape, window_step)

    num_windows = 1 + (len(array) - window_shape) // window_step
    assert len(result) == num_windows

    for i in range(num_windows):
        np.testing.assert_array_equal(
            result[i], array[i * window_step : i * window_step + window_shape]
        )


@pytest.fixture
def test_data_folder(request):
    return request.config.rootpath / "tests" / "test_data"


def test_upscaling_downscaling_reversible_2(test_data_folder):
    array = np.loadtxt(test_data_folder / "adxl_series.csv")
    fs = 2000
    seg = np.loadtxt(test_data_folder / "adxl_segmentation.csv")
    seg = seg.astype(int)

    window_length = 0.025
    window_step = 0.010

    result = segmentation.scale_segmentation_to_series(
        seg,
        series_fs=fs,
        series_length=len(array),
        seg_window_length=window_length,
        seg_window_step=window_step,
    )

    downscale = segmentation.apply_windows_to_segmentation(result, fs, window_length, window_step)
    np.testing.assert_array_equal(seg, downscale)


def test_array_to_table_reversible(test_data_folder):
    seg = np.loadtxt(test_data_folder / "adxl_segmentation.csv")
    seg = seg.astype(int)

    df = segmentation.segmentation_array_to_table(seg)
    reconstructed_seg = segmentation.segmentation_table_to_array(df)
    np.testing.assert_array_equal(seg, reconstructed_seg)
