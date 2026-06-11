import numpy as np

from optidiag.analysis.diffraction_metrics import compute_diffraction_metrics


def test_compute_diffraction_metrics_has_required_keys():
    image = np.zeros((64, 64), dtype=np.float32)
    image[28:36, :] = 0.7
    image[:, 30:34] = 1.0

    metrics = compute_diffraction_metrics(image)

    required = {
        "mean_intensity",
        "std_intensity",
        "contrast",
        "snr_estimate",
        "saturation_ratio",
        "dark_pixel_ratio",
        "laplacian_variance",
        "center_of_mass",
        "center_offset_px",
        "background_uniformity",
        "fringe_visibility",
        "fft_peak_strength",
        "main_lobe_width_estimate",
    }
    assert required.issubset(metrics.keys())
    assert isinstance(metrics["center_offset_px"], list)
    assert 0.0 <= metrics["fringe_visibility"] <= 1.0

