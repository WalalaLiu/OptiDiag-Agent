"""Shared labels for diffraction image diagnosis."""

IMAGE_TYPES = [
    "single_slit",
    "double_slit",
    "triple_slit",
    "rectangle_aperture",
    "circular_aperture",
    "double_circular_aperture",
]

ISSUE_TYPES = [
    "over_exposure",
    "under_exposure",
    "gaussian_noise",
    "poisson_noise",
    "dark_noise",
    "background_gradient",
    "blur_defocus",
    "misalignment",
    "rotation_tilt",
    "cropping_incomplete",
    "low_contrast",
]

LEGACY_TO_MVP_IMAGE_TYPE = {
    "single_slit": "single_slit",
    "double_slit": "double_slit",
    "triple_slit": "triple_slit",
    "rectangle": "rectangle_aperture",
    "circle": "circular_aperture",
    "double_circle": "double_circular_aperture",
}

SUGGESTION_LABELS = {
    "over_exposure": "reduce_exposure",
    "under_exposure": "increase_exposure",
    "gaussian_noise": "improve_signal_average",
    "poisson_noise": "increase_photon_budget",
    "dark_noise": "capture_dark_frame",
    "background_gradient": "shield_background_light",
    "blur_defocus": "refocus_system",
    "misalignment": "realign_optical_axis",
    "rotation_tilt": "level_aperture_camera",
    "cropping_incomplete": "recenter_and_reframe",
    "low_contrast": "improve_fringe_contrast",
}

