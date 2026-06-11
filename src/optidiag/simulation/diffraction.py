"""Fraunhofer diffraction simulation for six aperture types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from optidiag.constants import IMAGE_TYPES


@dataclass
class DiffractionSimulator:
    """Generate normalized Fraunhofer diffraction intensity images."""

    image_size: int = 256
    grid_size: int = 512
    wavelength_nm: float = 632.8
    focal_length_mm: float = 200.0

    def _coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        axis = np.linspace(-1.0, 1.0, self.grid_size, dtype=np.float32)
        return np.meshgrid(axis, axis)

    def random_params(self, image_type: str, rng: np.random.Generator) -> Dict[str, float]:
        """Sample physically plausible aperture parameters for a class."""
        if image_type not in IMAGE_TYPES:
            raise ValueError(f"Unsupported image_type: {image_type}")

        if image_type == "single_slit":
            width = float(rng.uniform(0.025, 0.11))
            return {"width": width, "height": float(rng.uniform(1.1, 1.8))}
        if image_type == "double_slit":
            width = float(rng.uniform(0.02, 0.08))
            return {
                "width": width,
                "height": float(rng.uniform(1.0, 1.7)),
                "distance": float(width * rng.uniform(2.0, 4.5)),
            }
        if image_type == "triple_slit":
            width = float(rng.uniform(0.018, 0.07))
            return {
                "width": width,
                "height": float(rng.uniform(1.0, 1.6)),
                "distance": float(width * rng.uniform(2.0, 3.8)),
            }
        if image_type == "rectangle_aperture":
            return {
                "width": float(rng.uniform(0.08, 0.32)),
                "height": float(rng.uniform(0.08, 0.32)),
            }
        if image_type == "circular_aperture":
            return {"radius": float(rng.uniform(0.06, 0.18))}

        radius = float(rng.uniform(0.045, 0.15))
        return {
            "radius": radius,
            "distance": float(radius * rng.uniform(2.2, 4.6)),
        }

    def create_aperture(self, image_type: str, params: Dict[str, float]) -> np.ndarray:
        """Create a binary aperture mask."""
        x, y = self._coordinates()
        aperture = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if image_type == "single_slit":
            aperture = ((np.abs(x) <= params["width"]) & (np.abs(y) <= params["height"])).astype(np.float32)
        elif image_type == "double_slit":
            half_distance = params["distance"] / 2.0
            slit1 = (np.abs(x - half_distance) <= params["width"]) & (np.abs(y) <= params["height"])
            slit2 = (np.abs(x + half_distance) <= params["width"]) & (np.abs(y) <= params["height"])
            aperture = (slit1 | slit2).astype(np.float32)
        elif image_type == "triple_slit":
            distance = params["distance"]
            center = (np.abs(x) <= params["width"]) & (np.abs(y) <= params["height"])
            left = (np.abs(x + distance) <= params["width"]) & (np.abs(y) <= params["height"])
            right = (np.abs(x - distance) <= params["width"]) & (np.abs(y) <= params["height"])
            aperture = (left | center | right).astype(np.float32)
        elif image_type == "rectangle_aperture":
            aperture = ((np.abs(x) <= params["width"]) & (np.abs(y) <= params["height"])).astype(np.float32)
        elif image_type == "circular_aperture":
            aperture = (np.sqrt(x**2 + y**2) <= params["radius"]).astype(np.float32)
        elif image_type == "double_circular_aperture":
            half_distance = params["distance"] / 2.0
            circle1 = np.sqrt((x - half_distance) ** 2 + y**2) <= params["radius"]
            circle2 = np.sqrt((x + half_distance) ** 2 + y**2) <= params["radius"]
            aperture = (circle1 | circle2).astype(np.float32)
        else:
            raise ValueError(f"Unsupported image_type: {image_type}")

        return aperture

    def calculate_diffraction(self, aperture: np.ndarray, gamma: float = 0.32) -> np.ndarray:
        """Calculate a normalized FFT-based far-field diffraction pattern."""
        field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(aperture)))
        intensity = np.abs(field) ** 2
        intensity = np.log1p(intensity)
        intensity -= intensity.min()
        peak = intensity.max()
        if peak > 0:
            intensity /= peak
        intensity = np.power(np.clip(intensity, 0.0, 1.0), gamma)
        return intensity.astype(np.float32)

    def crop_to_image_size(self, image: np.ndarray) -> np.ndarray:
        """Center-crop the simulation grid to the configured output size."""
        h, w = image.shape
        size = min(self.image_size, h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        cropped = image[y0 : y0 + size, x0 : x0 + size]
        if size == self.image_size:
            return cropped.astype(np.float32)

        output = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        y_pad = (self.image_size - size) // 2
        x_pad = (self.image_size - size) // 2
        output[y_pad : y_pad + size, x_pad : x_pad + size] = cropped
        return output

    def render(self, image_type: str, params: Optional[Dict[str, float]] = None, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, Dict[str, float]]:
        """Render one clean diffraction image and return the image plus parameters."""
        rng = rng or np.random.default_rng()
        params = params or self.random_params(image_type, rng)
        aperture = self.create_aperture(image_type, params)
        clean = self.calculate_diffraction(aperture, gamma=float(rng.uniform(0.25, 0.42)))
        return self.crop_to_image_size(clean), params

