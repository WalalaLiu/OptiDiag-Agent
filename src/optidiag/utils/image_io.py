"""Image loading and saving helpers."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image


ImageLike = Union[str, Path, BinaryIO, Image.Image, np.ndarray]


def to_float_gray(image: ImageLike) -> np.ndarray:
    """Convert an image-like object to a grayscale float array in [0, 1]."""
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        if arr.max(initial=0) > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(image)

    arr = np.asarray(pil_image.convert("L"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def array_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a float array in [0, 1] to a grayscale PIL image."""
    arr = np.clip(image, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def save_gray_png(image: np.ndarray, path: Union[str, Path]) -> None:
    """Save a grayscale float array as a PNG file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array_to_pil(image).save(output_path)


def load_gray_image(path: Union[str, Path]) -> np.ndarray:
    """Load a file path as a grayscale float array."""
    return to_float_gray(Path(path))
