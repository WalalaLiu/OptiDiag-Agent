"""API image loading helpers."""

from __future__ import annotations

import io
import urllib.request
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

from optidiag.utils.image_io import to_float_gray


async def read_upload_file(file: UploadFile) -> object:
    """Read an uploaded image file as a PIL image."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        return Image.open(io.BytesIO(content)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法读取上传图片: {exc}") from exc


def read_image_url(image_url: str, timeout: float = 8.0) -> object:
    """Download an image URL and return a PIL image."""
    if not image_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="image_url 必须是 http 或 https 地址")
    try:
        request = urllib.request.Request(image_url, headers={"User-Agent": "OptiDiag-Agent/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-provided analysis URL
            data = response.read(max(8 * 1024 * 1024, 1))
        return Image.open(io.BytesIO(data)).convert("L")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法下载或读取 image_url: {exc}") from exc


async def load_request_image(file: Optional[UploadFile], image_url: Optional[str]) -> object:
    """Load request image with file taking precedence over image_url."""
    if file is not None:
        return await read_upload_file(file)
    if image_url:
        return read_image_url(image_url)
    raise HTTPException(status_code=400, detail="请提供 file 上传字段或 image_url；两者同时存在时优先使用 file")


def ensure_supported_experiment(experiment_type: str) -> str:
    """Validate experiment type for the MVP."""
    normalized = (experiment_type or "diffraction").strip().lower()
    if normalized != "diffraction":
        raise HTTPException(status_code=400, detail="MVP 阶段仅支持 experiment_type=diffraction")
    return normalized


def pil_to_array(image: object):
    """Convert a PIL image-like object to a float array."""
    return to_float_gray(image)

