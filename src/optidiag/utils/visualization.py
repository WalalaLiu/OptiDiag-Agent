"""Pillow-based visualizations for course-report demo cases."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from optidiag.utils.image_io import to_float_gray


FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a Chinese-capable font when available."""
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=1 if bold else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, width: int = 34) -> List[str]:
    """Wrap mixed Chinese/English text by character count."""
    lines: List[str] = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=True, replace_whitespace=False))
    return lines


def _format_issues(issues: Sequence[Dict[str, object]]) -> str:
    if not issues:
        return "未发现明显问题"
    return "；".join(f"{item['type']}({item['severity']}, {float(item['score']):.2f})" for item in issues[:3])


def _metric_lines(metrics: Dict[str, object]) -> List[str]:
    center_offset = metrics.get("center_offset_px", [0.0, 0.0])
    return [
        f"饱和像素比例: {float(metrics.get('saturation_ratio', 0.0)):.4f}",
        f"中心偏移: ({float(center_offset[0]):.1f}, {float(center_offset[1]):.1f}) px",
        f"条纹可见度: {float(metrics.get('fringe_visibility', 0.0)):.3f}",
        f"拉普拉斯方差: {float(metrics.get('laplacian_variance', 0.0)):.5f}",
        f"信噪比估计: {float(metrics.get('snr_estimate', 0.0)):.3f}",
        f"背景均匀性: {float(metrics.get('background_uniformity', 0.0)):.3f}",
    ]


def _draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: Iterable[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (40, 40, 40),
    line_gap: int = 8,
) -> int:
    """Draw text lines and return the next y coordinate."""
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or "A", font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def _render_image_panel(image: object, size: int = 520) -> Image.Image:
    """Render image with center cross and center-of-mass marker."""
    arr = to_float_gray(image)
    h, w = arr.shape
    pil = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(pil)
    center_x = size / 2.0
    center_y = size / 2.0
    draw.line((center_x, 0, center_x, size), fill=(255, 80, 80), width=2)
    draw.line((0, center_y, size, center_y), fill=(255, 80, 80), width=2)

    total = float(arr.sum())
    if total > 1e-8:
        y_idx, x_idx = np.indices(arr.shape)
        com_x = float((x_idx * arr).sum() / total) * size / max(1, w - 1)
        com_y = float((y_idx * arr).sum() / total) * size / max(1, h - 1)
        radius = 8
        draw.ellipse((com_x - radius, com_y - radius, com_x + radius, com_y + radius), outline=(40, 220, 255), width=4)
        draw.line((com_x - 14, com_y, com_x + 14, com_y), fill=(40, 220, 255), width=3)
        draw.line((com_x, com_y - 14, com_x, com_y + 14), fill=(40, 220, 255), width=3)

    return pil


def create_demo_visualization(
    image: object,
    analysis_result: Dict[str, object],
    output_path: str | Path,
    title: str,
    expected_issue: Optional[str] = None,
) -> Path:
    """Create a report-ready PNG with image markers, metrics, diagnosis, and suggestions."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGB", (1180, 680), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(30, bold=True)
    section_font = _load_font(22, bold=True)
    body_font = _load_font(18)
    small_font = _load_font(15)

    draw.text((36, 26), title, font=title_font, fill=(25, 35, 50))
    draw.text((36, 62), "红色十字: 图像中心；青色标记: 强度质心", font=small_font, fill=(80, 88, 100))

    image_panel = _render_image_panel(image, size=520)
    canvas.paste(image_panel, (36, 108))

    x = 600
    y = 106
    draw.text((x, y), "诊断摘要", font=section_font, fill=(25, 35, 50))
    y += 36
    summary_lines = [
        f"图像类型: {analysis_result.get('image_type', 'unknown')}",
        f"模型可用: {analysis_result.get('model_available', False)}",
        f"回退模式: {analysis_result.get('fallback', 'rule_based')}",
    ]
    if expected_issue:
        summary_lines.append(f"演示预期问题: {expected_issue}")
    summary_lines.append(f"检测问题: {_format_issues(analysis_result.get('issues', []))}")
    y = _draw_multiline(draw, (x, y), summary_lines, body_font)

    y += 6
    draw.text((x, y), "关键指标", font=section_font, fill=(25, 35, 50))
    y += 34
    y = _draw_multiline(draw, (x, y), _metric_lines(analysis_result.get("metrics", {})), body_font)

    y += 6
    draw.text((x, y), "诊断结论", font=section_font, fill=(25, 35, 50))
    y += 34
    y = _draw_multiline(draw, (x, y), _wrap_text(str(analysis_result.get("diagnosis", "")), 32), body_font)

    y += 6
    draw.text((x, y), "主要建议", font=section_font, fill=(25, 35, 50))
    y += 34
    suggestions = analysis_result.get("suggestions", [])[:3]
    suggestion_lines: List[str] = []
    for idx, suggestion in enumerate(suggestions, 1):
        suggestion_lines.extend(_wrap_text(f"{idx}. {suggestion}", 34))
    _draw_multiline(draw, (x, y), suggestion_lines, body_font)

    canvas.save(output)
    return output
