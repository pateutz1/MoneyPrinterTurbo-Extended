import os
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from moviepy import VideoFileClip
from PIL import Image, ImageDraw, ImageFont

from app.utils import utils


def thumbnail_output_path(video_path: str) -> str:
    path = Path(video_path)
    return str(path.with_name(f"{path.stem}-thumbnail.jpg"))


def frame_timestamp(duration: float) -> float:
    if duration <= 0:
        return 0.0
    return duration / 3


def _resolve_font(font_name: str = ""):
    candidates = []
    if font_name:
        candidates.append(os.path.join(utils.font_dir(), font_name))
    for fallback_name in ("MicrosoftYaHeiBold.ttc", "STHeitiMedium.ttc"):
        candidates.append(os.path.join(utils.font_dir(), fallback_name))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, size=48)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_title(image: Image.Image, title: str, font) -> Image.Image:
    text = " ".join((title or "").split())
    if not text:
        return image

    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    margin = 24
    x = max(margin, (image.width - text_width) // 2)
    y = image.height - text_height - margin
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return image


def generate_thumbnail(
    video_path: str,
    title: str = "",
    font_name: str = "",
) -> Optional[str]:
    if not video_path or not os.path.isfile(video_path):
        return None

    output_path = thumbnail_output_path(video_path)
    clip = None
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration or 0
        if duration <= 0:
            logger.warning("thumbnail generation failed: invalid video duration")
            return None

        timestamp = frame_timestamp(duration)
        frame = clip.get_frame(timestamp)
        image = Image.fromarray(np.asarray(frame))
        image = _draw_title(image, title, _resolve_font(font_name))
        image.save(output_path, format="JPEG", quality=90)
        return output_path
    except Exception as exc:
        logger.warning(f"thumbnail generation failed: {type(exc).__name__}")
        return None
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
