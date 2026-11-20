"""Readable export annotations, preserving measured/interpolated provenance."""

import cv2
import numpy as np


def draw_track_box(image: np.ndarray, box: tuple, color: tuple, measured: bool) -> None:
    """Draw a measured box or four separated corners for an interpolated image box."""
    x1, y1, x2, y2 = map(int, box)
    if measured:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return
    span = max(1, min(10, (x2 - x1) // 4, (y2 - y1) // 4))
    for x, dx in ((x1, span), (x2, -span)):
        for y, dy in ((y1, span), (y2, -span)):
            cv2.line(image, (x, y), (x + dx, y), color, 2)
            cv2.line(image, (x, y), (x, y + dy), color, 2)


def review_caption(image: np.ndarray, source_frame: int, valid: bool) -> np.ndarray:
    """Append a caption outside the footage; availability is not a confidence score."""
    height, width = image.shape[:2]
    band = max(42, round(width * 0.075))
    result = cv2.copyMakeBorder(image, 0, band, 0, 0, cv2.BORDER_CONSTANT, value=(32, 45, 38))
    lines = [
        f"Source frame {source_frame} | {'Estimated field map' if valid else 'Map unavailable'}",
        "Green = estimated field | Broken boxes / rings = interpolated",
    ]
    padding = max(4, round(width * 0.015))
    for row, text in enumerate(lines):
        scale = max(0.3, width / 1200)
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1.0, (width - 2 * padding) / max(1, text_width))
        cv2.putText(
            result,
            text,
            (padding, height + round(band * (0.38 + 0.43 * row))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 248, 245),
            1,
            cv2.LINE_AA,
        )
    return result
