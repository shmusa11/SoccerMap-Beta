"""Readable export annotations, preserving measured/interpolated provenance."""

import cv2
import numpy as np


CLASS_TAGS = {0: "A", 1: "B", -2: "REF", -3: "GK"}


def class_tag(team: int) -> str:
    """Short text class for a team code, so boxes do not rely on colour alone."""
    return CLASS_TAGS.get(team, "?")


def draw_track_box(
    image: np.ndarray, box: tuple, color: tuple, measured: bool, label: str | None = None
) -> None:
    """Draw a measured box or four separated corners for an interpolated image box.

    `label` is an optional class tag (A, B, GK, REF or ?) written above the box.
    """
    x1, y1, x2, y2 = map(int, box)
    if label:
        origin = (x1, max(10, y1 - 4))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
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
        "A/B teams, GK goalkeeper, REF referee, ? unknown | Broken box/ring = interpolated",
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
