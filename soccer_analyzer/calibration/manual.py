"""User-picked field landmarks, in source pixels and corner-origin pitch metres.

Four correspondences determine a plane; they do not establish accuracy. Callers must show
an overlay for review and verify the camera view before reusing the resulting mapping.
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel


def landmarks(pitch: PitchModel) -> list[dict]:
    """Named line intersections. Box dimensions stay fixed when pitch size changes."""
    length, width = pitch.length, pitch.width
    points = []
    for side, x in (("left", 0.0), ("right", length)):
        for edge, y in (("top", 0.0), ("bottom", width)):
            points.append(
                {"id": f"{side}_{edge}", "label": f"{side.title()} {edge} corner", "x": x, "y": y}
            )
        for box, depth, half_width in (("penalty", 16.5, 20.16), ("goal", 5.5, 9.16)):
            inner = depth if side == "left" else length - depth
            for edge, y in (("top", width / 2 - half_width), ("bottom", width / 2 + half_width)):
                for end, px in (("goal_line", x), ("inner", inner)):
                    points.append(
                        {
                            "id": f"{side}_{box}_{edge}_{end}",
                            "label": f"{side.title()} {box} box · {edge}, {'goal line' if end == 'goal_line' else 'field side'}",
                            "x": px,
                            "y": y,
                        }
                    )
    for edge, y in (("top", 0.0), ("bottom", width), ("center", width / 2)):
        points.append(
            {"id": f"halfway_{edge}", "label": f"Halfway line · {edge}", "x": length / 2, "y": y}
        )
    return points


def solve_manual(points: list[dict], pitch: PitchModel, image_size: tuple[int, int]) -> np.ndarray:
    """Fit image-to-pitch H from exactly four distinct, non-collinear named landmarks.

    Raises ValueError for ambiguous geometry, out-of-image pixels or a folded pitch. Point
    order is arbitrary; each point names its field landmark, so mirrored camera views work.
    """
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("Place four different field markings before saving the setup.")
    known = {p["id"]: (p["x"], p["y"]) for p in landmarks(pitch)}
    image, world, used = [], [], set()
    width, height = image_size
    for point in points:
        if not isinstance(point, dict) or set(point) != {"landmark", "x", "y"}:
            raise ValueError("Each point needs a field marking and its video position.")
        name, x, y = point["landmark"], point["x"], point["y"]
        if not isinstance(name, str) or name not in known or name in used:
            raise ValueError("Choose a different field marking for each point.")
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in (x, y)) or not (
            0 <= x < width and 0 <= y < height
        ):
            raise ValueError("Place each point inside the video image.")
        used.add(name)
        image.append((x, y))
        world.append(known[name])
    image, world = np.asarray(image, np.float64), np.asarray(world, np.float64)
    for values, scale in ((image, max(width, height)), (world, max(pitch.length, pitch.width))):
        normalized = values / scale
        for a, b, c in combinations(normalized, 3):
            u, v = b - a, c - a
            area = abs(float(u[0] * v[1] - u[1] * v[0]))
            if area < 0.0005:
                raise ValueError(
                    "Spread the points across the field. No three can be on the same line."
                )
    inverse, _ = cv2.findHomography(world, image, 0)
    if inverse is None or not np.isfinite(inverse).all():
        raise ValueError("These points do not define a usable field. Check the selected markings.")
    try:
        matrix = np.linalg.inv(inverse)
    except np.linalg.LinAlgError as exc:
        raise ValueError("These points are too close together to map the field.") from exc
    # A horizon through the pitch or crossed correspondence order gives a folded projection.
    corners = np.array(
        [[0, 0, 1], [pitch.length, 0, 1], [pitch.length, pitch.width, 1], [0, pitch.width, 1]]
    )
    denominators = corners @ inverse[2]
    if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
        raise ValueError("The field folds across the image. Check which marking each point names.")
    return matrix / np.linalg.norm(matrix)
