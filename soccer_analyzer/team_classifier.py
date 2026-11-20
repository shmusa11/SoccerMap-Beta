"""
Team classification by jersey color.

Two modes:
  - Auto: KMeans K=3 in LAB color space (team A, team B, other)
  - Manual: user provides team colors, classifier matches against them

Uses LAB color space (perceptually uniform) instead of HSV.
Masks out green (grass) pixels before extracting dominant color.
Adds a confidence threshold so uncertain players are labeled "unknown".
"""

import numpy as np
import cv2
import supervision as sv
from sklearn.cluster import KMeans


# Named colors for manual mode: name -> LAB approximate center
# These are rough - the classifier will use them as starting points
NAMED_COLORS_BGR = {
    "red": (0, 0, 200),
    "blue": (200, 0, 0),
    "white": (240, 240, 240),
    "black": (30, 30, 30),
    "yellow": (0, 220, 220),
    "green": (0, 180, 0),
    "orange": (0, 140, 255),
    "maroon": (0, 0, 128),
    "navy": (128, 0, 0),
    "purple": (128, 0, 128),
    "pink": (180, 105, 255),
    "gray": (128, 128, 128),
}


class TeamClassifier:
    def __init__(self, team_a_color=None, team_b_color=None):
        """
        Args:
            team_a_color: color name string (e.g. "red") or None for auto-detect
            team_b_color: color name string (e.g. "white") or None for auto-detect
        """
        self.is_fitted = False
        self.team_centers_lab = {}  # {0: lab_center, 1: lab_center}
        self.max_distance = 60  # LAB distance threshold for "unknown"

        # CLAHE for preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # If manual colors specified, set them up
        if team_a_color and team_b_color:
            self._set_manual_colors(team_a_color, team_b_color)

    def _set_manual_colors(self, color_a, color_b):
        """Set team colors from named colors."""
        bgr_a = NAMED_COLORS_BGR.get(color_a.lower())
        bgr_b = NAMED_COLORS_BGR.get(color_b.lower())
        if bgr_a is None or bgr_b is None:
            valid = ", ".join(NAMED_COLORS_BGR.keys())
            print(f"Unknown color name. Valid colors: {valid}")
            return

        # Convert BGR to LAB
        pixel_a = np.uint8([[list(bgr_a)]])
        pixel_b = np.uint8([[list(bgr_b)]])
        self.team_centers_lab[0] = cv2.cvtColor(pixel_a, cv2.COLOR_BGR2LAB)[0][0].astype(float)
        self.team_centers_lab[1] = cv2.cvtColor(pixel_b, cv2.COLOR_BGR2LAB)[0][0].astype(float)
        self.is_fitted = True
        print(f"Team 0: {color_a}, Team 1: {color_b}")

    def preprocess_frame(self, frame):
        """Apply CLAHE enhancement, return (bgr_enhanced, lab_enhanced)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return bgr_enhanced, lab_enhanced

    def _extract_jersey_color(self, bgr_crop, lab_crop):
        """Extract dominant jersey color from a player crop in LAB space.

        Uses the upper body region, masks out green (grass) pixels,
        and finds the dominant color via KMeans.
        """
        if bgr_crop.size == 0 or lab_crop.size == 0:
            return None

        h, w = bgr_crop.shape[:2]
        if h < 10 or w < 5:
            return None

        # Upper body: top 20-65% height, middle 80% width (jersey region)
        y1, y2 = int(h * 0.2), int(h * 0.65)
        x1, x2 = int(w * 0.1), int(w * 0.9)
        jersey_bgr = bgr_crop[y1:y2, x1:x2]
        jersey_lab = lab_crop[y1:y2, x1:x2]

        if jersey_bgr.size == 0:
            return None

        # Convert jersey region to HSV to filter grass
        jersey_hsv = cv2.cvtColor(jersey_bgr, cv2.COLOR_BGR2HSV)
        hsv_pixels = jersey_hsv.reshape(-1, 3)
        lab_pixels = jersey_lab.reshape(-1, 3).astype(float)

        # Mask: exclude green/grass pixels (hue 35-85 in HSV)
        # Also exclude very dark pixels (shadows) and very bright pixels (sky)
        hue = hsv_pixels[:, 0]
        sat = hsv_pixels[:, 1]
        val = hsv_pixels[:, 2]
        mask = ((hue < 35) | (hue > 85)) & (val > 40) & (val < 240) & (sat > 20)

        filtered = lab_pixels[mask]
        if len(filtered) < 50:
            return None

        # Find dominant color with KMeans K=2 (jersey + any remaining noise)
        try:
            km = KMeans(n_clusters=min(2, len(filtered)), n_init="auto", random_state=42)
            km.fit(filtered)
            # Pick the largest cluster's center
            counts = np.bincount(km.labels_)
            dominant = km.cluster_centers_[np.argmax(counts)]
            return dominant
        except Exception:
            return None

    def fit_auto(self, frame, detections, lab_frame):
        """Automatically detect team colors from a frame with many players.

        Uses KMeans K=3 to find: team A, team B, and "other" (referee/noise).
        The two largest clusters become the teams.
        """
        colors = []
        for bbox in detections.xyxy:
            bgr_crop = sv.crop_image(frame, bbox)
            lab_crop = sv.crop_image(lab_frame, bbox)
            color = self._extract_jersey_color(bgr_crop, lab_crop)
            if color is not None:
                colors.append(color)

        if len(colors) < 6:
            print(f"Only {len(colors)} valid color samples. Need at least 6.")
            return False

        try:
            km = KMeans(n_clusters=3, n_init="auto", random_state=42)
            labels = km.fit_predict(colors)
            counts = np.bincount(labels, minlength=3)

            # The two largest clusters are the teams
            top2 = np.argsort(counts)[-2:]
            # Assign consistently: team 0 = cluster with lower L (darker)
            center_a = km.cluster_centers_[top2[0]]
            center_b = km.cluster_centers_[top2[1]]
            if center_a[0] > center_b[0]:  # L channel: higher = lighter
                center_a, center_b = center_b, center_a

            self.team_centers_lab[0] = center_a  # Darker team
            self.team_centers_lab[1] = center_b  # Lighter team
            self.is_fitted = True

            print(
                f"Auto-detected teams: Team 0 (darker) LAB={center_a.astype(int)}, "
                f"Team 1 (lighter) LAB={center_b.astype(int)}"
            )
            return True
        except Exception as e:
            print(f"Auto-detection failed: {e}")
            return False

    def predict(self, bgr_crops, lab_crops):
        """Classify each player crop as team 0, team 1, or unknown (2).

        Returns numpy array of team IDs.
        """
        if not self.is_fitted:
            return np.array([2] * len(bgr_crops))

        predictions = []
        for bgr_crop, lab_crop in zip(bgr_crops, lab_crops):
            color = self._extract_jersey_color(bgr_crop, lab_crop)
            if color is None:
                predictions.append(2)
                continue

            dist0 = np.linalg.norm(color - self.team_centers_lab[0])
            dist1 = np.linalg.norm(color - self.team_centers_lab[1])
            min_dist = min(dist0, dist1)

            if min_dist > self.max_distance:
                predictions.append(2)  # Too far from both teams = unknown
            elif dist0 < dist1:
                predictions.append(0)
            else:
                predictions.append(1)

        return np.array(predictions)
