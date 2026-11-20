# football_analyzer/team_classifier_enhanced.py

import numpy as np
import cv2
import supervision as sv
from sklearn.cluster import KMeans


class EnhancedTeamClassifier:
    def __init__(self, device='cuda'):
        self.is_fitted = False
        self.team_dominant_colors_hsv = {}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_frame(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        hsv_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2HSV)
        return bgr_enhanced, hsv_enhanced

    def extract_dominant_color(self, crop, hsv_crop):
        if crop.size == 0 or hsv_crop.size == 0: return None

        h, w = hsv_crop.shape[:2]
        if h < 10 or w < 5: return None
        torso = hsv_crop[int(h * 0.2):int(h * 0.8), int(w * 0.1):int(w * 0.9)]
        pixels = torso.reshape(-1, 3)

        # More aggressive filtering for jersey colors
        mask = (pixels[:, 1] > 40) & (pixels[:, 2] > 50) & (pixels[:, 2] < 225)
        # Exclude green (grass)
        mask &= ((pixels[:, 0] < 35) | (pixels[:, 0] > 85))

        filtered_pixels = pixels[mask]
        if len(filtered_pixels) < 100: return None  # Require more pixels for a confident sample

        try:
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
            kmeans.fit(filtered_pixels)
            dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
            return dominant_color
        except:
            return None

    def fit_automatically(self, frame, detections, hsv_frame):
        print("Running automated team classification...")
        dominant_colors = []
        for bbox in detections.xyxy:
            hsv_crop = sv.crop_image(hsv_frame, bbox)
            bgr_crop = sv.crop_image(frame, bbox)
            color = self.extract_dominant_color(bgr_crop, hsv_crop)
            if color is not None:
                dominant_colors.append(color)

        if len(dominant_colors) < 4:
            print("Could not extract enough dominant colors to classify teams.")
            return False

        try:
            kmeans = KMeans(n_clusters=2, n_init='auto', random_state=42)
            labels = kmeans.fit_predict(dominant_colors)

            # Use brightness (Value channel) to consistently label Team 0 and Team 1
            # For Red vs White, Red will have a lower V value.
            v_mean_0 = np.mean([c[2] for c, l in zip(dominant_colors, labels) if l == 0])
            v_mean_1 = np.mean([c[2] for c, l in zip(dominant_colors, labels) if l == 1])

            darker_team_label, brighter_team_label = (0, 1) if v_mean_0 < v_mean_1 else (1, 0)

            self.team_dominant_colors_hsv[0] = kmeans.cluster_centers_[darker_team_label]
            self.team_dominant_colors_hsv[1] = kmeans.cluster_centers_[brighter_team_label]

            print(f"Team 0 (Darker) Center: {self.team_dominant_colors_hsv[0]}")
            print(f"Team 1 (Brighter) Center: {self.team_dominant_colors_hsv[1]}")
            self.is_fitted = True
            return True
        except:
            return False

    def predict_teams(self, player_crops, hsv_crops):
        if not self.is_fitted: return np.array([2] * len(player_crops))  # Return 'unknown'

        predictions = []
        for bgr_crop, hsv_crop in zip(player_crops, hsv_crops):
            color = self.extract_dominant_color(bgr_crop, hsv_crop)
            if color is None:
                predictions.append(2)  # Unknown
            else:
                dist0 = np.linalg.norm(color - self.team_dominant_colors_hsv[0])
                dist1 = np.linalg.norm(color - self.team_dominant_colors_hsv[1])
                predictions.append(0 if dist0 < dist1 else 1)
        return np.array(predictions)