"""
Interactive pitch calibration tool.

Usage:
    python calibrate.py video.mov [output.json]

Opens a window showing the first frame of the video. Click on known pitch
points to establish the camera-to-pitch mapping.

Orientation convention (looking from camera):
    - Far touchline = TOP of screen (y=0)
    - Near touchline = BOTTOM of screen / camera side (y=68)
    - Left goal = LEFT on screen (x=0)
    - Right goal = RIGHT on screen (x=105)

Controls:
    1-7     Select a center-area pitch point
    8-0,a   Select a penalty box point
    b-f     Select a corner flag point
    c       Enter custom pitch coordinates
    u       Undo last point
    n       Next frame (if current frame isn't good)
    s       Save calibration and quit
    q       Quit without saving
"""

import sys
import os
import json
import cv2
import numpy as np


# Standard pitch dimensions (meters)
DEFAULT_PITCH_WIDTH = 105.0
DEFAULT_PITCH_HEIGHT = 68.0

# Center circle radius = 9.15m. Center spot at (52.5, 34).
# Labels describe what you SEE on screen from the camera.
PITCH_PRESETS = {
    # --- Center area (most useful for stands-filmed games) ---
    "1": (52.5, 34, "Center spot"),
    "2": (52.5, 0, "Centerline @ far sideline (top of screen)"),
    "3": (52.5, 68, "Centerline @ near sideline (bottom of screen)"),
    "4": (43.35, 34, "Circle widest — LEFT on screen"),
    "5": (61.65, 34, "Circle widest — RIGHT on screen"),
    "6": (52.5, 24.85, "Circle @ centerline — UP on screen"),
    "7": (52.5, 43.15, "Circle @ centerline — DOWN on screen"),
    # --- Penalty boxes ---
    "8": (16.5, 13.84, "Left box corner (far)"),
    "9": (16.5, 54.16, "Left box corner (near)"),
    "0": (88.5, 13.84, "Right box corner (far)"),
    "a": (88.5, 54.16, "Right box corner (near)"),
    # --- Corner flags ---
    "b": (0, 0, "Corner flag: left-far"),
    "d": (105, 0, "Corner flag: right-far"),
    "e": (0, 68, "Corner flag: left-near"),
    "f": (105, 68, "Corner flag: right-near"),
}


class Calibrator:
    def __init__(self, frame, pitch_w=DEFAULT_PITCH_WIDTH, pitch_h=DEFAULT_PITCH_HEIGHT):
        self.original_frame = frame.copy()
        self.frame_h, self.frame_w = frame.shape[:2]
        self.pitch_w = pitch_w
        self.pitch_h = pitch_h
        self.panel_w = 330

        self.points = []
        self.selected_preset = None
        self.custom_mode = False
        self.message = "Press 1-7 to select a pitch point, then click on the frame"
        self._frame_display_w = 1
        self._frame_display_h = 1

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Ignore clicks on the panel
        if x > self._frame_display_w:
            return

        # Scale from display coords to original frame coords
        orig_x = x * (self.frame_w / self._frame_display_w)
        orig_y = y * (self.frame_h / self._frame_display_h)

        if self.selected_preset is not None:
            px, py, label = PITCH_PRESETS[self.selected_preset]
            self.points.append(
                {
                    "frame": [round(orig_x, 1), round(orig_y, 1)],
                    "pitch": [px, py],
                    "label": label,
                }
            )
            self.message = f"Marked: {label}. {len(self.points)} points total."
            self.selected_preset = None
        elif self.custom_mode:
            print(f"\nClicked at frame position ({orig_x:.0f}, {orig_y:.0f})")
            try:
                coords = input("Enter pitch coordinates (x y): ").strip().split()
                px, py = float(coords[0]), float(coords[1])
                label = input("Label (optional): ").strip() or "Custom"
                self.points.append(
                    {
                        "frame": [round(orig_x, 1), round(orig_y, 1)],
                        "pitch": [px, py],
                        "label": label,
                    }
                )
                self.message = f"Marked: {label} ({px}, {py}). {len(self.points)} points total."
            except (ValueError, IndexError):
                self.message = "Invalid input. Try again."
            self.custom_mode = False

    def draw_ui(self):
        """Draw the calibration UI: video frame + legend panel."""
        # Scale frame to fit (max 1000px wide)
        max_frame_w = 1000
        scale = min(1.0, max_frame_w / self.frame_w)
        display_w = int(self.frame_w * scale)
        display_h = int(self.frame_h * scale)

        # Ensure minimum height so panel + preview fit
        min_h = 650
        if display_h < min_h:
            display_h = min_h
            display_w = int(self.frame_w * (display_h / self.frame_h))

        display = cv2.resize(self.original_frame.copy(), (display_w, display_h))

        # Draw marked points on frame
        for i, pt in enumerate(self.points):
            fx, fy = pt["frame"]
            dx = int(fx / self.frame_w * display_w)
            dy = int(fy / self.frame_h * display_h)
            cv2.circle(display, (dx, dy), 6, (0, 255, 0), -1)
            cv2.circle(display, (dx, dy), 6, (0, 0, 0), 2)
            cv2.putText(
                display,
                f"{i + 1}",
                (dx + 10, dy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        # --- Panel ---
        panel = np.full((display_h, self.panel_w, 3), 40, dtype=np.uint8)
        y = 22

        cv2.putText(
            panel, "PITCH CALIBRATION", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        y += 20
        cv2.putText(
            panel,
            f"Points: {len(self.points)} (need 4+)",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )
        y += 24

        # Section: Center area
        cv2.putText(
            panel, "CENTER AREA:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1
        )
        y += 16
        for key in ["1", "2", "3", "4", "5", "6", "7"]:
            px, py, label = PITCH_PRESETS[key]
            color = (0, 255, 0) if self.selected_preset == key else (190, 190, 190)
            cv2.putText(panel, f"[{key}] {label}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y += 15

        y += 6
        cv2.putText(
            panel, "PENALTY BOXES:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1
        )
        y += 16
        for key in ["8", "9", "0", "a"]:
            px, py, label = PITCH_PRESETS[key]
            color = (0, 255, 0) if self.selected_preset == key else (160, 160, 160)
            cv2.putText(panel, f"[{key}] {label}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y += 15

        y += 6
        cv2.putText(panel, "CORNERS:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 255), 1)
        y += 16
        for key in ["b", "d", "e", "f"]:
            px, py, label = PITCH_PRESETS[key]
            color = (0, 255, 0) if self.selected_preset == key else (140, 140, 140)
            cv2.putText(panel, f"[{key}] {label}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y += 15

        y += 10
        cv2.putText(
            panel,
            "[c] Custom  [u] Undo  [n] Next",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (180, 180, 180),
            1,
        )
        y += 15
        cv2.putText(
            panel, "[s] Save  [q] Quit", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1
        )

        # Pitch preview (always show, with or without points)
        y += 22
        cv2.putText(panel, "PITCH MAP:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
        y += 6
        preview = self._draw_pitch_preview()
        ph, pw = preview.shape[:2]
        if y + ph < display_h - 25:
            panel[y : y + ph, 8 : 8 + pw] = preview

        # Combine
        combined = np.hstack([display, panel])

        # Status bar at bottom
        cv2.putText(
            combined,
            self.message,
            (10, display_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
        )

        return combined, display_w, display_h

    def _draw_pitch_preview(self):
        """Draw a mini pitch diagram showing calibration points."""
        pw, ph = 310, 200
        img = np.full((ph, pw, 3), (0, 100, 0), dtype=np.uint8)

        margin = 5
        field_w = pw - 2 * margin
        field_h = ph - 2 * margin
        sx = field_w / self.pitch_w
        sy = field_h / self.pitch_h

        def pitch_to_px(mx, my):
            return int(margin + mx * sx), int(margin + my * sy)

        # Field outline
        cv2.rectangle(img, pitch_to_px(0, 0), pitch_to_px(105, 68), (255, 255, 255), 1)
        # Center line
        cv2.line(img, pitch_to_px(52.5, 0), pitch_to_px(52.5, 68), (255, 255, 255), 1)
        # Center circle
        cx, cy = pitch_to_px(52.5, 34)
        cv2.circle(img, (cx, cy), int(9.15 * sy), (255, 255, 255), 1)
        # Center spot
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
        # Penalty boxes
        cv2.rectangle(img, pitch_to_px(0, 13.84), pitch_to_px(16.5, 54.16), (255, 255, 255), 1)
        cv2.rectangle(img, pitch_to_px(88.5, 13.84), pitch_to_px(105, 54.16), (255, 255, 255), 1)

        # Labels: "LEFT GOAL" and "RIGHT GOAL"
        cv2.putText(
            img,
            "LEFT",
            pitch_to_px(2, 36)[0:1] and (margin + 2, ph // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            img,
            "RIGHT",
            (pw - margin - 38, ph // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.25,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            img,
            "FAR (top of screen)",
            (pw // 2 - 55, margin + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.22,
            (180, 180, 180),
            1,
        )
        cv2.putText(
            img,
            "NEAR (camera side)",
            (pw // 2 - 50, ph - margin - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.22,
            (180, 180, 180),
            1,
        )

        # Draw calibration points
        for pt in self.points:
            px_m, py_m = pt["pitch"]
            x, y = pitch_to_px(px_m, py_m)
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(img, (x, y), 4, (0, 0, 0), 1)

        # Show camera view region if we have a valid homography
        if len(self.points) >= 4:
            try:
                H = self._compute_homography()
                if H is not None:
                    corners = np.array(
                        [
                            [[0, 0]],
                            [[self.frame_w, 0]],
                            [[self.frame_w, self.frame_h]],
                            [[0, self.frame_h]],
                        ],
                        dtype=np.float32,
                    )
                    pitched = cv2.perspectiveTransform(corners, H)
                    pts = []
                    for p in pitched:
                        x, y = pitch_to_px(p[0][0], p[0][1])
                        pts.append([x, y])
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(img, [pts], True, (0, 200, 255), 1)
            except Exception:
                pass

        return img

    def _compute_homography(self):
        if len(self.points) < 4:
            return None
        frame_pts = np.array([p["frame"] for p in self.points], dtype=np.float32)
        pitch_pts = np.array([p["pitch"] for p in self.points], dtype=np.float32)
        H, mask = cv2.findHomography(frame_pts, pitch_pts, cv2.RANSAC, 5.0)
        return H

    def save(self, filepath):
        H = self._compute_homography()
        if H is None:
            print("Need at least 4 points to save calibration.")
            return False

        data = {
            "pitch_width": self.pitch_w,
            "pitch_height": self.pitch_h,
            "frame_width": self.frame_w,
            "frame_height": self.frame_h,
            "points": self.points,
            "homography": H.tolist(),
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nCalibration saved to {filepath} ({len(self.points)} points)")
        return True

    def run(self, output_path, cap):
        """Run the interactive calibration loop."""
        window_name = "Pitch Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        first_draw = True
        while True:
            combined, frame_disp_w, frame_disp_h = self.draw_ui()
            self._frame_display_w = frame_disp_w
            self._frame_display_h = frame_disp_h
            img_h, img_w = combined.shape[:2]

            if first_draw:
                cv2.resizeWindow(window_name, img_w, img_h)
                first_draw = False

            cv2.setMouseCallback(window_name, self.mouse_callback)
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue

            if key == ord("q"):
                print("Quit without saving.")
                break
            elif key == ord("s"):
                if self.save(output_path):
                    break
                else:
                    self.message = "Need at least 4 points! Keep clicking."
            elif key == ord("u"):
                if self.points:
                    removed = self.points.pop()
                    self.message = f"Removed: {removed['label']}. {len(self.points)} points left."
                else:
                    self.message = "Nothing to undo."
            elif key == ord("n"):
                ret, frame = cap.read()
                if ret:
                    self.original_frame = frame.copy()
                    self.message = "Advanced to next frame."
                else:
                    self.message = "No more frames."
            elif key == ord("c"):
                self.custom_mode = True
                self.selected_preset = None
                self.message = (
                    "Custom mode: click on the frame, then enter coordinates in terminal."
                )
            elif chr(key) in PITCH_PRESETS:
                self.selected_preset = chr(key)
                label = PITCH_PRESETS[self.selected_preset][2]
                self.message = f"Selected: {label}. Click on that point in the frame."
                self.custom_mode = False

        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <video_path> [output.json]")
        print("\nOpens a window to interactively calibrate pitch mapping.")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(
            "calibrations", os.path.splitext(os.path.basename(video_path))[0] + ".json"
        )
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        sys.exit(1)

    print(f"Video: {video_path}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Output: {output_path}")
    print()
    print("=== ORIENTATION GUIDE ===")
    print("The pitch map assumes you're filming from the NEAR sideline:")
    print("  - TOP of your screen    = FAR sideline (y=0)")
    print("  - BOTTOM of your screen = NEAR sideline / camera (y=68)")
    print("  - LEFT on screen        = left goal (x=0)")
    print("  - RIGHT on screen       = right goal (x=105)")
    print()
    print("For the center circle, identify these 4 points:")
    print("  [4] The leftmost point of the circle")
    print("  [5] The rightmost point of the circle")
    print("  [6] Where the circle crosses the center line going UP")
    print("  [7] Where the circle crosses the center line going DOWN")
    print()

    calibrator = Calibrator(frame)
    calibrator.run(output_path, cap)
    cap.release()


if __name__ == "__main__":
    main()
