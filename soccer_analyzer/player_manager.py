"""
Simplified player tracking.

Uses ByteTrack tracker IDs directly as player identifiers.
No fake jersey numbers. Maintains team assignments and basic
position history for each tracked player.
"""

import numpy as np
import supervision as sv
from soccer_analyzer.models import PlayerData


class TrackedPlayer:
    def __init__(self, tracker_id, team, bbox):
        self.tracker_id = tracker_id
        self.team = team
        self.bbox = bbox
        self.position = self._feet_position(bbox)
        self.time_since_seen = 0
        self.is_active = True

    def _feet_position(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, y2])

    def update(self, bbox, team=None):
        self.bbox = bbox
        self.position = self._feet_position(bbox)
        self.time_since_seen = 0
        self.is_active = True
        if team is not None and team != 2:
            self.team = team

    def mark_lost(self):
        self.time_since_seen += 1
        self.is_active = False


class PlayerManager:
    def __init__(self):
        self.players = {}  # tracker_id -> TrackedPlayer
        self.max_lost_frames = 90

    def update(self, frame, tracked_detections, team_classifier, lab_frame):
        """Update all player states from new detections."""
        # Mark all as lost (will be un-marked if detected)
        for player in self.players.values():
            player.mark_lost()

        # Get player detections only (class_id == 2)
        player_mask = tracked_detections.class_id == 2
        player_dets = tracked_detections[player_mask]

        if len(player_dets) == 0:
            self._cleanup()
            return

        # Classify teams for all detected players
        bgr_crops = [sv.crop_image(frame, bbox) for bbox in player_dets.xyxy]
        lab_crops = [sv.crop_image(lab_frame, bbox) for bbox in player_dets.xyxy]
        team_predictions = team_classifier.predict(bgr_crops, lab_crops)

        # Update existing or create new players
        for i, tracker_id in enumerate(player_dets.tracker_id):
            bbox = player_dets.xyxy[i]
            team = int(team_predictions[i])

            if tracker_id in self.players:
                self.players[tracker_id].update(bbox, team)
            else:
                self.players[tracker_id] = TrackedPlayer(tracker_id, team, bbox)

        self._cleanup()

    def _cleanup(self):
        """Remove players lost for too long."""
        to_remove = [
            tid for tid, p in self.players.items() if p.time_since_seen > self.max_lost_frames
        ]
        for tid in to_remove:
            del self.players[tid]

    def get_active_players(self):
        """Return PlayerData list for all currently visible players."""
        return [
            PlayerData(
                id=int(p.tracker_id),
                team=p.team if p.team in (0, 1) else None,
                bbox=tuple(p.bbox.astype(int)),
                position=tuple(p.position),
            )
            for p in self.players.values()
            if p.is_active
        ]
