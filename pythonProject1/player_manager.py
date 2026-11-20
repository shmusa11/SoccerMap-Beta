# football_analyzer/player_manager.py

from collections import defaultdict, deque
import numpy as np
from data_models import PlayerData
import supervision as sv


class OptimizedPlayer:
    def __init__(self, tracker_id: int, team: int, jersey_number: int, initial_bbox: np.ndarray):
        self.tracker_id = tracker_id
        self.team = team
        self.jersey_number = jersey_number
        self.bbox = initial_bbox
        # --- FIX: Position is now the player's feet (bottom-center of bbox) ---
        self.position = self._get_ground_position(initial_bbox)
        self.position_history = deque([self.position], maxlen=30)
        self.time_since_seen = 0
        self.is_active = True
        self.confidence_score = 1.0

    def _get_ground_position(self, bbox):
        """Calculates the bottom-center of the bounding box, representing the player's feet."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, y2])

    def update(self, tracker_id: int, bbox: np.ndarray):
        self.tracker_id = tracker_id
        self.bbox = bbox
        # --- FIX: Update position using the correct ground-level anchor point ---
        self.position = self._get_ground_position(bbox)
        self.position_history.append(self.position)
        self.time_since_seen = 0
        self.is_active = True
        self.confidence_score = min(1.0, self.confidence_score + 0.1)

    def mark_as_lost(self):
        self.time_since_seen += 1
        self.is_active = False
        self.confidence_score = max(0.0, self.confidence_score - 0.05)


class OptimizedPlayerManager:
    def __init__(self, pitch_mapper):
        self.pitch_mapper = pitch_mapper
        self.team_players = {0: {}, 1: {}}
        self.available_jerseys = {0: list(range(1, 21)), 1: list(range(1, 21))}
        self.frame_count = 0
        self.reidentification_frequency = 5
        self.max_lost_frames = 60

    def initial_assignment(self, frame, player_detections, team_classifier, hsv_frame):
        print("Performing initial jersey assignment...")
        crops_bgr = [sv.crop_image(frame, bbox) for bbox in player_detections.xyxy]
        crops_hsv = [sv.crop_image(hsv_frame, bbox) for bbox in player_detections.xyxy]
        team_predictions = team_classifier.predict_teams(crops_bgr, crops_hsv)

        players_by_team = defaultdict(list)
        for i, tracker_id in enumerate(player_detections.tracker_id):
            team = team_predictions[i]
            if team not in [0, 1]: continue
            info = {'tracker_id': tracker_id, 'bbox': player_detections.xyxy[i]}
            players_by_team[team].append(info)

        for team, players in players_by_team.items():
            for player_info in players:
                if self.available_jerseys[team]:
                    jersey_num = self.available_jerseys[team].pop(0)
                    self._create_player(player_info, team, jersey_num)
        print(f"Initial assignment complete. Team 0: {len(self.team_players[0])}, Team 1: {len(self.team_players[1])}")

    def _create_player(self, player_info, team, jersey_number):
        player_obj = OptimizedPlayer(player_info['tracker_id'], team, jersey_number, player_info['bbox'])
        self.team_players[team][jersey_number] = player_obj

    def update(self, frame, tracked_detections, team_classifier, hsv_frame):
        self.frame_count += 1
        for team in self.team_players:
            for player in self.team_players[team].values():
                player.mark_as_lost()

        self._update_existing_players(tracked_detections)

        if self.frame_count % self.reidentification_frequency == 0:
            unassigned_detections = self._get_unassigned_detections(tracked_detections)
            if unassigned_detections:
                self._handle_unassigned_players(frame, unassigned_detections, team_classifier, hsv_frame)

        if self.frame_count % 30 == 0: self._cleanup_old_players()

    def _update_existing_players(self, tracked_detections):
        known_ids = {p.tracker_id: (team, j_num) for team, players in self.team_players.items() for j_num, p in
                     players.items()}
        for i, tracker_id in enumerate(tracked_detections.tracker_id):
            if tracked_detections.class_id[i] == 2 and tracker_id in known_ids:
                team, j_num = known_ids[tracker_id]
                self.team_players[team][j_num].update(tracker_id, tracked_detections.xyxy[i])

    def _get_unassigned_detections(self, tracked_detections):
        known_ids = {p.tracker_id for players in self.team_players.values() for p in players.values()}
        return [{'tracker_id': tid, 'bbox': bbox} for tid, bbox, cid in
                zip(tracked_detections.tracker_id, tracked_detections.xyxy, tracked_detections.class_id)
                if cid == 2 and tid not in known_ids]

    def _handle_unassigned_players(self, frame, unassigned_detections, team_classifier, hsv_frame):
        if not unassigned_detections: return
        bboxes = [det['bbox'] for det in unassigned_detections]
        crops_bgr = [sv.crop_image(frame, bbox) for bbox in bboxes]
        crops_hsv = [sv.crop_image(hsv_frame, bbox) for bbox in bboxes]
        team_predictions = team_classifier.predict_teams(crops_bgr, crops_hsv)
        for i, det in enumerate(unassigned_detections):
            team = team_predictions[i]
            if team in [0, 1] and self.available_jerseys[team]:
                jersey_num = self.available_jerseys[team].pop(0)
                self._create_player(det, team, jersey_num)

    def _cleanup_old_players(self):
        for team, players in list(self.team_players.items()):
            for jersey, player in list(players.items()):
                if player.time_since_seen > self.max_lost_frames:
                    del self.team_players[team][jersey]
                    self.available_jerseys[team].append(jersey)
                    self.available_jerseys[team].sort()

    def get_player_data_list(self):
        data = []
        for players in self.team_players.values():
            for player in players.values():
                if player.is_active:
                    data.append(PlayerData(
                        id=player.jersey_number, team=player.team,
                        bbox=tuple(player.bbox.astype(int)),
                        position=tuple(player.position),
                        confidence=player.confidence_score
                    ))
        return data