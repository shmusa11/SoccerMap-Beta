# football_analyzer/data_models.py

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class PlayerData:
    """Enhanced player data structure"""
    id: int
    team: Optional[int]
    bbox: Tuple[int, int, int, int]
    position: Tuple[float, float]
    field_position: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    player_type: str = "player"


@dataclass
class BallData:
    """Ball tracking data"""
    id: int
    bbox: Tuple[int, int, int, int]
    position: Tuple[float, float]
    field_position: Optional[Tuple[float, float]] = None
    confidence: float = 0.0


@dataclass
class FrameAnalytics:
    """Complete frame analysis"""
    frame_number: int
    players: List[PlayerData]
    ball: Optional[BallData]
    team_possession: Optional[int]
    possession_percentage: Dict[int, float] = field(default_factory=dict)
    team_stats: Dict[str, any] = field(default_factory=dict)
    # --- NEW: Field to hold the screen coordinates of detected landmarks for visualization ---
    detected_landmarks_screen_positions: Optional[List[Tuple[float, float]]] = None