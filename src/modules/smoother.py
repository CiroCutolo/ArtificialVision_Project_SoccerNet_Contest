import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger("BoxSmoother")

class OnlineBoxSmoother:
    """
    Applies Exponential Moving Average (EMA) to bounding box coordinates in an online fashion.
    
    This module stabilizes the visual output (jitter reduction) without accessing future frames,
    making it suitable for real-time applications or strict online protocols.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Initialize the smoother.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1).
                   Higher value = Less smoothing, faster response (more jitter).
                   Lower value = More smoothing, slower response (more lag).
        """
        self.alpha = alpha
        self.tracks: Dict[int, Tuple[float, float, float, float]] = {}

    def update(self, tracks: List[Any]) -> List[Any]:
        """
        Process a batch of tracks for the current frame and apply smoothing.

        Args:
            tracks: List of tracks, where each track is a list/array.
                    Expected format: [x1, y1, x2, y2, track_id, conf, ...others]

        Returns:
            List of tracks with updated [x1, y1, x2, y2] coordinates.
        """
        smoothed_tracks = []
        current_frame_ids = set()

        for t in tracks:
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            track_id = int(t[4])
            current_frame_ids.add(track_id)

            w = x2 - x1
            h = y2 - y1
            cx = x1 + (w / 2)
            cy = y1 + h  

            if track_id not in self.tracks:
                self.tracks[track_id] = (cx, cy, w, h)
                s_cx, s_cy, s_w, s_h = cx, cy, w, h
            else:
                prev_cx, prev_cy, prev_w, prev_h = self.tracks[track_id]
                
                s_cx = self.alpha * cx + (1 - self.alpha) * prev_cx
                s_cy = self.alpha * cy + (1 - self.alpha) * prev_cy
                s_w  = self.alpha * w  + (1 - self.alpha) * prev_w
                s_h  = self.alpha * h  + (1 - self.alpha) * prev_h
                
                self.tracks[track_id] = (s_cx, s_cy, s_w, s_h)

            s_x1 = s_cx - (s_w / 2)
            s_y1 = s_cy - s_h
            s_x2 = s_x1 + s_w
            s_y2 = s_y1 + s_h

            new_track = list(t)
            new_track[0] = s_x1
            new_track[1] = s_y1
            new_track[2] = s_x2
            new_track[3] = s_y2
            
            smoothed_tracks.append(new_track)

        existing_ids = list(self.tracks.keys())
        for tid in existing_ids:
            if tid not in current_frame_ids:
                del self.tracks[tid]

        return smoothed_tracks