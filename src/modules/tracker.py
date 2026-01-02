import numpy as np
from pathlib import Path
from boxmot import BotSort, BoostTrack
# Se usi anche BoostTrack importalo, altrimenti puoi rimuoverlo
# from boxmot import BoostTrack 

class PlayerTracker:
    """Wrapper for BoTSORT and BoostTrack trackers."""

    def __init__(self, config):
        """
        Initialize the tracker.

        Args:
            config (dict): Configuration dictionary with keys:
                - tracking.tracker_type (str): "botsort" or "boosttrack"
                - tracking.reid_model_path (str): path to reid weights
                - system.device (str): device identifier
                - system.fp16 (bool): use half precision for model
                - tracking.track_buffer (int)
                - tracking.match_thresh (float)
                - tracking.track_high_thresh (float)
                - tracking.track_low_thresh (float)
                - tracking.new_track_thresh (float)
                - tracking.appearance_thresh (float)
                - tracking.proximity_thresh (float)
                - tracking.cmc_method (str)
                - tracking.reid (bool)

        Raises:
            ValueError: if tracker_type is not supported.
        """
        self.tracker_type = config["tracking"]["tracker_type"]
        self.model_weights = Path(config["tracking"]["reid_model_path"])
        self.device = config["system"]["device"]
        self.fp16 = config["system"]["fp16"]

        self.track_buffer = config["tracking"]["track_buffer"]
        self.match_thresh = config["tracking"]["match_thresh"]
        self.track_high_thresh = config["tracking"]["track_high_thresh"]
        self.track_low_thresh = config["tracking"]["track_low_thresh"]
        self.new_track_thresh = config["tracking"]["new_track_thresh"]
        self.iou_threshold = config["tracking"]["iou_threshold"]

        self.appearance_thresh = config["tracking"]["appearance_thresh"]
        self.proximity_thresh = config["tracking"]["proximity_thresh"]

        self.cmc_method = config["tracking"]["cmc_method"]
        self.reid = config["tracking"]["reid"]

        if self.tracker_type == "botsort":
            print(f"[ Tracker | Init BoTSORT | Buffer: {self.track_buffer} | CMC: {self.cmc_method} | Single Class ]")

            self.tracker = BotSort(
                reid_weights=self.model_weights,
                device=self.device,
                half=self.fp16,
                track_buffer=self.track_buffer,
                max_age=self.track_buffer,
                nr_classes=1,
                per_class=False,
                track_high_thresh=self.track_high_thresh,
                track_low_thresh=self.track_low_thresh,
                new_track_thresh=self.new_track_thresh,
                match_thresh=self.match_thresh,
                iou_threshold=self.iou_threshold,
                appearance_thresh=self.appearance_thresh,
                proximity_thresh=self.proximity_thresh,
                cmc_method=self.cmc_method,
                with_reid=self.reid
            )

        elif self.tracker_type == "boosttrack":
            self.tracker = BoostTrack(
                model_weights=self.model_weights,
                device=self.device,
                fp16=self.fp16,
                track_buffer=self.track_buffer,
                max_age=self.track_buffer,
                match_thresh=self.match_thresh,
                track_high_thresh=self.track_high_thresh
            )
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")

    def update(self, detections, frame):
        """
        Update tracker with detections for the current frame.

        Args:
            detections (ndarray): Nx6 array [x1, y1, x2, y2, score, class_id] or empty array.
            frame (ndarray): Current image (BGR) used for appearance-based tracking.

        Returns:
            tracked_objects: Tracker-specific output describing active tracks.
        """
        if len(detections) == 0:
            tracked_objects = self.tracker.update(np.empty((0, 6)), frame)
        else:
            tracked_objects = self.tracker.update(detections, frame)

        return tracked_objects