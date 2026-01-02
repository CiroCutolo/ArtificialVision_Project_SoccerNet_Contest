import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Union

logger = logging.getLogger("AV_project_behavior")

class ROIHandler:
    """Handles loading and geometric verification of Regions of Interest (ROI)."""
    
    def __init__(self, json_path: Union[str, Path], img_width: int, img_height: int):
        """Initialize the ROI Handler.

        Args:
            json_path: Path to the JSON file containing ROI definitions.
            img_width: Width of the video frame.
            img_height: Height of the video frame.
        """
        self.width = img_width
        self.height = img_height
        self.rois = self._load_rois(Path(json_path))
        
    def _load_rois(self, json_path: Path) -> Dict[int, Tuple[float, float, float, float]]:
        """Load ROIs from JSON and convert relative coordinates to absolute pixels.

        Args:
            json_path: Path object to the JSON file.

        Returns:
            Dictionary mapping ROI IDs to tuple coordinates (x1, y1, x2, y2).

        Raises:
            FileNotFoundError: If the JSON file does not exist.
        """
        if not json_path.exists():
            raise FileNotFoundError(f"[ ERROR | ROI JSON not found: {json_path} ]")
            
        logger.info(f"[ DEBUG | Loading ROIs from {json_path.name} (Scale: {self.width}x{self.height}) ]")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        roi_map = {}
        for rid, key in [(1, "roi1"), (2, "roi2")]:
            if key in data:
                r = data[key]
                x1 = float(r["x"]) * self.width
                y1 = float(r["y"]) * self.height
                x2 = (float(r["x"]) + float(r["width"])) * self.width
                y2 = (float(r["y"]) + float(r["height"])) * self.height
                roi_map[rid] = (x1, y1, x2, y2)
                
        return roi_map

    def is_inside(self, x: float, y: float, roi_id: int) -> bool:
        """Check if a point is inside a specific ROI.

        Args:
            x: X coordinate of the point.
            y: Y coordinate of the point.
            roi_id: ID of the ROI to check against.

        Returns:
            True if the point is inside the ROI, False otherwise.
        """
        if roi_id not in self.rois:
            return False
        x1, y1, x2, y2 = self.rois[roi_id]
        return x1 <= x <= x2 and y1 <= y <= y2


class BehaviorAnalyzer:
    """Calculates player counts within ROIs frame by frame."""

    def __init__(self, config: dict):
        """Initialize the Behavior Analyzer.

        Args:
            config: Configuration dictionary loaded from YAML.
        """
        self.cfg = config
        
        self.roi_handler = ROIHandler(
            json_path=config["paths"]["roi_json_path"],
            img_width=int(config["video"]["width"]),
            img_height=int(config["video"]["height"])
        )
        
        self.counts = defaultdict(int)
        self.max_frame = 0

    def process_tracks(self, tracks: List[Tuple[int, float, float, float, float]]):
        """Process a batch list of tracking detections.

        Args:
            tracks: List of tuples containing (frame, x, y, w, h).
        """
        self.counts.clear()
        self.max_frame = 0
        
        for frame, x, y, w, h in tracks:
            self.update(frame, x, y, w, h)

    def update(self, frame: int, x: float, y: float, w: float, h: float):
        """Update counts for a single detection at runtime.

        Args:
            frame: Frame index.
            x: Bounding box top-left X.
            y: Bounding box top-left Y.
            w: Bounding box width.
            h: Bounding box height.
        """
        self.max_frame = max(self.max_frame, frame)
        
        px = x + w / 2.0
        py = y + h

        for rid in (1, 2):
            if self.roi_handler.is_inside(px, py, rid):
                self.counts[(frame, rid)] += 1

    def save_results(self, output_path: Path):
        """Export counts to a CSV file in SoccerNet format.

        Args:
            output_path: Destination path for the result file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[ DEBUG | Saving behavior results to: {output_path} ]")
        with output_path.open("w", encoding="utf-8") as f:
            for frame in range(1, self.max_frame + 1):
                for rid in (1, 2):
                    count = self.counts.get((frame, rid), 0)
                    f.write(f"{frame},{rid},{count}\n")